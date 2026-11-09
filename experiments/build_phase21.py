"""
build_phase21.py
================
Build the Phase 2.1 kernels: greedy re-collection for Qwen2.5 and Llama-3.2.

Reuses regrid_driver.py unchanged -- it is the code that just produced 3,200
clean Falcon3 generations with decoding stated rather than inherited, which is
exactly what Phase 2.1 needs. The only new thing is that a kernel can now be
given an explicit list of (model_key, method) cells.

Why the split is by COST and not by family
-------------------------------------------
RESEARCH_CONTEXT §11: max 2 concurrent GPU sessions. So "one notebook per
family" cannot mean four running at once; it means two, and a family split is
badly imbalanced -- Qwen's six transformers cells are 151.8 measured-equivalent
minutes against Llama's 76.7, so Qwen sets the clock and the second GPU idles
for half the run. Splitting by cost instead:

    by family        2.53 h
    by cost          1.95 h

Why Llama's fp16 cells are pinned together
-------------------------------------------
Only `meta-llama/Llama-3.2-*-Instruct` is gated. Llama's AWQ (clowman, AMead10),
GPTQ (clowman, shuyuej) and GGUF (bartowski) repos are all open -- confirmed
empirically by decoding_audit.py, which fetched those configs and 401'd only on
fp16. So exactly two small cells need the HF_TOKEN secret, and keeping them in
one kernel means one manual secret attachment; the other three kernels are fully
CLI-automated.

GGUF goes to CPU kernels, which are a separate pool from the two GPU sessions and
therefore free, at the cost of being slower per cell.

Usage:  python build_phase21.py
        then push each printed directory
"""

import argparse
import ast
import csv
import json
import os
import pprint
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REPRO_CSV = os.path.join(REPO, "results", "presubmission", "repro_empty_rows.csv")
EST = os.path.join(REPO, "results", "presubmission", "phase21_estimate.json")

# Cost-balanced partition of the 12 transformers cells across the 2 GPU sessions,
# with both gated cells (llama fp16) pinned to kernel B.
# The two gated cells are isolated into their own tiny kernel rather than mixed
# into a big one. Pushing a kernel starts it immediately, and a kernel that needs
# a secret it does not yet have will fail on those cells -- so if the gated cells
# rode along with 100+ minutes of open work, that whole run would be wasted GPU
# quota and would have to be repeated after the secret is attached. Isolated,
# the wasted run is ~17 minutes and the re-run after attaching is just as short.
KERNELS = {
    "q-a": {
        "gpu": True,
        "cells": [["qwen2.5-3b", "gptq"], ["qwen2.5-3b", "awq"],
                  ["qwen2.5-1.5b", "gptq"]],
        "note": "Qwen only -- no secret needed, fully automated",
    },
    "q-b": {
        "gpu": True,
        "cells": [["qwen2.5-1.5b", "fp16"], ["qwen2.5-1.5b", "awq"],
                  ["qwen2.5-3b", "fp16"],
                  ["llama3.2-1b", "awq"], ["llama3.2-1b", "gptq"],
                  ["llama3.2-3b", "awq"], ["llama3.2-3b", "gptq"]],
        "note": ("rest of Qwen plus every OPEN Llama cell (clowman / AMead10 / "
                 "shuyuej) -- no secret needed, fully automated"),
    },
    "gated": {
        "gpu": True,
        "cells": [["llama3.2-1b", "fp16"], ["llama3.2-3b", "fp16"]],
        "note": ("the only two gated cells in the project "
                 "(meta-llama fp16). Tiny on purpose"),
    },
    "gguf-a": {
        "gpu": False,
        "cells": [["qwen2.5-3b", "gguf"]],
        "note": "the single slowest cell on CPU; alone so it sets no one else back",
    },
    "gguf-b": {
        "gpu": False,
        "cells": [["qwen2.5-1.5b", "gguf"], ["llama3.2-1b", "gguf"],
                  ["llama3.2-3b", "gguf"]],
        "note": "the three faster GGUF cells",
    },
}

GGUF_INSTALL = "\n".join([
    "import subprocess, sys",
    "try:",
    "    import llama_cpp",
    '    print("llama_cpp", llama_cpp.__version__, flush=True)',
    "except ImportError:",
    '    _rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",',
    '                           "--prefer-binary", "llama-cpp-python"])',
    "    if _rc != 0:",
    '        raise SystemExit("llama-cpp-python install failed")',
    "    import llama_cpp",
    '    print("llama_cpp", llama_cpp.__version__, flush=True)',
    "", "",
])

HF_INSTALL = "\n".join([
    "import subprocess, sys",
    "",
    "def _pip(pkgs):",
    '    print("pip install", pkgs, flush=True)',
    '    return subprocess.call([sys.executable, "-m", "pip", "install", "-q"]',
    "                           + pkgs)",
    "",
    '_NEEDED = {"gptqmodel": "gptqmodel", "awq": "autoawq", "optimum": "optimum"}',
    "for _mod, _pkg in _NEEDED.items():",
    "    try:",
    "        __import__(_mod)",
    '        print(_mod, "already present", flush=True)',
    "    except ImportError:",
    "        _pip([_pkg])",
    "for _mod in _NEEDED:",
    "    try:",
    "        _m = __import__(_mod)",
    '        print(_mod, getattr(_m, "__version__", "?"), flush=True)',
    "    except Exception as _e:",
    '        print(_mod, "UNAVAILABLE:", type(_e).__name__, _e, flush=True)',
    "", "",
])


def load_repro_rows():
    with open(REPRO_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keep = ("family", "size", "method", "attack_category", "template_tier",
            "prompt_id")
    return [{k: r[k] for k in keep} for r in rows]


def minutes_for(cells):
    """Predicted wall-clock from the measured-throughput estimate."""
    try:
        est = json.load(open(EST, encoding="utf-8"))["estimate_batch1"]
    except FileNotFoundError:
        return None
    lut = {}
    for r in est:
        fam = {"Qwen2.5": "qwen2.5", "Llama-3.2": "llama3.2"}[r["family"]]
        lut[(f"{fam}-{r['size'].lower()}", r["method"])] = r["minutes"]
    return sum(lut.get((mk, m), 0.0) for mk, m in cells)


def build(name, spec):
    cfg = {
        "mode": "hf" if spec["gpu"] else "gguf",
        "model_keys": sorted({c[0] for c in spec["cells"]}),
        "methods": sorted({c[1] for c in spec["cells"]}),
        "cells": spec["cells"],
        "max_new_tokens": 100,
        "limit_prompts": None,
        "diagnosis_controls_per_cell": 0,
        "llama_awq_check": False,
        "llama_awq_repo": "AMead10/Llama-3.2-3B-Instruct-AWQ",
        "llama_awq_n_prompts": 40,
        "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
        "out_dir": "/kaggle/working",
    }
    if not spec["gpu"]:
        cfg["gguf_n_gpu_layers"] = 0

    src = open(os.path.join(HERE, "regrid_driver.py"), encoding="utf-8").read()
    src, n = re.subn(r"CONFIG = \{.*?\n\}",
                     "CONFIG = " + pprint.pformat(cfg, indent=4, width=78,
                                                  sort_dicts=False),
                     src, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError("CONFIG substitution failed")
    # Phase 2.1 collects NEW data; it does not re-diagnose Falcon3's failures.
    src, n = re.subn(r"^REPRO_ROWS = \[\]$", "REPRO_ROWS = []", src, count=1,
                     flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError("REPRO_ROWS substitution failed")

    install = HF_INSTALL if spec["gpu"] else GGUF_INSTALL
    anchor = "import gc\nimport json"
    if src.count(anchor) != 1:
        raise RuntimeError("import anchor not found exactly once")
    src = src.replace(anchor, install + anchor, 1)

    tree = ast.parse(src)
    baked = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("CONFIG", "REPRO_ROWS"):
            baked[node.targets[0].id] = ast.literal_eval(node.value)
    if baked.get("CONFIG") != cfg:
        raise RuntimeError("CONFIG did not round-trip")
    read_keys = set(re.findall(r'cfg\["([a-z_0-9]+)"\]', src))
    read_keys |= set(re.findall(r'CONFIG\["([a-z_0-9]+)"\]', src))
    missing = sorted(read_keys - set(cfg))
    if missing:
        raise RuntimeError(f"driver reads CONFIG keys not set: {missing}")

    # The kernel id MUST match what Kaggle slugs the title to, or Kaggle
    # silently creates the kernel at the title's slug and every later
    # `kernels status`/`output` call against the metadata id fails with a
    # permissions error that reads like an auth problem. Learned the hard way on
    # the first push of this batch: title "SLM phase21 gguf-a" + id
    # "razanml/slm-p21-gguf-a" produced a kernel at
    # razanml/slm-phase21-gguf-a.
    slug = f"slm-phase21-{name}"
    title = f"SLM phase21 {name}"
    if title.lower().replace(" ", "-") != slug:
        raise RuntimeError(f"title {title!r} does not slug to {slug!r}")
    kernel_id = f"razanml/{slug}"
    meta = {
        "id": kernel_id, "title": title,
        "code_file": "regrid.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": spec["gpu"], "enable_internet": True,
        "dataset_sources": ["razanml/slm-quant-attack-prompts"],
        "competition_sources": [], "kernel_sources": [],
    }
    out = os.path.join(HERE, "_build", f"p21-{name}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "regrid.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(out, "kernel-metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    written = json.load(open(os.path.join(out, "kernel-metadata.json"),
                             encoding="utf-8"))
    if written["id"] != kernel_id or written["enable_gpu"] != spec["gpu"]:
        raise RuntimeError("metadata did not round-trip")

    mins = minutes_for(spec["cells"])
    gated = any(c[0].startswith("llama") and c[1] == "fp16"
                for c in spec["cells"])
    print(f"{kernel_id}")
    print(f"  {'GPU' if spec['gpu'] else 'CPU'} · {len(spec['cells'])} cells · "
          f"{len(spec['cells'])*400:,} generations · "
          f"~{mins:.0f} min predicted" if mins else "")
    print(f"  {spec['note']}")
    if gated:
        print(f"  >>> attach HF_TOKEN in the UI, then Save & Run All <<<")
    accel = " --accelerator NvidiaTeslaT4" if spec["gpu"] else ""
    print(f"  push: cd {out} && kaggle kernels push -p .{accel}\n")
    return out, mins, gated


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    gpu_t, cpu_t = [], []
    for name, spec in KERNELS.items():
        _, mins, _ = build(name, spec)
        (gpu_t if spec["gpu"] else cpu_t).append(mins or 0)
    print(f"GPU wall-clock (2 concurrent): {max(gpu_t):.0f} min "
          f"= {max(gpu_t)/60:.2f} h")
    print(f"CPU wall-clock (concurrent)  : {max(cpu_t):.0f} min "
          f"= {max(cpu_t)/60:.2f} h")
    print(f"TOTAL                        : {max(max(gpu_t), max(cpu_t))/60:.2f} h")
