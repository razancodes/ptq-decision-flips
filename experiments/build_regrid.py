"""
build_regrid.py
===============
Build the Falcon3 re-run kernel (task 1.2) from regrid_driver.py.

Two modes, split by dependency risk and by device need:

  --mode hf    fp16 + awq + gptq, both Falcon3 sizes, plus the diagnosis phase
               and the Llama-3.2-3B-AWQ degeneracy check. 2,400 generations.
               GPU kernel. transformers 5.x routes AWQ *and* GPTQ through
               gptqmodel/optimum, neither of which is in the Kaggle image, so
               both are installed at the top -- version 2 of this kernel lost
               all four quantized cells to that ImportError.

  --mode gguf  gguf, both Falcon3 sizes. 800 generations. CPU kernel: llama.cpp
               needs no GPU, so this costs no GPU quota and installs a prebuilt
               CPU wheel in seconds instead of compiling CUDA. What is being
               fixed in the GGUF cells is the sampling temperature (llama.cpp's
               create_chat_completion defaults to 0.2), not the device.

Both modes emit the same schema and the same repaired config, so their CSVs
concatenate into one 3,200-row grid.

Build discipline, following build_reseed.py: CONFIG replaced wholesale and
round-tripped through ast.literal_eval; every CONFIG key the driver reads must
exist in the build (a missing key cost the mechanism kernel a whole session);
written metadata re-read and its kernel id asserted.

Usage:  python build_regrid.py --mode hf
        cd experiments/_build/regrid-hf && kaggle kernels push -p . \
            --accelerator NvidiaTeslaT4
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

MODE_METHODS = {
    "hf": ["fp16", "awq", "gptq"],
    "gguf": ["gguf"],
}

# Exactly the set notebook 03 installed, which is the environment where these
# load paths are known to work. An earlier version of this kernel installed only
# `optimum` and `gptqmodel`, inferred from transformers 5.x's error string --
# but the original pipeline also installed `autoawq`, and inferring a dependency
# set from an error message is weaker evidence than the environment that
# actually ran the experiment.
HF_INSTALL = "\n".join([
    "import subprocess, sys",
    "",
    "def _pip(pkgs):",
    '    print("pip install", pkgs, flush=True)',
    '    return subprocess.call([sys.executable, "-m", "pip", "install", "-q"]',
    "                           + pkgs)",
    "",
    "# import name -> pip name, where they differ",
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

GGUF_INSTALL = "\n".join([
    "import subprocess, sys",
    "try:",
    "    import llama_cpp",
    '    print("llama_cpp", llama_cpp.__version__, flush=True)',
    "except ImportError:",
    '    _rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",',
    '                           "--prefer-binary", "llama-cpp-python"])',
    "    if _rc != 0:",
    '        raise SystemExit("llama-cpp-python install failed; the GGUF cells "',
    '                         "are reported as not computed, not measured by "',
    '                         "some other path")',
    "    import llama_cpp",
    '    print("llama_cpp", llama_cpp.__version__, flush=True)',
    "", "",
])


def load_repro_rows():
    """The Study A rows that returned an empty reply, as plain dicts."""
    with open(REPRO_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{REPRO_CSV} is empty -- run forensics.py first")
    keep = ("family", "size", "method", "attack_category", "template_tier",
            "prompt_id")
    return [{k: r[k] for k in keep} for r in rows]


def build(mode):
    methods = MODE_METHODS[mode]
    cfg = {
        "mode": mode,
        "model_keys": ["falcon3-1b", "falcon3-3b"],
        "methods": methods,
        "max_new_tokens": 100,
        "limit_prompts": None,
        "diagnosis_controls_per_cell": 10,
        "llama_awq_check": mode == "hf",
        "llama_awq_repo": "AMead10/Llama-3.2-3B-Instruct-AWQ",
        "llama_awq_n_prompts": 40,
        "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
        "out_dir": "/kaggle/working",
    }
    if mode == "gguf":
        cfg["gguf_n_gpu_layers"] = 0
    repro = load_repro_rows()

    src = open(os.path.join(HERE, "regrid_driver.py"), encoding="utf-8").read()

    src, n = re.subn(
        r"CONFIG = \{.*?\n\}",
        "CONFIG = " + pprint.pformat(cfg, indent=4, width=78, sort_dicts=False),
        src, count=1, flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("CONFIG substitution failed")

    src, n = re.subn(
        r"^REPRO_ROWS = \[\]$",
        lambda _m: "REPRO_ROWS = " + pprint.pformat(repro, indent=4, width=100),
        src, count=1, flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("REPRO_ROWS substitution failed")

    install = HF_INSTALL if mode == "hf" else GGUF_INSTALL
    anchor = "import gc\nimport json"
    if src.count(anchor) != 1:
        raise RuntimeError("import anchor not found exactly once")
    src = src.replace(anchor, install + anchor, 1)

    tree = ast.parse(src)  # syntax gate
    baked = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("CONFIG", "REPRO_ROWS"):
            baked[node.targets[0].id] = ast.literal_eval(node.value)
    if baked.get("CONFIG") != cfg:
        raise RuntimeError(f"CONFIG did not round-trip:\n {baked.get('CONFIG')}")
    if baked.get("REPRO_ROWS") != repro:
        raise RuntimeError("REPRO_ROWS did not round-trip")

    read_keys = set(re.findall(r'cfg\["([a-z_0-9]+)"\]', src))
    read_keys |= set(re.findall(r'CONFIG\["([a-z_0-9]+)"\]', src))
    missing = sorted(read_keys - set(cfg))
    if missing:
        raise RuntimeError(f"driver reads CONFIG keys the build does not set: "
                           f"{missing}")

    kernel_id = f"razanml/slm-regrid-{mode}"
    meta = {
        "id": kernel_id,
        "title": f"SLM regrid {mode}",
        "code_file": "regrid.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": mode != "gguf",
        "enable_internet": True,
        "dataset_sources": ["razanml/slm-quant-attack-prompts"],
        "competition_sources": [],
        "kernel_sources": [],
    }

    out = os.path.join(HERE, "_build", f"regrid-{mode}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "regrid.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    written = json.load(open(os.path.join(out, "kernel-metadata.json"),
                             encoding="utf-8"))
    if written["id"] != kernel_id:
        raise RuntimeError(f"metadata id mismatch: {written['id']} != {kernel_id}")
    if mode != "gguf" and not written["enable_gpu"]:
        raise RuntimeError("the hf regrid kernel needs a GPU")
    if mode == "gguf" and written["enable_gpu"]:
        raise RuntimeError("the gguf regrid kernel is meant to be CPU-only")

    n_gen = len(cfg["model_keys"]) * len(methods) * 400
    print(f"Built {out}")
    print(f"  kernel id  : {written['id']}")
    print(f"  gpu        : {written['enable_gpu']}")
    print(f"  methods    : {', '.join(methods)}")
    print(f"  repro rows : {len(repro)} baked in")
    print(f"  {len(cfg['model_keys'])} models x {len(methods)} methods x 400 "
          f"prompts = {n_gen:,} generations")
    accel = "" if mode == "gguf" else " --accelerator NvidiaTeslaT4"
    print(f"  push with  : cd {out} && kaggle kernels push -p .{accel}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_METHODS), default="hf")
    build(ap.parse_args().mode)
