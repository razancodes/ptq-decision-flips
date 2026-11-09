"""
build_reseed.py
===============
Build the E4 re-seed control kernel (generation only; classification is done
locally with a classifier trained on the existing human labels).

Two modes, split by repo gating rather than by science:

  --mode open   Qwen2.5 x2, Falcon3 x2   -> kernel razanml/slm-reseed-open
  --mode llama  Llama-3.2 x2             -> kernel razanml/slm-reseed-llama

Llama needs its own notebook because a freshly-pushed kernel has no HF_TOKEN
secret attached, and `kaggle kernels push` RESETS any secret attachment. So the
gated models must live in a kernel the user can attach a secret to and then
commit from the UI, without us pushing over it afterwards.

Both modes emit identical code and CONFIG shape, so the two runs are directly
comparable and their CSVs concatenate locally.

Usage:  python build_reseed.py --mode open
        python build_reseed.py --mode llama
"""

import argparse
import ast
import json
import os
import pprint
import re

HERE = os.path.dirname(os.path.abspath(__file__))

MODE_MODELS = {
    "open": ["qwen2.5-1.5b", "qwen2.5-3b", "falcon3-1b", "falcon3-3b"],
    "llama": ["llama3.2-1b", "llama3.2-3b"],
}


def build(mode):
    cfg = {
        "models": MODE_MODELS[mode],
        "seeds": [1000, 1001, 1002],
        "limit_prompts": None,
        "max_new_tokens": 100,
        "batch_size": 16,
        "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
        "out_dir": "/kaggle/working",
    }

    src = open(os.path.join(HERE, "reseed_driver.py"), encoding="utf-8").read()
    src, n = re.subn(
        r"CONFIG = \{.*?\n\}",
        "CONFIG = " + pprint.pformat(cfg, indent=4, width=78, sort_dicts=False),
        src, count=1, flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("CONFIG substitution failed")

    # literal_eval rejects bare names, so a json-style null/true can't slip through.
    baked = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets):
            baked = ast.literal_eval(node.value)
    if baked != cfg:
        raise RuntimeError(f"CONFIG did not round-trip:\n wanted {cfg}\n got    {baked}")

    kernel_id = f"razanml/slm-reseed-{mode}"
    meta = {
        "id": kernel_id,
        "title": f"SLM reseed {mode}",
        "code_file": "reseed-control.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["razanml/slm-quant-attack-prompts"],
        "competition_sources": [],
        "kernel_sources": [],
    }

    out = os.path.join(HERE, "_build", f"reseed-{mode}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "reseed-control.py"), "w", encoding="utf-8") as f:
        f.write(src)
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Verify what was actually written -- the previous version of this script
    # patched the metadata by string substitution, the substitution silently did
    # not match, and a Llama build was pushed over a running open-model kernel.
    written = json.load(open(os.path.join(out, "kernel-metadata.json"), encoding="utf-8"))
    if written["id"] != kernel_id:
        raise RuntimeError(f"metadata id mismatch: {written['id']} != {kernel_id}")

    n_gen = len(cfg["models"]) * len(cfg["seeds"]) * 400
    print(f"Built {out}")
    print(f"  kernel id : {written['id']}")
    print(f"  models    : {', '.join(cfg['models'])}")
    print(f"  {len(cfg['models'])} models x {len(cfg['seeds'])} seeds x 400 prompts "
          f"= {n_gen:,} generations")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_MODELS), default="open")
    build(ap.parse_args().mode)
