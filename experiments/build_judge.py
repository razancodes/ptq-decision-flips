"""
build_judge.py
==============
Build the re-labelling kernel (razanml/slm-judge) from judge_driver.py.

The rubrics are NOT retyped here. They are extracted from
legacy/notebooks/04_classify.ipynb and passed through ast.literal_eval, so the string
this kernel sends is the RUNTIME string the original pass sent -- backslash line
continuations resolved, whitespace preserved. The SHA-256 of each resolved string
is baked into the kernel, which refuses to run if it does not match. That is the
only way to be sure a re-labelling pass is comparable to the primary labels.

Same discipline as build_reseed.py: CONFIG is replaced wholesale, round-tripped
through ast.literal_eval, and the written metadata is re-read and its kernel id
asserted -- never regex surgery on a live kernel.

Usage:  python build_judge.py
        kaggle kernels push -p experiments/_build/judge
"""

import argparse
import ast
import hashlib
import json
import os
import pprint
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOTEBOOK = os.path.join(REPO, "legacy", "notebooks", "04_classify.ipynb")

KERNEL_ID = "razanml/slm-judge"
CODE_FILE = "judge-driver.py"
QUEUE_DATASET = "razanml/slm-judge-queue"


def extract_rubrics():
    """Return {name: runtime_string} for the two rubrics in 04_classify.ipynb."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"]
                       if c["cell_type"] == "code")
    out = {}
    for name in ("RUBRIC_JAILBREAK", "RUBRIC_INJECTION"):
        m = re.search(name + r' = """(.*?)"""', source, re.DOTALL)
        if not m:
            raise RuntimeError(f"{name} not found in {NOTEBOOK}")
        # literal_eval on the full literal, so Python's own escape handling
        # (notably the '\<newline>' continuations) produces the runtime string.
        out[name] = ast.literal_eval('"""' + m.group(1) + '"""')
    return out


def build():
    rubrics = extract_rubrics()
    sha = {
        "jailbreak": hashlib.sha256(rubrics["RUBRIC_JAILBREAK"].encode()).hexdigest(),
        "injection": hashlib.sha256(rubrics["RUBRIC_INJECTION"].encode()).hexdigest(),
    }

    cfg = {
        "judge_model": "deepseek-ai/DeepSeek-V4-Flash",
        "base_url": "https://router.huggingface.co/v1",
        "max_workers": 8,
        "max_tokens": 350,
        "temperature": 0,
        "retries": 3,
        "retry_sleep": 2,
        "checkpoint_every": 250,
        "out_dir": "/kaggle/working",
    }

    src = open(os.path.join(HERE, "judge_driver.py"), encoding="utf-8").read()

    src, n = re.subn(
        r"CONFIG = \{.*?\n\}",
        "CONFIG = " + pprint.pformat(cfg, indent=4, width=78, sort_dicts=False),
        src, count=1, flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("CONFIG substitution failed")

    rubric_block = (
        f"RUBRIC_JAILBREAK = {rubrics['RUBRIC_JAILBREAK']!r}\n"
        f"RUBRIC_INJECTION = {rubrics['RUBRIC_INJECTION']!r}\n"
        f"RUBRIC_SHA = {sha!r}"
    )
    src, n = re.subn(
        r"RUBRIC_JAILBREAK = \"\"\nRUBRIC_INJECTION = \"\"\nRUBRIC_SHA = \{[^}]*\}",
        lambda _m: rubric_block,
        src, count=1,
    )
    if n != 1:
        raise RuntimeError("RUBRIC substitution failed")

    # Round-trip every baked literal through literal_eval: a bare name (the
    # json.dumps 'null' trap) or a mangled string would be caught here, before
    # a kernel burns a run on it.
    tree = ast.parse(src)
    baked = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CONFIG", "RUBRIC_JAILBREAK", "RUBRIC_INJECTION", "RUBRIC_SHA"):
                baked[name] = ast.literal_eval(node.value)
    if baked.get("CONFIG") != cfg:
        raise RuntimeError(f"CONFIG did not round-trip: {baked.get('CONFIG')}")
    for key, name in (("jailbreak", "RUBRIC_JAILBREAK"), ("injection", "RUBRIC_INJECTION")):
        if baked.get(name) != rubrics[name]:
            raise RuntimeError(f"{name} did not round-trip")
        got = hashlib.sha256(baked[name].encode()).hexdigest()
        if got != baked["RUBRIC_SHA"][key]:
            raise RuntimeError(f"{name} sha mismatch after bake: {got}")

    meta = {
        "id": KERNEL_ID,
        "title": "SLM judge",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [QUEUE_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }

    out = os.path.join(HERE, "_build", "judge")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, CODE_FILE), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    written = json.load(open(os.path.join(out, "kernel-metadata.json"), encoding="utf-8"))
    if written["id"] != KERNEL_ID:
        raise RuntimeError(f"metadata id mismatch: {written['id']} != {KERNEL_ID}")
    if written["enable_gpu"]:
        raise RuntimeError("judge kernel must not request a GPU")

    print(f"Built {out}")
    print(f"  kernel id   : {written['id']}")
    print(f"  gpu         : {written['enable_gpu']}  internet: {written['enable_internet']}")
    print(f"  dataset     : {written['dataset_sources']}")
    print(f"  rubric sha  : jailbreak={sha['jailbreak'][:16]} injection={sha['injection'][:16]}")
    print(f"  rubric len  : jailbreak={len(rubrics['RUBRIC_JAILBREAK'])} "
          f"injection={len(rubrics['RUBRIC_INJECTION'])}")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build()
