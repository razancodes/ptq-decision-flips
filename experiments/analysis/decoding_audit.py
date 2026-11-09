"""
decoding_audit.py  --  which Falcon3 cells were actually decoded greedily?

§4.7 rests on one sentence: "Because Falcon3 decodes greedily, every
fp16->quantized flip is 100% attributable to quantization." §4.6 verified that
claim by checking `model.generation_config.do_sample` on the fp16 checkpoints and
by observing that Falcon3's fp16 re-seeds were 100% identical.

Two things that check did not cover, and both matter:

1. **The GGUF rows never went through transformers at all.** They were produced by
   `model_loaders.generate_reply_gguf()`, which calls
   `llm.create_chat_completion(messages=..., max_tokens=...)` with no sampling
   arguments. llama-cpp-python's `create_chat_completion` defaults to
   **temperature=0.2, top_p=0.95, top_k=40, min_p=0.05** (verified against the
   library source, 2026-08-27) -- so the GGUF column was SAMPLED, not greedy.
   `create_completion` defaults to 0.8, which is not the path used, but the point
   stands either way: the default is not 0.

2. **A quantized repo ships its own generation_config.json.** AWQ and GPTQ
   checkpoints are separate repos, and `generate_reply()` passes no sampling args,
   so each inherits whatever its own repo declares. That was never checked.

This script downloads only the config files (a few kB each, no weights) and
reports what each Falcon3 checkpoint actually declares.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")
sys.path.insert(0, REPO)

REPOS = {}
_REG = None


def _registry():
    global _REG
    if _REG is None:
        sys.path.insert(0, REPO)
        import model_registry
        _REG = model_registry
    return _REG


def _build_repos():
    """Every (family, size, method) in the project, from the registry itself."""
    reg = _registry()
    out = {}
    for key, entry in reg.MODELS.items():
        fam, size = entry["family"], entry["size"]
        for method in ("fp16", "gptq", "awq", "gguf"):
            m = entry.get(method)
            if m and m.get("repo"):
                out[(fam, size, method)] = m["repo"]
    return out


# Verified against llama_cpp/llama.py on 2026-08-27.
LLAMA_CPP_CHAT_DEFAULTS = {"temperature": 0.2, "top_p": 0.95, "top_k": 40,
                           "min_p": 0.05, "repeat_penalty": 1.0}


def main():
    from huggingface_hub import hf_hub_download
    repos = _build_repos()
    rows = []
    for (fam, size, method), repo in sorted(repos.items()):
        if method == "gguf":
            continue
        rec = {"family": fam, "size": size, "method": method, "repo": repo,
               "engine": "transformers"}
        try:
            path = hf_hub_download(repo_id=repo, filename="generation_config.json")
            cfg = json.load(open(path, encoding="utf-8"))
            rec.update({
                "do_sample": cfg.get("do_sample"),
                "temperature": cfg.get("temperature"),
                "top_p": cfg.get("top_p"),
                "top_k": cfg.get("top_k"),
                "greedy": cfg.get("do_sample") in (None, False),
                "raw": cfg,
            })
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}", "greedy": None})
        rows.append(rec)

    for (fam, size, method), repo in sorted(repos.items()):
        if method != "gguf":
            continue
        rows.append({"family": fam, "size": size, "method": "gguf",
                     "repo": repo,
                     "engine": "llama.cpp via create_chat_completion",
                     "do_sample": "n/a (llama.cpp)",
                     "temperature": LLAMA_CPP_CHAT_DEFAULTS["temperature"],
                     "top_p": LLAMA_CPP_CHAT_DEFAULTS["top_p"],
                     "top_k": LLAMA_CPP_CHAT_DEFAULTS["top_k"],
                     "greedy": False,
                     "raw": {"library_defaults": LLAMA_CPP_CHAT_DEFAULTS,
                             "note": ("generate_reply_gguf passes only "
                                      "max_tokens, so these defaults applied")}})

    out = {"cells": rows,
           "llama_cpp_chat_defaults": LLAMA_CPP_CHAT_DEFAULTS,
           "verified": "2026-08-27, against llama_cpp/llama.py source"}
    greedy_cells = [r for r in rows if r.get("greedy") is True]
    sampled_cells = [r for r in rows if r.get("greedy") is False]
    out["summary"] = {
        "n_greedy_cells": len(greedy_cells),
        "n_sampled_cells": len(sampled_cells),
        "sampled": [f"{r['family']}-{r['size']}-{r['method']}"
                    for r in sampled_cells],
        "greedy": [f"{r['family']}-{r['size']}-{r['method']}"
                   for r in greedy_cells],
        "unknown_no_generation_config": [
            f"{r['family']}-{r['size']}-{r['method']}" for r in rows
            if r.get("greedy") is None],
    }
    with open(os.path.join(OUT, "decoding_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("=== declared decoding, every cell in the project ===")
    for r in rows:
        print(f"  {r['family']:<10} {r['size']:>5} {r['method']:<5} "
              f"{r['engine'][:26]:<26} "
              f"do_sample={str(r.get('do_sample')):<12} "
              f"temp={str(r.get('temperature')):<6} "
              f"top_p={str(r.get('top_p')):<6} greedy={r.get('greedy')}")
    print(f"\n  greedy cells : {out['summary']['greedy']}")
    print(f"  sampled cells: {out['summary']['sampled']}")
    print(f"\nwrote {OUT}/decoding_audit.json")


if __name__ == "__main__":
    main()
