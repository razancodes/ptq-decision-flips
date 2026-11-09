"""
continuation_token_check.py  --  what m_first is actually a contrast between.

margin_core.py's comment says three refusal continuations are used "so the
margin isn't hostage to one phrasing". That is true of m_seq, which averages
over the whole ~12-token phrase. It is not true of m_first, which is the metric
every reported number uses: the three refusal strings share a first token, so
the refusal side of m_first averages three copies of one logit.

This script asserts that, per model, from the repositories' own tokenizers. It
needs no weights and no GPU -- tokenizer configs only -- so it runs in CI and
fails loudly if a future tokenizer revision changes the split.

It also reports the same thing for ALT_REFUSAL_CONTINUATIONS, the ablation set,
whose whole purpose is to give three DISTINCT first tokens. If that set does not
separate, the ablation is not testing what it claims to and must be rewritten
before any GPU time is spent on it.

Interpretation, fixed before running:

  * default refusal set, distinct first tokens == 1  -> the disclosure in
    margin_core.py and in the paper's measures table is correct as written
  * default refusal set, distinct first tokens > 1   -> the disclosure is wrong
    for that model and must be narrowed to the models where it holds
  * alt refusal set, distinct first tokens == 3      -> the ablation is live
  * alt refusal set, distinct first tokens < 3       -> rewrite the alt set

Inputs
  model_registry.py    fp16 repo per model
  (tokenizer files, from the local HuggingFace cache or the hub)

Output
  results/analysis/continuation_tokens.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

RES = os.path.join(REPO, "results", "analysis")

# Llama-3.2's official repositories are gated. These mirrors carry the identical
# tokenizer and are used ONLY here, never for generation or margins, so that the
# check covers all six models without a token. When a token is available, re-run
# with USE_MIRRORS = False to confirm against the gated repos.
USE_MIRRORS = True
MIRRORS = {
    "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
}

MODEL_KEYS = ["falcon3-1b", "falcon3-3b", "llama3.2-1b", "llama3.2-3b",
              "qwen2.5-1.5b", "qwen2.5-3b"]


def main():
    from transformers import AutoTokenizer
    import model_registry as registry
    from margin_core import (REFUSAL_CONTINUATIONS, COMPLIANCE_CONTINUATIONS,
                             ALT_REFUSAL_CONTINUATIONS,
                             ALT_COMPLIANCE_CONTINUATIONS, first_token_ids)

    sets = {
        "default_refusal": REFUSAL_CONTINUATIONS,
        "default_compliance": COMPLIANCE_CONTINUATIONS,
        "alt_refusal": ALT_REFUSAL_CONTINUATIONS,
        "alt_compliance": ALT_COMPLIANCE_CONTINUATIONS,
    }

    out = {"use_mirrors": USE_MIRRORS, "sets": sets, "models": {}, "failures": []}

    for key in MODEL_KEYS:
        repo = registry.MODELS[key]["fp16"]["repo"]
        used = MIRRORS.get(repo, repo) if USE_MIRRORS else repo
        try:
            tok = AutoTokenizer.from_pretrained(used)
        except Exception as exc:
            out["models"][key] = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
            out["failures"].append(f"{key}: tokenizer unavailable")
            continue

        rec = {"repo": repo, "tokenizer_from": used}
        for name, conts in sets.items():
            ids = first_token_ids(tok, conts)
            rec[name] = {
                "first_token_ids": ids,
                "first_tokens": [tok.convert_ids_to_tokens(i) for i in ids],
                "n_distinct": len(set(ids)),
                "n_strings": len(conts),
            }
        out["models"][key] = rec

        if rec["default_refusal"]["n_distinct"] != 1:
            out["failures"].append(
                f"{key}: default refusal set has "
                f"{rec['default_refusal']['n_distinct']} distinct first tokens, "
                "so the single-token disclosure does not hold here")
        if rec["alt_refusal"]["n_distinct"] != 3:
            out["failures"].append(
                f"{key}: alt refusal set has only "
                f"{rec['alt_refusal']['n_distinct']} distinct first tokens, "
                "so it does not ablate the token choice")

    out["verdict"] = {
        "default_refusal_is_one_token_on_every_model": all(
            m.get("default_refusal", {}).get("n_distinct") == 1
            for m in out["models"].values() if "error" not in m),
        "alt_refusal_separates_on_every_model": all(
            m.get("alt_refusal", {}).get("n_distinct") == 3
            for m in out["models"].values() if "error" not in m),
        "n_models_checked": sum(1 for m in out["models"].values() if "error" not in m),
    }

    os.makedirs(RES, exist_ok=True)
    dest = os.path.join(RES, "continuation_tokens.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== first token of each continuation, per model ===")
    for key, m in out["models"].items():
        if "error" in m:
            print(f"  {key:13} ERROR {m['error']}")
            continue
        d, a = m["default_refusal"], m["alt_refusal"]
        c = m["default_compliance"]
        print(f"  {key:13} ({m['tokenizer_from']})")
        print(f"      default refusal   {d['first_tokens']} -> "
              f"{d['n_distinct']} distinct of {d['n_strings']}")
        print(f"      default compliance{c['first_tokens']} -> "
              f"{c['n_distinct']} distinct of {c['n_strings']}")
        print(f"      alt refusal       {a['first_tokens']} -> "
              f"{a['n_distinct']} distinct of {a['n_strings']}")
    print("\n=== VERDICT ===")
    for k, v in out["verdict"].items():
        print(f"  {k}: {v}")
    if out["failures"]:
        print("\n=== FAILURES ===")
        for f in out["failures"]:
            print(f"  {f}")
    print(f"\nwrote {dest}")
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
