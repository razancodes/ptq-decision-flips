"""
empty_reproducibility.py  --  do the empty outputs survive a change of batching?

Task 1.2 step 1 asks whether the 23 empty Falcon3-1B-AWQ replies reproduce at the
original config, and warns that if a re-run produced different output the
generation conditions must have changed. There is already an answer available for
the fp16 cell at zero GPU cost: Study C re-generated every fp16 row under a
DIFFERENT configuration (batch 16, left-padded, three seeds) and those rows have
now been judge-labelled.

So: are the prompts that returned empty at batch=1 the same prompts that return
empty under batched left-padded generation? If yes, the empty output is stable
under a numerical perturbation large enough to change 8.1% of all outputs --
strong evidence that it is deterministic model behaviour rather than a fragile
padding or batching artefact, established before the GPU re-run reports.
"""

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")


def main():
    lab = pd.read_csv(os.path.join(OUT, "judge_labels.csv"), dtype=str)
    orig = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                       dtype={"success": str})
    orig["success"] = orig["success"].fillna("").str.strip()
    reseed = pd.read_csv(os.path.join(REPO, "results", "margin",
                                      "reseed_runs_all.csv"), dtype=str)

    rs = lab[lab.source == "reseed"].copy()
    err = rs[rs.label == "error"]

    rsr = reseed[(reseed.family == "Falcon3") &
                 (reseed.attack_category == "jailbreak")].copy()
    rsr["isempty"] = rsr.model_reply.fillna("").str.strip() == ""

    f = orig[(orig.family == "Falcon3") & (orig.method == "fp16") &
             (orig.attack_category == "jailbreak")].copy()
    f["isempty"] = f.model_reply.fillna("").str.strip() == ""

    empty_reseed = set(zip(rsr[rsr.isempty]["size"], rsr[rsr.isempty].prompt_id))
    empty_orig = set(zip(f[f.isempty]["size"], f[f.isempty].prompt_id))

    # how many of the 3 seeds did each re-seed prompt return empty on?
    per_prompt = (rsr[rsr.isempty].groupby(["size", "prompt_id"]).size()
                  .rename("n_seeds_empty").reset_index())

    out = {
        "reseed_rows_labelled_error": int(len(err)),
        "reseed_empty_rows": int(rsr.isempty.sum()),
        "reseed_distinct_prompts_empty": len(empty_reseed),
        "original_fp16_empty_rows": int(f.isempty.sum()),
        "original_fp16_distinct_prompts_empty": len(empty_orig),
        "overlap": sorted(f"{s}/{p}" for s, p in (empty_reseed & empty_orig)),
        "n_overlap": len(empty_reseed & empty_orig),
        "reseed_only": sorted(f"{s}/{p}" for s, p in (empty_reseed - empty_orig)),
        "original_only": sorted(f"{s}/{p}" for s, p in (empty_orig - empty_reseed)),
        "seeds_empty_per_prompt": per_prompt.to_dict("records"),
    }
    with open(os.path.join(OUT, "empty_reproducibility.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=== empty fp16 outputs: batch=1 (Study A) vs batch=16 left-padded (Study C) ===")
    print(f"  Study A fp16 jailbreak empties: {out['original_fp16_empty_rows']} rows, "
          f"{out['original_fp16_distinct_prompts_empty']} distinct prompts")
    print(f"  Study C re-seed empties: {out['reseed_empty_rows']} rows over 3 seeds, "
          f"{out['reseed_distinct_prompts_empty']} distinct prompts")
    print(f"  prompts empty in BOTH configurations: {out['n_overlap']}")
    for x in out["overlap"]:
        print(f"    {x}")
    if out["original_only"]:
        print(f"  empty only at batch=1: {out['original_only']}")
    if out["reseed_only"]:
        print(f"  empty only when batched: {out['reseed_only']}")
    print("\n  seeds (of 3) on which each prompt returned empty:")
    print(per_prompt.to_string(index=False))
    print(f"\nwrote {OUT}/empty_reproducibility.json")


if __name__ == "__main__":
    main()
