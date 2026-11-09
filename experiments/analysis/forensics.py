"""
forensics.py  --  Task 1.2 pre-diagnosis, and the correction to §4.9.

Everything here comes from data already on disk. It costs no GPU and it narrows
what the re-run kernel has to establish, which matters because Falcon3 decodes
greedily: any hypothesis that can be settled from the existing outputs should be
settled before spending a GPU session on it.

Three questions:

  1. What is the true empty-output census? §4.9 attributes it to Falcon3-1B-AWQ.
  2. Is the "Falcon3-3B-GPTQ produced repetition loops" claim supported?
  3. Which of the five candidate root causes can already be ruled in or out?

Writes results/analysis/forensics.json + a human-readable summary.
"""

import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")


def ttr_degeneracy(text):
    """1 - type/token ratio over whitespace tokens. High = repetitive."""
    toks = str(text).split()
    if len(toks) < 10:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)


def max_ngram_loop(text, n=4):
    """Largest share of any single n-gram. Catches 'A B A B A B' loops that a
    type/token ratio can miss when the vocabulary of the loop is small."""
    toks = str(text).split()
    if len(toks) < n + 4:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) / len(grams)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                     dtype={"success": str})
    df["success"] = df["success"].fillna("").str.strip()
    df["rep"] = df["model_reply"].fillna("")
    df["isempty"] = df["rep"].str.strip() == ""
    df["plen"] = df["prompt_text"].str.len()
    out = {}

    # ---------------------------------------------------------------- 1. census
    emp = df[df.isempty]
    census = (emp.groupby(["family", "size", "method", "template_tier"])
                 .size().rename("n_empty").reset_index())
    out["empty_total"] = int(len(emp))
    out["empty_by_cell"] = census.to_dict("records")
    out["empty_families"] = emp["family"].value_counts().to_dict()
    out["empty_labels"] = emp["success"].value_counts().to_dict()

    fal = df[df.family == "Falcon3"]
    out["falcon3_empty_total"] = int(fal.isempty.sum())
    out["falcon3_empty_by_method"] = fal[fal.isempty]["method"].value_counts().to_dict()
    out["falcon3_1b_awq_jb"] = {
        "n": int(len(fal[(fal["size"] == "1B") & (fal.method == "awq") &
                         (fal.attack_category == "jailbreak")])),
        "n_empty": int(fal[(fal["size"] == "1B") & (fal.method == "awq") &
                           (fal.attack_category == "jailbreak")].isempty.sum()),
    }
    # fp16 empties are the decisive fact: they exist under no quantization at all
    out["falcon3_fp16_empty"] = int(fal[(fal.method == "fp16")].isempty.sum())

    # do the same prompts fail in more than one cell?
    byp = (fal[fal.isempty].groupby("prompt_id")
              .agg(n_cells=("method", "size"),
                   cells=("method", lambda s: ",".join(sorted(s)))))
    out["distinct_failing_prompts"] = int(len(byp))
    out["prompts_failing_multiple_cells"] = int((byp.n_cells > 1).sum())
    out["top_failing_prompts"] = (byp.sort_values("n_cells", ascending=False)
                                     .head(10).reset_index().to_dict("records"))

    # --------------------------------------------------- 2. repetition claim
    df["ttr"] = df["rep"].map(ttr_degeneracy)
    df["loop4"] = df["rep"].map(max_ngram_loop)
    deg = df[(df.ttr > 0.8) | (df.loop4 > 0.5)]
    out["degenerate_total"] = int(len(deg))
    out["degenerate_by_cell"] = (deg.groupby(["family", "size", "method"]).size()
                                 .sort_values(ascending=False)
                                 .rename("n").reset_index().to_dict("records"))
    g = df[(df.family == "Falcon3") & (df["size"] == "3B") & (df.method == "gptq")]
    out["falcon3_3b_gptq"] = {
        "n": int(len(g)),
        "ttr_gt_0.8": int((g.ttr > 0.8).sum()),
        "ttr_gt_0.6": int((g.ttr > 0.6).sum()),
        "loop4_gt_0.5": int((g.loop4 > 0.5).sum()),
        "ttr_max": round(float(g.ttr.max()), 4),
        "loop4_max": round(float(g.loop4.max()), 4),
        "ttr_mean": round(float(g.ttr.mean()), 4),
    }
    # a fair comparison: is 3B-gptq worse than its own family's other cells?
    fal_deg = (df[df.family == "Falcon3"].groupby(["size", "method"])
                 .agg(ttr_mean=("ttr", "mean"), ttr_max=("ttr", "max"),
                      n_ttr_gt_06=("ttr", lambda s: int((s > 0.6).sum())),
                      loop4_mean=("loop4", "mean")).round(4).reset_index())
    out["falcon3_degeneracy_by_cell"] = fal_deg.to_dict("records")

    # ------------------------------------------- 3. candidate-cause screening
    prompts = df.drop_duplicates("prompt_id")[
        ["prompt_id", "template_tier", "attack_category", "prompt_text", "plen"]]
    marker = re.compile(r"<\|[^>]*\|>|\[INST\]|###\s*(?:Assistant|Response)|"
                        r"<\s*/?\s*(?:s|system|assistant)\s*>", re.I)
    with_marker = prompts[prompts.prompt_text.str.contains(marker, regex=True)]
    out["prompts_with_chat_markers"] = {
        "n": int(len(with_marker)),
        "examples": with_marker["prompt_id"].head(5).tolist(),
    }

    failing = prompts[prompts.prompt_id.isin(set(fal[fal.isempty].prompt_id))]
    rp = prompts[prompts.template_tier.isin(["roleplay", "persuasion"])]
    out["prompt_length"] = {
        "failing_mean": round(float(failing.plen.mean()), 1),
        "failing_min": int(failing.plen.min()),
        "failing_max": int(failing.plen.max()),
        "roleplay_persuasion_mean": round(float(rp.plen.mean()), 1),
        "roleplay_persuasion_max": int(rp.plen.max()),
        "all_mean": round(float(prompts.plen.mean()), 1),
    }
    out["failing_prompt_tiers"] = failing.template_tier.value_counts().to_dict()

    # Is length alone sufficient? Compare failing vs non-failing WITHIN the
    # worst cell, where every row saw the same model and the same config.
    cell = fal[(fal["size"] == "1B") & (fal.method == "awq") &
               (fal.attack_category == "jailbreak")]
    out["within_worst_cell_length"] = {
        "empty_mean_chars": round(float(cell[cell.isempty].plen.mean()), 1),
        "nonempty_mean_chars": round(float(cell[~cell.isempty].plen.mean()), 1),
        "empty_n": int(cell.isempty.sum()),
        "nonempty_n": int((~cell.isempty).sum()),
    }
    # ...and whether the longest prompts in the cell are the ones that failed
    ranked = cell.sort_values("plen", ascending=False)
    top50 = ranked.head(50)
    out["within_worst_cell_length"]["empties_among_50_longest"] = int(top50.isempty.sum())

    # the exact reproduce list for the kernel
    repro = fal[fal.isempty][["family", "size", "method", "attack_category",
                              "template_tier", "prompt_id"]].copy()
    repro.to_csv(os.path.join(OUT, "repro_empty_rows.csv"), index=False)
    out["repro_rows_written"] = int(len(repro))

    with open(os.path.join(OUT, "forensics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ------------------------------------------------------------------ report
    print("=== 1. EMPTY-OUTPUT CENSUS ===")
    print(f"empty replies in the whole dataset: {out['empty_total']}")
    print(f"  by family: {out['empty_families']}")
    print(f"  all carry judge label: {out['empty_labels']}")
    print(f"Falcon3 empties by method: {out['falcon3_empty_by_method']}")
    print(f"Falcon3 fp16 empties: {out['falcon3_fp16_empty']}  "
          f"<-- unquantized rows also return empty")
    print(f"Falcon3-1B-AWQ jailbreak: {out['falcon3_1b_awq_jb']['n_empty']}"
          f"/{out['falcon3_1b_awq_jb']['n']}")
    print(f"distinct failing prompts: {out['distinct_failing_prompts']}, "
          f"failing in >1 cell: {out['prompts_failing_multiple_cells']}")

    print("\n=== 2. REPETITION-LOOP CLAIM (§4.9) ===")
    print(f"Falcon3-3B-GPTQ: {out['falcon3_3b_gptq']}")
    print("Falcon3 degeneracy by cell:")
    print(fal_deg.to_string(index=False))
    print(f"whole-dataset degenerate rows ({out['degenerate_total']}) by cell:")
    for r in out["degenerate_by_cell"][:6]:
        print(f"  {r}")

    print("\n=== 3. CANDIDATE-CAUSE SCREENING ===")
    print(f"prompts containing chat-template markers: "
          f"{out['prompts_with_chat_markers']['n']}")
    print(f"failing-prompt tiers: {out['failing_prompt_tiers']}")
    print(f"prompt length: {out['prompt_length']}")
    print(f"within worst cell: {out['within_worst_cell_length']}")
    print(f"\nwrote {OUT}/forensics.json and repro_empty_rows.csv "
          f"({out['repro_rows_written']} rows)")


if __name__ == "__main__":
    main()
