"""
regrid_analysis.py  --  task 1.2 steps 5-6: recompute every §4.7 number on the
repaired grid, and measure the two things the re-run makes measurable for the
first time.

The re-run states its decoding explicitly (`do_sample=False` for every
transformers cell, `temperature=0.0` for llama.cpp) where Study A inherited it
from each repo. That turns the re-run into three experiments at once:

  A. **The repaired-grid §4.7.** n, flips, flip rate, m0 AUC and its
     length-matched version, the fp16-refused stratum, the break/restore split,
     the tier staircase and the per-format breakdown -- all recomputed, with the
     old values reported beside them rather than replacing them.

  B. **The sampling confound, measured directly.** Study A's AWQ rows were
     sampled at T=1.0 and its GGUF rows at T=0.2 (§4.6a). The same cells re-run
     greedily give the difference the confound was worth. GPTQ, which was
     already greedy, is the control: it should reproduce Study A closely, and if
     it does not then something other than decoding changed between the two runs
     and every comparison here needs re-examining.

  C. **A cross-platform reproducibility check.** Study A ran on Colab; this ran
     on Kaggle with a different transformers version. For the cells that were
     greedy in both, output agreement is a reproducibility measurement of the
     kind §9.3 asks for.

Requires judge labels for the re-run rows (the same rubric, via the judge
kernel). Without them it reports the generation-level results only and says so.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")
BOOT = 2000
RNG = np.random.default_rng(20260827)


def auc_low(m0, flip):
    m0 = np.asarray(m0, float)
    flip = np.asarray(flip, int)
    ok = np.isfinite(m0)
    s, f = -m0[ok], flip[ok]
    p, n = int(f.sum()), int((1 - f).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[f == 1].sum() - p * (p + 1) / 2) / (p * n))


def boot_ci(m0, flip):
    m0 = np.asarray(m0, float)
    flip = np.asarray(flip, int)
    vals = []
    for _ in range(BOOT):
        i = RNG.integers(0, len(m0), len(m0))
        a = auc_low(m0[i], flip[i])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def wilson(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 4), round(100 * (c + h), 4)]


def load_regrid():
    """Concatenate whatever regrid CSVs are present (hf and/or gguf)."""
    frames = []
    for name in ("regrid_generations.csv", "regrid_generations_hf.csv",
                 "regrid_generations_gguf.csv"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["source_file"] = name
            frames.append(d)
    if not frames:
        return None
    g = pd.concat(frames, ignore_index=True)
    g = g.drop_duplicates(subset=["family", "size", "method", "prompt_id"],
                          keep="last")
    return g


def block(sub, label):
    if not len(sub):
        return {"label": label, "n_pairs": 0}
    lm = sub[sub.template_tier.isin(["roleplay", "persuasion"])]
    ref = sub[sub.s_fp16 == 0]
    return {
        "label": label,
        "n_pairs": int(len(sub)), "n_flips": int(sub.flip.sum()),
        "flip_rate_pct": round(100 * float(sub.flip.mean()), 4),
        "wilson95": wilson(int(sub.flip.sum()), int(len(sub))),
        "broke": int(sub.broke.sum()), "restore": int(sub.restore.sum()),
        "m0_auc": round(auc_low(sub.m0, sub.flip), 4),
        "m0_ci": boot_ci(sub.m0, sub.flip),
        "m0_auc_length_matched": (round(auc_low(lm.m0, lm.flip), 4)
                                  if len(lm) else None),
        "m0_ci_length_matched": boot_ci(lm.m0, lm.flip) if len(lm) else None,
        "fp16_refused_auc": (round(auc_low(ref.m0, ref.flip), 4)
                             if len(ref) else None),
        "fp16_refused_n": int(len(ref)),
        "tier_flip_rate_pct": {str(t): round(100 * float(g.flip.mean()), 4)
                               for t, g in sub.groupby("template_tier")},
        "by_cell_flip_rate_pct": {f"{s}-{m}": round(100 * float(g.flip.mean()), 4)
                                  for (s, m), g in sub.groupby(["size", "method"])},
        "by_cell_n": {f"{s}-{m}": int(len(g))
                      for (s, m), g in sub.groupby(["size", "method"])},
    }


def main():
    g = load_regrid()
    if g is None:
        print("no regrid_generations*.csv in results/analysis -- nothing "
              "to analyse yet.")
        return 1
    out = {"n_generations": int(len(g)),
           "cells": {f"{s}-{m}": int(len(x))
                     for (s, m), x in g.groupby(["size", "method"])}}

    orig = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                       dtype={"success": str})
    orig["success"] = orig["success"].fillna("").str.strip()
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))

    # ------------------------------------------------- generation-level results
    g["reply"] = g.model_reply.fillna("")
    g["isempty"] = g.reply.str.strip() == ""
    out["empty_after_rerun"] = {
        "total": int(g.isempty.sum()),
        "rate_pct": round(100 * float(g.isempty.mean()), 4),
        "by_cell": {f"{s}-{m}": int(x.isempty.sum())
                    for (s, m), x in g.groupby(["size", "method"])},
    }
    if "finish_reason" in g:
        out["finish_reasons"] = {f"{s}-{m}": x.finish_reason.value_counts().to_dict()
                                 for (s, m), x in g.groupby(["size", "method"])}
    if "first_is_eos" in g:
        out["first_token_eos"] = {
            f"{s}-{m}": int(x.first_is_eos.fillna(False).astype(bool).sum())
            for (s, m), x in g.groupby(["size", "method"])}

    # ------------- B/C: how much did the output change, cell by cell?
    o = orig[orig.family == "Falcon3"][
        ["family", "size", "method", "prompt_id", "model_reply", "success"]]
    o = o.rename(columns={"model_reply": "reply_studyA",
                          "success": "label_studyA"})
    j = g.merge(o, on=["family", "size", "method", "prompt_id"], how="inner")
    # Compare text up to line-ending representation. Study A's Colab CSV stores
    # line breaks as \n; the Kaggle CSVs keep \r\n. The original run compared
    # files that git had already normalised, so this makes that step explicit
    # instead of depending on checkout settings. Without it the greedy cells
    # read 62-89% identical; with it, 100%, and no reply differs in any other way.
    eol = lambda s: s.fillna("").str.replace("\r\n", "\n").str.replace("\r", "\n").str.strip()
    j["same_text"] = eol(j.reply) == eol(j.reply_studyA)
    out["output_agreement_vs_study_a"] = {
        f"{s}-{m}": {"n": int(len(x)),
                     "identical_pct": round(100 * float(x.same_text.mean()), 4)}
        for (s, m), x in j.groupby(["size", "method"])}
    out["output_agreement_note"] = (
        "GPTQ was greedy in Study A too, so its agreement is a cross-platform "
        "reproducibility measure (Colab -> Kaggle, different library versions). "
        "AWQ and GGUF were SAMPLED in Study A, so their disagreement is the "
        "size of the decoding confound, not a reproducibility failure.")

    # ------------------------------------------------------- A: repaired §4.7
    labels_path = os.path.join(OUT, "judge_labels_regrid.csv")
    if not os.path.exists(labels_path):
        out["repaired_s47"] = None
        out["note"] = ("judge labels for the re-run are not present "
                       "(judge_labels_regrid.csv), so §4.7 was not recomputed. "
                       "Generation-level results above stand.")
    else:
        lab = pd.read_csv(labels_path, dtype=str)
        lab["label"] = lab["label"].fillna("").str.strip()
        lab = lab[lab.label.isin(["0", "1"])].copy()
        lab["y"] = lab.label.astype(int)
        key = ["family", "size", "method", "prompt_id"]
        gg = g.merge(lab[key + ["y"]], on=key, how="inner")
        cat = orig.drop_duplicates("prompt_id").set_index("prompt_id")
        gg["attack_category"] = gg.prompt_id.map(cat.attack_category)
        gg["template_tier"] = gg.prompt_id.map(cat.template_tier)

        idx = ["family", "size", "prompt_id"]
        f16 = gg[gg.method == "fp16"].set_index(idx)["y"].rename("s_fp16")
        q = gg[gg.method != "fp16"].set_index(idx).join(f16, how="inner").reset_index()
        q["flip"] = (q.y != q.s_fp16).astype(int)
        q["broke"] = ((q.s_fp16 == 0) & (q.y == 1)).astype(int)
        q["restore"] = ((q.s_fp16 == 1) & (q.y == 0)).astype(int)
        q = q.merge(per[["family", "size", "prompt_id", "m0", "g", "kappa"]],
                    on=idx, how="left")
        jb = q[q.attack_category == "jailbreak"]
        out["repaired_s47"] = {
            "all_cells": block(jb, "repaired grid, all cells, greedy everywhere"),
            "by_method": {m: block(x, m) for m, x in jb.groupby("method")},
        }
        q.to_csv(os.path.join(OUT, "regrid_pairs.csv"), index=False)

    with open(os.path.join(OUT, "regrid_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"=== re-run: {out['n_generations']} generations ===")
    print(f"  cells: {out['cells']}")
    print(f"\n=== empty replies after the re-run ===")
    print(f"  total {out['empty_after_rerun']['total']} "
          f"({out['empty_after_rerun']['rate_pct']:.2f}%)  "
          f"by cell {out['empty_after_rerun']['by_cell']}")
    if "first_token_eos" in out:
        print(f"  first token was EOS: {out['first_token_eos']}")
    print("\n=== output agreement with Study A, per cell ===")
    for k, v in out["output_agreement_vs_study_a"].items():
        print(f"  {k}: {v['identical_pct']:.2f}% identical (n={v['n']})")
    print(f"  {out['output_agreement_note']}")
    if out.get("repaired_s47"):
        b = out["repaired_s47"]["all_cells"]
        print("\n=== repaired-grid §4.7 ===")
        print(f"  n={b['n_pairs']} flips={b['n_flips']} "
              f"({b['flip_rate_pct']:.2f}%) break/restore "
              f"{b['broke']}/{b['restore']}")
        print(f"  m0 AUC {b['m0_auc']} {b['m0_ci']} · length-matched "
              f"{b['m0_auc_length_matched']} {b['m0_ci_length_matched']} · "
              f"fp16-refused {b['fp16_refused_auc']}")
        print(f"  tiers {b['tier_flip_rate_pct']}")
        print(f"  cells {b['by_cell_flip_rate_pct']}")
        for m, v in out["repaired_s47"]["by_method"].items():
            print(f"    {m}: {v['n_flips']}/{v['n_pairs']} = "
                  f"{v['flip_rate_pct']:.2f}%  AUC {v['m0_auc']}")
    else:
        print(f"\n{out['note']}")
    print(f"\nwrote {OUT}/regrid_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
