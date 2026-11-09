"""
phase21_analysis.py  --  task 2.1: does the headline hold in three families?

Phase 2.1 re-collected Qwen2.5 and Llama-3.2 greedily, with decoding stated
rather than inherited, so that both can be analysed on the same footing as the
repaired Falcon3 grid. Two rules from the task govern how it may be used, and
both are enforced here rather than left to discipline:

  * these rows are NEW DATA, not a control. Greedy Qwen and Llama output is not
    comparable to the sampled rows in results.csv, so nothing here joins to
    them. The Study A grid is loaded only to report, separately, how far the
    greedy re-collection moved each cell -- which is the size of the decoding
    confound in those families, exactly as it was measured for Falcon3.
  * a flip needs its fp16 partner. If a family's fp16 cells are missing, that
    family yields no pairs at all and is reported as not computed rather than
    silently analysed on whatever cells did arrive.

Outputs results/analysis/phase21_analysis.json and a per-family table
matching §4.7's shape, so the document can report three families side by side.
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

P21_FILES = ("p21_generations_q-a.csv", "p21_generations_q-b.csv",
             "p21_generations_gated.csv", "p21_generations_gguf-a.csv",
             "p21_generations_gguf-b.csv")


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


def load_generations():
    frames = []
    for name in P21_FILES:
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["p21_source_file"] = name
            frames.append(d)
    if not frames:
        return None
    g = pd.concat(frames, ignore_index=True)
    return g.drop_duplicates(subset=["family", "size", "method", "prompt_id"],
                             keep="last")


def block(sub, label):
    lm = sub[sub.template_tier.isin(["roleplay", "persuasion"])]
    ref = sub[sub.s_fp16 == 0]
    k, n = int(sub.flip.sum()), int(len(sub))
    return {
        "label": label, "n_pairs": n, "n_flips": k,
        "flip_rate_pct": round(100 * k / n, 4) if n else None,
        "wilson95": wilson(k, n),
        "broke": int(sub.broke.sum()), "restore": int(sub.restore.sum()),
        "pct_degrading": (round(100 * sub.broke.sum() / k, 2) if k else None),
        "m0_auc": round(auc_low(sub.m0, sub.flip), 4) if k else None,
        "m0_ci": boot_ci(sub.m0, sub.flip) if k else None,
        "m0_auc_length_matched": (round(auc_low(lm.m0, lm.flip), 4)
                                  if len(lm) and lm.flip.sum() else None),
        "fp16_refused_auc": (round(auc_low(ref.m0, ref.flip), 4)
                             if len(ref) and ref.flip.sum() else None),
        "fp16_refused_n": int(len(ref)),
        "tier_flip_rate_pct": {str(t): round(100 * float(x.flip.mean()), 4)
                               for t, x in sub.groupby("template_tier")},
        "by_cell_flip_rate_pct": {f"{s}-{m}": round(100 * float(x.flip.mean()), 4)
                                  for (s, m), x in sub.groupby(["size", "method"])},
    }


def main():
    gen = load_generations()
    if gen is None:
        print("no p21_generations_*.csv present yet.")
        return 1
    cells = gen.groupby(["family", "size", "method"]).size()
    print(f"=== Phase 2.1 generations: {len(gen):,} rows, {len(cells)} cells ===")
    print(cells.to_string())

    out = {"n_generations": int(len(gen)),
           "cells": {f"{f}-{s}-{m}": int(v) for (f, s, m), v in cells.items()},
           "empty_replies": int((gen.model_reply.fillna("").str.strip() == "").sum())}

    # a family without fp16 cells yields no pairs at all
    have_fp16 = {f for (f, _s, m) in cells.index if m == "fp16"}
    families = sorted({f for (f, _s, _m) in cells.index})
    out["families_present"] = families
    out["families_with_fp16"] = sorted(have_fp16)
    out["families_not_computable"] = sorted(set(families) - have_fp16)
    for fam in out["families_not_computable"]:
        print(f"\n  !! {fam}: no fp16 cells present, so no fp16<->quantized "
              f"pairs exist. Reported as NOT COMPUTED, not analysed on the "
              f"quantized cells alone.")

    labels_path = os.path.join(OUT, "judge_labels_phase21.csv")
    if not os.path.exists(labels_path):
        out["note"] = ("judge labels for Phase 2.1 are not present; "
                       "generation-level results only")
        with open(os.path.join(OUT, "phase21_analysis.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\n{out['note']}")
        return 0

    lab = pd.read_csv(labels_path, dtype=str)
    lab = lab[lab.label.isin(["0", "1"])].copy()
    lab["y"] = lab.label.astype(int)
    key = ["family", "size", "method", "prompt_id"]
    g = gen.merge(lab[key + ["y"]], on=key, how="inner")

    orig = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                       dtype={"success": str})
    cat = orig.drop_duplicates("prompt_id").set_index("prompt_id")
    g["attack_category"] = g.prompt_id.map(cat.attack_category)
    g["template_tier"] = g.prompt_id.map(cat.template_tier)
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))

    idx = ["family", "size", "prompt_id"]
    f16 = g[g.method == "fp16"].set_index(idx)["y"].rename("s_fp16")
    q = g[g.method != "fp16"].set_index(idx).join(f16, how="inner").reset_index()
    q["flip"] = (q.y != q.s_fp16).astype(int)
    q["broke"] = ((q.s_fp16 == 0) & (q.y == 1)).astype(int)
    q["restore"] = ((q.s_fp16 == 1) & (q.y == 0)).astype(int)
    q = q.merge(per[["family", "size", "prompt_id", "m0", "g", "kappa"]],
                on=idx, how="left")
    jb = q[q.attack_category == "jailbreak"]
    q.to_csv(os.path.join(OUT, "phase21_pairs.csv"), index=False)

    out["by_family"] = {f: block(x, f) for f, x in jb.groupby("family")}
    out["pooled"] = block(jb, "Qwen2.5 + Llama-3.2 pooled, jailbreak, greedy")

    # How far did greedy re-collection move each cell? Same measurement made for
    # Falcon3 -- this is the size of the decoding confound in these families.
    o = orig[["family", "size", "method", "prompt_id", "model_reply"]].rename(
        columns={"model_reply": "reply_studyA"})
    j2 = gen.merge(o, on=key, how="inner")
    j2["same"] = (j2.model_reply.fillna("").str.strip() ==
                  j2.reply_studyA.fillna("").str.strip())
    out["output_agreement_vs_study_a"] = {
        f"{f}-{s}-{m}": {"n": int(len(x)),
                         "identical_pct": round(100 * float(x.same.mean()), 4)}
        for (f, s, m), x in j2.groupby(["family", "size", "method"])}

    with open(os.path.join(OUT, "phase21_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("\n=== per family, jailbreak, greedy on both sides ===")
    for fam, b in out["by_family"].items():
        print(f"  {fam}: n={b['n_pairs']:,} flips={b['n_flips']} "
              f"({b['flip_rate_pct']}%) break/restore {b['broke']}/{b['restore']}"
              f"  m0 AUC {b['m0_auc']} {b['m0_ci']}  "
              f"length-matched {b['m0_auc_length_matched']}")
        print(f"      tiers {b['tier_flip_rate_pct']}")
    p = out["pooled"]
    print(f"  POOLED: n={p['n_pairs']:,} flips={p['n_flips']} "
          f"({p['flip_rate_pct']}%) m0 AUC {p['m0_auc']} {p['m0_ci']}")
    print("\n=== how far greedy moved each cell from Study A ===")
    for k2, v in sorted(out["output_agreement_vs_study_a"].items()):
        print(f"  {k2}: {v['identical_pct']:.2f}% identical")
    print(f"\nwrote {OUT}/phase21_analysis.json and phase21_pairs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
