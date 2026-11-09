"""
dose_response_recheck.py  --  is the "3.5x to 454x" failure one mechanism dying
uniformly, or structure I flattened into a range?

Two problems with the number as reported.

FIRST, AND IT IS A BUG. The comparison read observed flip rates out of
decoding_confound.json, which is the STUDY A grid -- the one where AWQ was
sampled at T=1.0 and GGUF at T=0.2. So four of the six "observed" rates carried
the sampling confound that the whole re-run existed to remove, and the
under-prediction factors were computed against inflated targets. The repaired
grid has been available since task 1.2 landed and should have been used.

SECOND, a 113-fold spread across six cells is not one thing failing. If the
worst-predicted cells are the ones whose alpha_equiv is known to EXCLUDE part of
the real perturbation, then the model is not being falsified -- it is being fed a
number that is too small, and the honest claim narrows to "the model holds where
the perturbation can be measured completely, and fails where it cannot".

There is a concrete, pre-stated ranking of how complete each format's
alpha_equiv is, taken from measurements already in hand (§4.11a, §4.11b):

  GPTQ  complete      -- norms, embeddings and lm_head all measured at exactly 0,
                         so the target-tensor alpha is the whole perturbation
  AWQ   incomplete    -- scale migration rewrites the norms; alpha is taken from
                         o_proj alone, which excludes whatever migration does
  GGUF  incomplete    -- quantizes embed_tokens (0.063) and lm_head (0.016),
                         neither of which is in Study B's tensor set at all

If under-prediction tracks that ranking, the structure is the finding.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")

# Declared before looking: how much of the real perturbation each format's
# alpha_equiv actually captures.
COMPLETENESS = {"gptq": "complete", "awq": "excludes scale migration",
                "gguf": "excludes embed_tokens + lm_head"}
RANK = {"gptq": 0, "awq": 1, "gguf": 2}


def analytic_flip_rate(m0, g, alpha):
    m0, g = np.asarray(m0, float), np.asarray(g, float)
    sd = g * alpha
    ok = np.isfinite(m0) & np.isfinite(sd) & (sd > 0)
    z = np.full(m0.shape, np.nan)
    z[ok] = -np.abs(m0[ok]) / sd[ok]
    return float(np.nanmean(stats.norm.cdf(z)))


def main():
    V = json.load(open(os.path.join(OUT, "mechanism_verdict.json"),
                       encoding="utf-8"))
    conf = json.load(open(os.path.join(OUT, "decoding_confound.json"),
                          encoding="utf-8"))
    rg = json.load(open(os.path.join(OUT, "regrid_summary.json"),
                        encoding="utf-8"))
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))
    fal = per[per.model_key.str.startswith("falcon3") &
              (per.attack_category == "jailbreak")]

    A1 = V["Q1_alpha_estimates"]
    old_obs = conf["all_cells"]["by_cell_flip_rate_pct"]          # Study A
    new_obs = rg["repaired_s47"]["all_cells"]["by_cell_flip_rate_pct"]  # repaired

    rows = []
    for cell, a in A1.items():
        size, method = cell.split("-")
        mk = "falcon3-" + size.lower()
        g = fal[fal.model_key == mk]
        pred = 100 * analytic_flip_rate(g.m0, g.g, a["alpha_equiv"])
        rows.append({
            "cell": cell, "method": method,
            "alpha_equiv": a["alpha_equiv"],
            "alpha_completeness": COMPLETENESS[method],
            "predicted_pct": round(pred, 5),
            "observed_STUDY_A_pct": old_obs.get(cell),
            "observed_REPAIRED_pct": new_obs.get(cell),
            "factor_vs_studyA": round(old_obs[cell] / pred, 1) if pred else None,
            "factor_vs_repaired": round(new_obs[cell] / pred, 1) if pred else None,
            "rank": RANK[method],
        })
    df = pd.DataFrame(rows).sort_values("factor_vs_repaired")

    print("=== the bug: which observed rate was the comparison made against? ===")
    print("  Study A rates carry the sampling confound for AWQ (T=1.0) and")
    print("  GGUF (T=0.2). The repaired grid is greedy everywhere.\n")
    print(df[["cell", "alpha_equiv", "predicted_pct", "observed_STUDY_A_pct",
              "observed_REPAIRED_pct", "factor_vs_studyA",
              "factor_vs_repaired"]].to_string(index=False))

    old_range = (df.factor_vs_studyA.min(), df.factor_vs_studyA.max())
    new_range = (df.factor_vs_repaired.min(), df.factor_vs_repaired.max())
    print(f"\n  range as reported (vs Study A) : "
          f"{old_range[0]:.1f}x - {old_range[1]:.1f}x")
    print(f"  range corrected (vs repaired)  : "
          f"{new_range[0]:.1f}x - {new_range[1]:.1f}x")

    print("\n=== the structure: does under-prediction track alpha completeness? ===")
    by_m = (df.groupby("method")
            .agg(alpha=("alpha_equiv", "mean"),
                 factor=("factor_vs_repaired", "mean"),
                 rank=("rank", "first"),
                 completeness=("alpha_completeness", "first"))
            .sort_values("rank"))
    print(by_m.round(3).to_string())

    rho, p = stats.spearmanr(df["rank"], df.factor_vs_repaired)
    rho_a, p_a = stats.spearmanr(df.alpha_equiv, df.factor_vs_repaired)
    print(f"\n  Spearman(completeness rank, under-prediction) = {rho:+.3f} "
          f"(p={p:.4f})")
    print(f"  Spearman(alpha_equiv,        under-prediction) = {rho_a:+.3f} "
          f"(p={p_a:.4f})")
    mono = list(by_m.factor) == sorted(by_m.factor)
    print(f"  monotone in the pre-stated completeness ranking: {mono}")

    print("\n=== what alpha WOULD each cell need, vs what we measured? ===")
    need = []
    for _, r in df.iterrows():
        mk = "falcon3-" + r.cell.split("-")[0].lower()
        g = fal[fal.model_key == mk]
        target = r.observed_REPAIRED_pct / 100.0
        lo, hi = 1e-6, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if analytic_flip_rate(g.m0, g.g, mid) < target:
                lo = mid
            else:
                hi = mid
        req = 0.5 * (lo + hi)
        need.append({"cell": r.cell, "method": r.method,
                     "alpha_measured": r.alpha_equiv,
                     "alpha_required": round(req, 4),
                     "shortfall_x": round(req / r.alpha_equiv, 2)})
    nd = pd.DataFrame(need).sort_values("shortfall_x")
    print(nd.to_string(index=False))
    print(f"\n  alpha shortfall ranges {nd.shortfall_x.min():.2f}x - "
          f"{nd.shortfall_x.max():.2f}x -- far tighter than the flip-rate "
          f"factor,\n  because Phi(-m/(g*alpha)) is violently nonlinear: a 2x "
          f"error in alpha becomes\n  orders of magnitude in predicted rate. "
          f"The flip-rate ratio exaggerates the miss.")

    out = {"cells": df.to_dict("records"),
           "range_vs_studyA": [float(x) for x in old_range],
           "range_vs_repaired": [float(x) for x in new_range],
           "by_method": by_m.reset_index().to_dict("records"),
           "spearman_completeness_vs_factor": [round(float(rho), 4), float(p)],
           "spearman_alpha_vs_factor": [round(float(rho_a), 4), float(p_a)],
           "monotone_in_completeness": bool(mono),
           "alpha_shortfall": nd.to_dict("records")}
    with open(os.path.join(OUT, "dose_response_recheck.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/dose_response_recheck.json")


if __name__ == "__main__":
    main()
