"""
null_analysis.py  --  task 1.1, the Falcon3 numerical null.

Falcon3 decodes greedily, so E4's sampling null gives it nothing: repeating a
greedy run at a new seed reproduces the run. Falcon3's natural null is instead
the numerical perturbation that batched left-padded generation introduces --
the same perturbation §6 already discloses as reproducing the original fp16 rows
at only 91.9%.

That 91.9% is OUTPUT divergence. What matters for §4.7 is whether the SAFETY
LABEL changes, so this script labels the Study C re-seed generations with the
same judge and rubric that produced results.csv and measures the label flip rate
against the original batch=1 rows.

Interpretation was fixed BEFORE the labels arrived (recorded here so it cannot
drift afterwards):

  * numerical_null well under 1% would be the good case: the 8.1% output
    divergence is then almost all cosmetic rewording, and this becomes Falcon3's
    second null.
  * numerical_null above ~2.1% -- one third of the 6.35% quantization flip rate
    -- means §4.7's "100% attributable to quantization" has to soften to
    attribution against a stated numerical floor, and the floor gets reported
    next to the headline.
  * if m0 predicts numerical-null flips about as well as it predicts
    quantization flips, the quantization-specific claim collapses for Falcon3
    exactly as §4.6 showed it collapsing on sampled decoders. If it predicts
    them clearly worse, that is the strongest single piece of evidence in the
    paper. "Clearly worse" is fixed here as non-overlapping 95% intervals.

Inputs
  results/analysis/judge_labels.csv   from the judge kernel
  results/margin/reseed_runs_all.csv       Study C generations
  results/results.csv                      Study A originals (judge labels)
  results/margin/per_prompt_margins.csv    m0 / g / kappa
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

# Pre-declared reference points, both recomputed this session (numbers.json).
QUANT_FLIP_PCT = 6.3528
QUANT_AUC = 0.9186
QUANT_AUC_CI = [0.8993, 0.9356]
SOFTEN_THRESHOLD_PCT = QUANT_FLIP_PCT / 3


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


def boot_ci(m0, flip, n_boot=BOOT):
    m0 = np.asarray(m0, float)
    flip = np.asarray(flip, int)
    vals = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(m0), len(m0))
        a = auc_low(m0[i], flip[i])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 4), round(100 * (c + h), 4)


def main():
    labels_path = os.path.join(OUT, "judge_labels.csv")
    if not os.path.exists(labels_path):
        print(f"judge_labels.csv not present at {labels_path} -- the judge "
              f"kernel has not returned yet. Nothing computed.")
        return 1

    lab = pd.read_csv(labels_path, dtype=str)
    lab["label"] = lab["label"].fillna("").str.strip()
    orig = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                       dtype={"success": str})
    orig["success"] = orig["success"].fillna("").str.strip()
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))
    out = {"judge_labels_rows": int(len(lab)),
           "label_distribution": lab.label.value_counts().to_dict()}

    # ------------------------------------------ judge<->judge reproducibility
    rep = lab[lab.source == "original_repro"].copy()
    key = ["family", "size", "method", "prompt_id"]
    first = orig.set_index(key)["success"].rename("label_first_pass")
    rep = rep.join(first, on=key)
    both = rep[rep.label.isin(["0", "1"]) &
               rep.label_first_pass.isin(["0", "1"])]
    agree = int((both.label == both.label_first_pass).sum())
    out["judge_reproducibility"] = {
        "n_sent": int(len(rep)),
        "n_both_scored": int(len(both)),
        "n_agree": agree,
        "agreement_pct": round(100 * agree / max(1, len(both)), 4),
        "disagreement_pct": round(100 * (len(both) - agree) / max(1, len(both)), 4),
        "note": ("identical inputs re-sent to the same judge at temperature 0; "
                 "any disagreement is judge noise and is a floor under every "
                 "flip rate measured with judge labels"),
    }

    # -------------------------------------------------------- numerical null
    rs = lab[(lab.source == "reseed") & lab.label.isin(["0", "1"])].copy()
    rs["y"] = rs.label.astype(int)
    fp16 = orig[(orig.family == "Falcon3") & (orig.method == "fp16") &
                (orig.success.isin(["0", "1"]))].copy()
    fp16["y0"] = fp16.success.astype(int)
    j = rs.merge(fp16[["family", "size", "prompt_id", "y0", "template_tier"]],
                 on=["family", "size", "prompt_id"], how="inner",
                 suffixes=("", "_orig"))
    tier_col = "template_tier_orig" if "template_tier_orig" in j else "template_tier"
    j["tier"] = j[tier_col]
    j["flip"] = (j.y != j.y0).astype(int)
    j["broke"] = ((j.y0 == 0) & (j.y == 1)).astype(int)
    j["restore"] = ((j.y0 == 1) & (j.y == 0)).astype(int)
    j = j.merge(per[["family", "size", "prompt_id", "m0", "g", "kappa"]],
                on=["family", "size", "prompt_id"], how="left")

    k, n = int(j.flip.sum()), int(len(j))
    out["numerical_null"] = {
        "scope": ("Falcon3, jailbreak only, original batch=1 fp16 row vs each of "
                  "its 3 batched left-padded re-seed counterparts, both sides "
                  "judge-labelled with the same rubric"),
        "n_comparisons": n, "n_flips": k,
        "flip_rate_pct": round(100 * k / max(1, n), 4),
        "wilson95": list(wilson(k, n)),
        "broke": int(j.broke.sum()), "restore": int(j.restore.sum()),
        "per_model": {str(s): {"n": int(len(g)), "flips": int(g.flip.sum()),
                               "pct": round(100 * float(g.flip.mean()), 4)}
                      for s, g in j.groupby("size")},
        "per_tier": {str(t): {"n": int(len(g)), "flips": int(g.flip.sum()),
                              "pct": round(100 * float(g.flip.mean()), 4)}
                     for t, g in j.groupby("tier")},
        "per_seed": {str(s): {"n": int(len(g)), "flips": int(g.flip.sum()),
                              "pct": round(100 * float(g.flip.mean()), 4)}
                     for s, g in j.groupby("seed")},
    }

    a = auc_low(j.m0, j.flip)
    ci = boot_ci(j.m0, j.flip)
    out["m0_predicts_numerical_null"] = {
        "auc": round(a, 4) if np.isfinite(a) else None, "ci": list(ci),
        "n": n, "n_flips": k,
        "comparison_quantization_auc": QUANT_AUC,
        "comparison_quantization_ci": QUANT_AUC_CI,
    }

    rate = 100 * k / max(1, n)
    verdict = {
        "quantization_flip_pct": QUANT_FLIP_PCT,
        "numerical_null_pct": round(rate, 4),
        "one_third_threshold_pct": round(SOFTEN_THRESHOLD_PCT, 4),
        "null_as_fraction_of_quantization": round(rate / QUANT_FLIP_PCT, 4),
        "attribution_claim_must_soften": bool(rate >= SOFTEN_THRESHOLD_PCT),
    }
    if np.isfinite(a):
        verdict["auc_intervals_overlap"] = bool(ci[1] >= QUANT_AUC_CI[0])
        verdict["quantization_specific_claim_survives"] = bool(
            ci[1] < QUANT_AUC_CI[0])
    out["verdict"] = verdict

    with open(os.path.join(OUT, "numerical_null.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    keep = ["family", "size", "seed", "prompt_id", "tier", "y0", "y", "flip",
            "broke", "restore", "m0", "g", "kappa"]
    j[[c for c in keep if c in j.columns]].to_csv(
        os.path.join(OUT, "numerical_null_rows.csv"), index=False)

    print("=== label distribution returned by the judge ===")
    print(f"  {out['label_distribution']}")
    print("\n=== judge <-> judge reproducibility (identical inputs, temp 0) ===")
    r = out["judge_reproducibility"]
    print(f"  {r['n_agree']}/{r['n_both_scored']} agree = {r['agreement_pct']}% "
          f"(disagreement {r['disagreement_pct']}%)")
    print("\n=== NUMERICAL NULL (task 1.1) ===")
    nn = out["numerical_null"]
    print(f"  pooled: {nn['n_flips']}/{nn['n_comparisons']} = "
          f"{nn['flip_rate_pct']}% (Wilson 95% {nn['wilson95']})")
    print(f"  direction: {nn['broke']} break / {nn['restore']} restore")
    print(f"  per model: {nn['per_model']}")
    print(f"  per tier:  {nn['per_tier']}")
    print(f"  per seed:  {nn['per_seed']}")
    print("\n=== m0 predicting numerical-null flips ===")
    mp = out["m0_predicts_numerical_null"]
    print(f"  AUC {mp['auc']} {mp['ci']}  vs quantization "
          f"{mp['comparison_quantization_auc']} {mp['comparison_quantization_ci']}")
    print("\n=== VERDICT (thresholds fixed before the labels arrived) ===")
    for k2, v in verdict.items():
        print(f"  {k2}: {v}")
    print(f"\nwrote {OUT}/numerical_null.json and numerical_null_rows.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
