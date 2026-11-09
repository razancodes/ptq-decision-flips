"""
mechanism_analysis.py  --  tasks 1.4 (the mean shift) and 2.2 (does kappa
transfer across sizes), from the mechanism kernel's output.

1.4 asks four things and they are computed in this order:

  2. delta_m per prompt / format / model, with the MEAN and the SD reported
     separately, overall and per tier.
  3. the sign of E[delta_m], and its size relative to the SD.
  4. the cross-check against Study B, whose perturbation is symmetric by
     construction so its own mean shift is the control (studyb_baseline.json).
  5. whether the first-order expansion still holds at the measured alpha_equiv:
     Study B's model says SD(delta_m) = g * alpha, so comparing the OBSERVED
     SD against g * alpha_equiv tests the variance half of the model directly,
     and comparing the observed MEAN against zero tests the half that the
     isotropic model has no term for.

2.2 asks whether a 1B kappa predicts 3B flips. Cheap with this data, and only
in scope because Phase 2.1 was not run.

Nothing here is tuned. Where the model misses, the miss and its direction are
reported.
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


def auc_low(score, flip):
    score = np.asarray(score, float)
    flip = np.asarray(flip, int)
    ok = np.isfinite(score)
    s, f = -score[ok], flip[ok]
    p, n = int(f.sum()), int((1 - f).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[f == 1].sum() - p * (p + 1) / 2) / (p * n))


def boot_ci(score, flip):
    score = np.asarray(score, float)
    flip = np.asarray(flip, int)
    vals = []
    for _ in range(BOOT):
        i = RNG.integers(0, len(score), len(score))
        a = auc_low(score[i], flip[i])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def describe(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n": int(len(d))}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    t = mean / se if se > 0 else float("nan")
    n_neg = int((d < 0).sum())
    # sign test, exact
    p_sign = float(stats.binomtest(n_neg, len(d), 0.5).pvalue)
    return {"n": int(len(d)), "mean": round(mean, 5), "sd": round(sd, 5),
            "se": round(se, 5), "t": round(t, 3),
            "mean_over_sd": round(mean / sd, 5) if sd > 0 else None,
            "frac_negative": round(n_neg / len(d), 4),
            "sign_test_p": p_sign}


def main():
    dm_path = os.path.join(OUT, "delta_m_prompts.csv")
    ae_path = os.path.join(OUT, "alpha_equiv_tensors.csv")
    if not os.path.exists(dm_path):
        print(f"{dm_path} missing -- the mechanism kernel has not returned yet.")
        return 1

    dm = pd.read_csv(dm_path)
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))
    base = json.load(open(os.path.join(OUT, "studyb_baseline.json"),
                          encoding="utf-8"))
    out = {"available_cells": sorted(
        f"{s}-{m}" for s, m in dm.groupby(["size", "method"]).groups)}

    # ---------------------------------------------------- 1.4 step 2: delta_m
    fp16 = (dm[dm.method == "fp16"]
            .set_index(["family", "size", "prompt_id"])["m_first"]
            .rename("m_fp16"))
    q = dm[dm.method != "fp16"].copy()
    q = q.join(fp16, on=["family", "size", "prompt_id"])
    q["delta_m"] = q.m_first - q.m_fp16
    q["model_key"] = ["falcon3-" + s.lower() for s in q["size"]]
    q = q.merge(per[["model_key", "prompt_id", "m0", "g", "kappa"]],
                on=["model_key", "prompt_id"], how="left")
    q.to_csv(os.path.join(OUT, "delta_m_joined.csv"), index=False)

    out["fp16_reference_check"] = {
        "note": ("m_first measured fresh in this kernel vs Study B's alpha=0 "
                 "value; a large gap would mean the two are not comparable"),
        "cells": {},
    }
    for (fam, size), g in dm[dm.method == "fp16"].groupby(["family", "size"]):
        mk = "falcon3-" + size.lower()
        ref = per[per.model_key == mk].set_index("prompt_id").m0
        j = g.set_index("prompt_id").m_first
        common = ref.index.intersection(j.index)
        d = (j[common] - ref[common]).abs()
        out["fp16_reference_check"]["cells"][f"{fam}-{size}"] = {
            "n": int(len(common)),
            "max_abs_diff": round(float(d.max()), 6),
            "median_abs_diff": round(float(d.median()), 6),
        }

    out["delta_m_overall"] = {}
    for (size, method), g in q.groupby(["size", "method"]):
        key = f"{size}-{method}"
        rec = describe(g.delta_m)
        rec["per_tier"] = {str(t): describe(gg.delta_m)
                           for t, gg in g.groupby("template_tier")}
        rec["per_category"] = {str(c): describe(gg.delta_m)
                               for c, gg in g.groupby("attack_category")}
        out["delta_m_overall"][key] = rec

    jb = q[q.attack_category == "jailbreak"]
    out["delta_m_jailbreak_pooled"] = {
        f"{s}-{m}": describe(g.delta_m)
        for (s, m), g in jb.groupby(["size", "method"])}

    # -------------------------------- 1.4 step 4: the Study B control
    out["studyb_control"] = {
        "overall_mean_delta_m": base["overall_shift"]["mean_delta_m"],
        "overall_sd_delta_m": base["overall_shift"]["sd_delta_m"],
        "mean_over_sd": base["overall_shift"]["mean_over_sd"],
        "note": ("Study B's perturbation is symmetric in W by construction, so "
                 "any mean shift there is second-order curvature, not "
                 "directionality of the perturbation. It is the baseline a real "
                 "format's mean shift must be judged against, not zero."),
        "falcon3_by_alpha": base["shift_falcon3_by_alpha"],
    }

    # -------------------------------- 1.4 step 5: does first order still hold?
    if os.path.exists(ae_path):
        ae = pd.read_csv(ae_path)
        tgt = ae[ae.group == "target"]
        alpha_by_cell = (tgt.groupby(["size", "method"]).alpha_equiv
                         .agg(["median", "mean", "max", "count"]))
        excl = ae[ae.group == "excluded"]
        out["alpha_equiv"] = {
            "targets": {f"{s}-{m}": {k: (round(float(v), 6) if k != "count"
                                         else int(v))
                                     for k, v in r.items()}
                        for (s, m), r in alpha_by_cell.iterrows()},
            "by_kind": {f"{s}-{m}-{k}": round(float(v), 6) for (s, m, k), v in
                        tgt.groupby(["size", "method", "kind"])
                        .alpha_equiv.median().items()},
            "excluded": {f"{s}-{m}-{k}": {"median": round(float(gg.alpha_equiv.median()), 6),
                                          "max": round(float(gg.alpha_equiv.max()), 6),
                                          "n": int(len(gg))}
                         for (s, m, k), gg in excl.groupby(["size", "method", "kind"])},
        }
        first_order = {}
        for (size, method), g in jb.groupby(["size", "method"]):
            if (size, method) not in alpha_by_cell.index:
                continue
            a_med = float(alpha_by_cell.loc[(size, method), "median"])
            gmed = float(g.g.median())
            obs_sd = float(g.delta_m.std(ddof=1))
            obs_mean = float(g.delta_m.mean())
            pred_sd = gmed * a_med
            first_order[f"{size}-{method}"] = {
                "alpha_equiv_median": round(a_med, 6),
                "g_median": round(gmed, 4),
                "predicted_sd_delta_m": round(pred_sd, 5),
                "observed_sd_delta_m": round(obs_sd, 5),
                "sd_ratio_observed_over_predicted": (
                    round(obs_sd / pred_sd, 4) if pred_sd else None),
                "observed_mean_delta_m": round(obs_mean, 5),
                "mean_in_predicted_sd_units": (
                    round(obs_mean / pred_sd, 4) if pred_sd else None),
                "verdict_variance_half": (
                    "holds within 2x" if pred_sd and 0.5 <= obs_sd / pred_sd <= 2
                    else "misses by more than 2x"),
            }
        out["first_order_check"] = first_order
    else:
        out["alpha_equiv"] = None
        out["first_order_check"] = None
        print("(alpha_equiv_tensors.csv absent -- the first-order check and the "
              "alpha_equiv tables are not computed)")

    # ------------------------------------------------ 2.2 does kappa transfer?
    results = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                          dtype={"success": str})
    results["success"] = results["success"].fillna("").str.strip()
    d = results[results.success.isin(["0", "1"])].copy()
    d["y"] = d.success.astype(int)
    idx = ["family", "size", "prompt_id"]
    ref16 = d[d.method == "fp16"].set_index(idx)["y"].rename("s_fp16")
    J = d[d.method != "fp16"].set_index(idx).join(ref16, how="inner").reset_index()
    J["flip"] = (J.y != J.s_fp16).astype(int)
    fal = J[(J.family == "Falcon3") & (J.attack_category == "jailbreak")]
    three = fal[fal["size"] == "3B"]
    one_m = per[per.model_key == "falcon3-1b"].set_index("prompt_id")
    three = three.assign(
        m0_1b=three.prompt_id.map(one_m.m0),
        kappa_1b=three.prompt_id.map(one_m.kappa),
        m0_3b=three.prompt_id.map(
            per[per.model_key == "falcon3-3b"].set_index("prompt_id").m0),
        kappa_3b=three.prompt_id.map(
            per[per.model_key == "falcon3-3b"].set_index("prompt_id").kappa))
    gptq3 = three[three.method == "gptq"]
    out["kappa_transfer"] = {
        "question": "does a Falcon3-1B screen predict Falcon3-3B flips?",
        "n_3b_pairs": int(len(three)), "n_3b_flips": int(three.flip.sum()),
        "all_formats": {
            "m0_1b_predicts_3b": round(auc_low(three.m0_1b, three.flip), 4),
            "m0_1b_ci": boot_ci(three.m0_1b, three.flip),
            "kappa_1b_predicts_3b": round(auc_low(three.kappa_1b, three.flip), 4),
            "kappa_1b_ci": boot_ci(three.kappa_1b, three.flip),
            "m0_3b_own_baseline": round(auc_low(three.m0_3b, three.flip), 4),
            "kappa_3b_own_baseline": round(auc_low(three.kappa_3b, three.flip), 4),
        },
        "gptq_only_confound_free": {
            "n": int(len(gptq3)), "flips": int(gptq3.flip.sum()),
            "m0_1b_predicts_3b": round(auc_low(gptq3.m0_1b, gptq3.flip), 4),
            "m0_3b_own_baseline": round(auc_low(gptq3.m0_3b, gptq3.flip), 4),
            "kappa_1b_predicts_3b": round(auc_low(gptq3.kappa_1b, gptq3.flip), 4),
        },
        "corr_m0_1b_vs_3b": round(float(np.corrcoef(
            three.m0_1b.fillna(0), three.m0_3b.fillna(0))[0, 1]), 4),
    }

    with open(os.path.join(OUT, "mechanism_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    # ------------------------------------------------------------------ report
    print("=== fp16 reference reproducibility (this kernel vs Study B alpha=0) ===")
    for k, v in out["fp16_reference_check"]["cells"].items():
        print(f"  {k}: n={v['n']} median|diff|={v['median_abs_diff']} "
              f"max|diff|={v['max_abs_diff']}")
    print("\n=== delta_m per cell: MEAN and SD reported separately (task 1.4) ===")
    for k, v in out["delta_m_overall"].items():
        print(f"  {k}: n={v['n']} mean={v['mean']:+.4f} sd={v['sd']:.4f} "
              f"mean/sd={v['mean_over_sd']:+.4f} t={v['t']:+.2f} "
              f"frac_negative={v['frac_negative']} sign p={v['sign_test_p']:.2e}")
        for t, tv in v["per_tier"].items():
            if tv.get("n", 0) > 2:
                print(f"      {t:<14} mean={tv['mean']:+.4f} sd={tv['sd']:.4f} "
                      f"frac_neg={tv['frac_negative']}")
    print("\n=== Study B control (symmetric perturbation) ===")
    print(f"  mean={out['studyb_control']['overall_mean_delta_m']:+.5f} "
          f"sd={out['studyb_control']['overall_sd_delta_m']:.5f} "
          f"mean/sd={out['studyb_control']['mean_over_sd']:+.5f}")
    if out["first_order_check"]:
        print("\n=== first-order check: SD(delta_m) vs g * alpha_equiv ===")
        for k, v in out["first_order_check"].items():
            print(f"  {k}: alpha_equiv={v['alpha_equiv_median']:.5f} "
                  f"g={v['g_median']:.2f} predicted_sd={v['predicted_sd_delta_m']:.4f} "
                  f"observed_sd={v['observed_sd_delta_m']:.4f} "
                  f"ratio={v['sd_ratio_observed_over_predicted']} "
                  f"| mean={v['observed_mean_delta_m']:+.4f} "
                  f"({v['mean_in_predicted_sd_units']} predicted SDs)")
    if out["alpha_equiv"]:
        print("\n=== alpha_equiv, target tensors ===")
        for k, v in out["alpha_equiv"]["targets"].items():
            print(f"  {k}: n={v['count']} median={v['median']:.5f} "
                  f"mean={v['mean']:.5f} max={v['max']:.5f}")
        print("=== alpha_equiv, tensors Study B EXCLUDED ===")
        for k, v in out["alpha_equiv"]["excluded"].items():
            print(f"  {k}: n={v['n']} median={v['median']:.5f} max={v['max']:.5f}")
    print("\n=== 2.2 does a 1B screen predict 3B flips? ===")
    kt = out["kappa_transfer"]
    print(f"  3B pairs={kt['n_3b_pairs']} flips={kt['n_3b_flips']} "
          f"corr(m0_1b, m0_3b)={kt['corr_m0_1b_vs_3b']}")
    for k, v in kt["all_formats"].items():
        print(f"    {k}: {v}")
    print(f"  GPTQ only: {kt['gptq_only_confound_free']}")
    print(f"\nwrote {OUT}/mechanism_analysis.json and delta_m_joined.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
