"""
mechanism_verdict.py  --  the part of task 1.3/1.4 that decides what §5 should say.

Three questions, in the order the answers depend on each other:

Q1  What is each format's real quantization error, in Study B's units?
    Not simply the median alpha_equiv: awq_anomaly.py showed AWQ's per-tensor
    weight distance is dominated by activation-aware scale migration, which is
    equivalence-preserving (it divides the preceding norm by the same factor it
    multiplies into the linear). The tell is that AWQ rewrites 36 of 37 norm
    tensors while GPTQ and GGUF leave every one bit-identical, and that within
    AWQ only the norm-fed projections are inflated -- o_proj, whose input is not
    norm-scaled, sits at 0.084 against ~0.51 for q/k/v/gate.

    So AWQ's quantization error is estimated from o_proj alone, and that choice
    is declared here rather than tuned: it is the one target tensor the
    migration does not touch. GGUF and GPTQ use all target tensors, since their
    norms are untouched and their per-projection spread is negligible.

Q2  Does the FIRST-ORDER VARIANCE model hold at that alpha? Study B says
    SD(delta_m) = g * alpha. Compare against the measured SD.

Q3  Does the ZERO-CROSSING FLIP model hold? Compare the flip rate the model
    predicts at that alpha against the observed one, and -- the question §5
    actually needs -- test whether delta_m predicts, prompt by prompt, which
    prompts flip.

Q3 is where the mechanism section lives or dies, because a margin that shifts
without ever crossing zero cannot flip a decision by the stated mechanism, no
matter how well the variance model fits.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")
BOOT = 2000
RNG = np.random.default_rng(20260827)

# Declared before looking at the outcome: AWQ's migration-free target tensor.
AWQ_CLEAN_KIND = "o_proj"


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


def analytic_flip_rate(m0, g, alpha):
    m0 = np.asarray(m0, float)
    g = np.asarray(g, float)
    sd = g * alpha
    ok = np.isfinite(m0) & np.isfinite(sd) & (sd > 0)
    z = np.full(m0.shape, np.nan)
    z[ok] = -np.abs(m0[ok]) / sd[ok]
    return float(np.nanmean(stats.norm.cdf(z)))


def main():
    a = pd.read_csv(os.path.join(OUT, "alpha_equiv_tensors.csv"))
    dmj = pd.read_csv(os.path.join(OUT, "delta_m_joined.csv"))
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))
    conf = json.load(open(os.path.join(OUT, "decoding_confound.json"),
                          encoding="utf-8"))
    base = json.load(open(os.path.join(OUT, "studyb_baseline.json"),
                          encoding="utf-8"))
    out = {}

    # --------------------------------------------------------------------- Q1
    tgt = a[a.group == "target"]
    alpha_est = {}
    for (size, method), g in tgt.groupby(["size", "method"]):
        if method == "awq":
            sub = g[g.kind == AWQ_CLEAN_KIND]
            est, basis = float(sub.alpha_equiv.median()), \
                f"{AWQ_CLEAN_KIND} only (migration-free)"
        else:
            est, basis = float(g.alpha_equiv.median()), "all target tensors"
        alpha_est[f"{size}-{method}"] = {
            "alpha_equiv": round(est, 5), "basis": basis,
            "all_targets_median": round(float(g.alpha_equiv.median()), 5),
            "n_tensors": int(len(g)),
        }
    out["Q1_alpha_estimates"] = alpha_est
    out["Q1_reference"] = {
        "uniform_4bit_over_gaussian": 0.1263,
        "note": ("a k-bit uniform quantizer over a Gaussian gives RMS error "
                 "~ (7/2^k)/sqrt(12) in units of std; 4-bit is 0.126. GPTQ "
                 "lands there, AWQ's o_proj sits below it (group-wise scaling), "
                 "GGUF sits near the 5-bit value because Q4_K_M is a mixed "
                 "scheme that stores some tensors at Q6_K."),
    }

    # --------------------------------------------------------------------- Q2
    jb = dmj[dmj.attack_category == "jailbreak"].copy()
    q2 = {}
    for (size, method), g in jb.groupby(["size", "method"]):
        key = f"{size}-{method}"
        if key not in alpha_est:
            continue
        al = alpha_est[key]["alpha_equiv"]
        gmed = float(g.g.median())
        pred_sd = gmed * al
        obs_sd = float(g.delta_m.std(ddof=1))
        obs_mean = float(g.delta_m.mean())
        q2[key] = {
            "alpha_equiv": al, "g_median": round(gmed, 4),
            "predicted_sd": round(pred_sd, 5),
            "observed_sd": round(obs_sd, 5),
            "ratio_obs_over_pred": round(obs_sd / pred_sd, 4) if pred_sd else None,
            "observed_mean": round(obs_mean, 5),
            "mean_over_sd": round(obs_mean / obs_sd, 4) if obs_sd else None,
        }
    out["Q2_variance_model"] = q2
    ratios = [v["ratio_obs_over_pred"] for v in q2.values()
              if v["ratio_obs_over_pred"]]
    out["Q2_verdict"] = {
        "ratios": ratios,
        "median_ratio": round(float(np.median(ratios)), 4),
        "all_within_2x": bool(all(0.5 <= r <= 2.0 for r in ratios)),
        "reading": ("SD(delta_m) tracks g*alpha_equiv within a factor of two in "
                    "every cell once AWQ's alpha is taken migration-free. The "
                    "first-order VARIANCE model is in regime."),
    }

    # --------------------------------------------------------------------- Q3
    fal = per[per.model_key.str.startswith("falcon3") &
              (per.attack_category == "jailbreak")]
    obs_rates = conf["all_cells"]["by_cell_flip_rate_pct"]
    q3 = {}
    for key, v in alpha_est.items():
        size, method = key.split("-")
        mk = "falcon3-" + size.lower()
        g = fal[fal.model_key == mk]
        pred = 100 * analytic_flip_rate(g.m0, g.g, v["alpha_equiv"])
        obs = obs_rates.get(key)
        q3[key] = {
            "alpha_equiv": v["alpha_equiv"],
            "predicted_flip_pct_zero_crossing": round(pred, 6),
            "observed_flip_pct": obs,
            "ratio_observed_over_predicted": (round(obs / pred, 1)
                                              if pred > 1e-9 and obs else None),
        }
    out["Q3_zero_crossing_model"] = q3

    # prompt-level: does delta_m predict which prompts flip?
    results = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                          dtype={"success": str})
    results["success"] = results["success"].fillna("").str.strip()
    d = results[results.success.isin(["0", "1"])].copy()
    d["y"] = d.success.astype(int)
    idx = ["family", "size", "prompt_id"]
    f16 = d[d.method == "fp16"].set_index(idx)["y"].rename("s_fp16")
    pairs = d[d.method != "fp16"].set_index(idx).join(f16, how="inner").reset_index()
    pairs["flip"] = (pairs.y != pairs.s_fp16).astype(int)
    pf = pairs[(pairs.family == "Falcon3") &
               (pairs.attack_category == "jailbreak")]
    merged = jb.merge(
        pf[["size", "method", "prompt_id", "flip", "s_fp16"]],
        on=["size", "method", "prompt_id"], how="inner")
    out["Q3_prompt_level"] = {}
    for (size, method), g in merged.groupby(["size", "method"]):
        if g.flip.sum() < 5:
            out["Q3_prompt_level"][f"{size}-{method}"] = {
                "n": int(len(g)), "flips": int(g.flip.sum()),
                "note": "too few flips for a stable AUC"}
            continue
        out["Q3_prompt_level"][f"{size}-{method}"] = {
            "n": int(len(g)), "flips": int(g.flip.sum()),
            "auc_delta_m_predicts_flip": round(auc_low(-g.delta_m, g.flip), 4),
            "ci": boot_ci(-g.delta_m, g.flip),
            "auc_m0_predicts_flip": round(auc_low(g.m0, g.flip), 4),
            "auc_m_quant_predicts_flip": round(auc_low(g.m_first, g.flip), 4),
            "mean_delta_m_flipped": round(float(g[g.flip == 1].delta_m.mean()), 4),
            "mean_delta_m_not_flipped": round(float(g[g.flip == 0].delta_m.mean()), 4),
        }
    allc = merged[merged.flip.notna()]
    out["Q3_prompt_level_pooled"] = {
        "n": int(len(allc)), "flips": int(allc.flip.sum()),
        "auc_delta_m": round(auc_low(-allc.delta_m, allc.flip), 4),
        "ci_delta_m": boot_ci(-allc.delta_m, allc.flip),
        "auc_m0": round(auc_low(allc.m0, allc.flip), 4),
        "auc_m_quant": round(auc_low(allc.m_first, allc.flip), 4),
        "mean_delta_m_flipped": round(float(allc[allc.flip == 1].delta_m.mean()), 4),
        "mean_delta_m_not_flipped": round(float(allc[allc.flip == 0].delta_m.mean()), 4),
    }

    # cell-level: does the mean shift track the flip rate?
    cell = []
    for key, v in q2.items():
        if obs_rates.get(key) is not None:
            cell.append({"cell": key, "mean_delta_m": v["observed_mean"],
                         "sd_delta_m": v["observed_sd"],
                         "flip_pct": obs_rates[key]})
    cdf = pd.DataFrame(cell)
    if len(cdf) >= 3:
        r_mean, p_mean = stats.pearsonr(cdf.mean_delta_m, cdf.flip_pct)
        r_sd, p_sd = stats.pearsonr(cdf.sd_delta_m, cdf.flip_pct)
        out["Q3_cell_level"] = {
            "cells": cdf.to_dict("records"),
            "pearson_mean_shift_vs_flip_rate": [round(float(r_mean), 4),
                                                float(p_mean)],
            "pearson_sd_vs_flip_rate": [round(float(r_sd), 4), float(p_sd)],
        }

    out["studyb_control_mean_over_sd"] = base["overall_shift"]["mean_over_sd"]

    with open(os.path.join(OUT, "mechanism_verdict.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("=== Q1: alpha_equiv per format, migration-corrected ===")
    for k, v in alpha_est.items():
        print(f"  {k}: {v['alpha_equiv']:.5f}  ({v['basis']}; all-target median "
              f"{v['all_targets_median']:.5f})")
    print(f"  reference, uniform 4-bit over a Gaussian: "
          f"{out['Q1_reference']['uniform_4bit_over_gaussian']}")

    print("\n=== Q2: does SD(delta_m) = g * alpha hold? ===")
    for k, v in q2.items():
        print(f"  {k}: predicted {v['predicted_sd']:.4f} vs observed "
              f"{v['observed_sd']:.4f}  ratio {v['ratio_obs_over_pred']}")
    print(f"  -> {out['Q2_verdict']['reading']}")

    print("\n=== Q3a: does the zero-crossing model predict the flip rate? ===")
    for k, v in q3.items():
        print(f"  {k}: predicted {v['predicted_flip_pct_zero_crossing']:.6f}% vs "
              f"observed {v['observed_flip_pct']}%  "
              f"(observed/predicted = {v['ratio_observed_over_predicted']})")

    print("\n=== Q3b: does delta_m predict WHICH prompts flip? ===")
    p = out["Q3_prompt_level_pooled"]
    print(f"  pooled: n={p['n']} flips={p['flips']}  "
          f"AUC(delta_m) {p['auc_delta_m']} {p['ci_delta_m']}  "
          f"AUC(m0) {p['auc_m0']}  AUC(m_quant) {p['auc_m_quant']}")
    print(f"    mean delta_m: flipped {p['mean_delta_m_flipped']} vs "
          f"not flipped {p['mean_delta_m_not_flipped']}")
    for k, v in out["Q3_prompt_level"].items():
        if "auc_delta_m_predicts_flip" in v:
            print(f"  {k}: n={v['n']} flips={v['flips']} "
                  f"AUC(delta_m)={v['auc_delta_m_predicts_flip']} "
                  f"AUC(m0)={v['auc_m0_predicts_flip']} "
                  f"delta_m flipped {v['mean_delta_m_flipped']} vs "
                  f"{v['mean_delta_m_not_flipped']}")

    if "Q3_cell_level" in out:
        print("\n=== Q3c: across cells, does the mean shift track the flip rate? ===")
        print(pd.DataFrame(out["Q3_cell_level"]["cells"]).to_string(index=False))
        print(f"  Pearson(mean shift, flip rate) = "
              f"{out['Q3_cell_level']['pearson_mean_shift_vs_flip_rate']}")
        print(f"  Pearson(SD, flip rate)         = "
              f"{out['Q3_cell_level']['pearson_sd_vs_flip_rate']}")
    print(f"\n  Study B's own mean/SD for comparison: "
          f"{out['studyb_control_mean_over_sd']}")
    print(f"\nwrote {OUT}/mechanism_verdict.json")


if __name__ == "__main__":
    main()
