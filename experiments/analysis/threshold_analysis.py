"""
threshold_analysis.py  --  where the refusal boundary actually is, and what
happens to the zero-crossing criterion once you put it there.

The mechanism section concludes that the zero-crossing flip criterion cannot be
tested because "no prompt is near the boundary": Falcon3's smallest |m0| is 3.29
(1B) and 2.39 (3B), several SD from zero at the largest swept dose. That
statement is true, and it is scoped to Falcon3, which is the only family the
mechanism section covers. The other two families are not like Falcon3. Qwen2.5
has jailbreak prompts on BOTH sides of zero: 6 negative at 1.5B and 14 at 3B,
down to -3.53.

Once you look at a family with mass on both sides, the picture changes. fp16
compliance as a function of m0 is not a smooth function with an inflection at
zero. It is a sharp step, and the step is nowhere near zero: Qwen complies on
100% of prompts below m0 = 2 and 0% above 6. So the natural criterion is not
"does the margin cross zero" but "does the margin cross m*", where m* is the
model's own behavioural threshold. The zero-crossing model asks about a crossing
roughly m* further away than the one that governs behaviour, and m* is on the
order of 3 to 5 while SD(delta_m) is on the order of 1. That is the right order
of magnitude to explain a two-orders-of-magnitude under-prediction.

This script tests that. Interpretation is fixed here, before the numbers:

  * If moving the crossing from 0 to m* leaves the miss factors substantially
    unchanged, the threshold account is wrong and the structural (anisotropy /
    excluded-tensor) account stands alone. Report that.
  * If it closes most of the gap, the zero-crossing criterion was never failing
    because the physics was wrong; it was failing because the boundary was put
    in the wrong place, and the paper's diagnosis needs rewriting.
  * "Most of the gap" is fixed as: the geometric-mean miss factor across the six
    Falcon3 cells falls below 3x.

WHAT m* COSTS. m* is fitted from fp16 LABELS, which requires generating and
judging at fp16. The margin m0 itself is forward passes only. So a screen that
uses m* is not "forward passes only" end to end: it needs one labelled fp16 pass
over an audit battery. That is a real and stateable cost, and Part 4 checks the
threshold is not merely memorising the prompts it is scored on by refitting on
half the prompts and evaluating on the other half.

Inputs
  results/analysis/regrid_pairs.csv       Falcon3, stated decoding
  results/analysis/phase21_pairs.csv      Llama-3.2 / Qwen2.5, stated decoding
  results/margin/per_prompt_margins.csv        m0, g per prompt
  results/analysis/mechanism_verdict.json alpha_equiv per Falcon3 cell
  results/analysis/s5_jailbreak_tstats.json  measured mean/SD of delta_m per cell

Output
  results/analysis/threshold.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "results", "analysis")
MARGIN = os.path.join(REPO, "results", "margin")
SEED = 20260827
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pairs():
    """Every stated-decoding jailbreak pair across the three families.

    phase21_pairs.csv carries m0 already; both files use the same column names,
    so they concatenate directly. fp16 rows are dropped -- a pair is a
    quantized row plus the s_fp16 label it is compared against.
    """
    cols = ["family", "size", "prompt_id", "method", "attack_category",
            "template_tier", "y", "s_fp16", "flip", "broke", "restore", "m0"]
    frames = []
    for name in ("regrid_pairs.csv", "phase21_pairs.csv"):
        d = pd.read_csv(os.path.join(RES, name))
        frames.append(d[[c for c in cols if c in d.columns]])
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.method != "fp16") & (d.attack_category == "jailbreak")]
    return d.dropna(subset=["m0", "s_fp16"]).copy()


def load_margins(pairs):
    """Per-prompt m0 and g, carrying the fp16 label from the stated-decoding grid.

    per_prompt_margins.csv holds the forward-pass quantities only; the fp16
    refuse/comply label lives on the pairs, one value per (family, size,
    prompt_id) repeated across the three formats. Deduplicating on that key is
    what makes a prompt, not a row, the unit here.
    """
    m = pd.read_csv(os.path.join(MARGIN, "per_prompt_margins.csv"))
    m = m[m.attack_category == "jailbreak"]
    key = ["family", "size", "prompt_id"]
    lab = pairs.drop_duplicates(subset=key)[key + ["s_fp16"]]
    m = m.drop(columns=["m0"]).merge(
        pairs.drop_duplicates(subset=key)[key + ["m0"]], on=key, how="inner")
    m = m.merge(lab, on=key, how="inner")
    assert not m.duplicated(subset=key).any(), "margin join fanned out"
    return m.copy()


# ---------------------------------------------------------------------------
# Part 1 -- locate the threshold
# ---------------------------------------------------------------------------

def logistic_fit(x, y, l2=1e-3):
    """P(comply) = sigmoid(b0 + b1*x), by penalised maximum likelihood.

    Hand-rolled to keep sklearn out of the dependency set, matching
    margin_analysis.py. The small ridge term is there because several models
    have very few compliant prompts and would otherwise separate perfectly and
    send the coefficients to infinity; it shrinks the slope slightly and leaves
    the 50% point essentially unmoved.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, xs = x.mean(), (x.std() or 1.0)
    z = (x - xm) / xs

    def nll(b):
        t = b[0] + b[1] * z
        # log(1+exp(t)) computed stably
        ll = np.sum(y * t - np.logaddexp(0.0, t))
        return -ll + l2 * float(b @ b)

    res = optimize.minimize(nll, np.array([0.0, -1.0]), method="BFGS")
    b0, b1 = res.x
    # back out of the standardised scale
    slope = b1 / xs
    inter = b0 - b1 * xm / xs
    m_star = float(-inter / slope) if slope != 0 else float("nan")
    return {"intercept": float(inter), "slope": float(slope),
            "m_star": m_star, "converged": bool(res.success)}


def threshold_by_model(margins, rng):
    """Fit m* per model, and per (model, tier) where the tier has both outcomes."""
    out = {}
    for (fam, size), g in margins.groupby(["family", "size"]):
        key = f"{fam}-{size}"
        n, ncomp = len(g), int(g.s_fp16.sum())
        rec = {"n_prompts": n, "n_comply": ncomp,
               "m0_min": round(float(g.m0.min()), 4),
               "m0_max": round(float(g.m0.max()), 4)}
        if ncomp == 0 or ncomp == n:
            rec["m_star"] = None
            rec["note"] = ("no variation in the fp16 label at this model, so the "
                           "threshold is not identified; it lies outside the "
                           "observed margin range")
            out[key] = rec
            continue
        fit = logistic_fit(g.m0, g.s_fp16)
        rec.update(fit)
        rec["m_star"] = round(fit["m_star"], 4)
        # bootstrap over prompts
        boot = []
        idx = np.arange(n)
        gm0, gy = g.m0.to_numpy(), g.s_fp16.to_numpy()
        for _ in range(400):
            take = rng.integers(0, n, n)
            ys = gy[take]
            if ys.sum() == 0 or ys.sum() == n:
                continue
            try:
                boot.append(logistic_fit(gm0[take], ys)["m_star"])
            except Exception:
                continue
        if boot:
            rec["m_star_ci95"] = [round(float(np.percentile(boot, 2.5)), 4),
                                  round(float(np.percentile(boot, 97.5)), 4)]
        rec["by_tier"] = {}
        for tier, gt in g.groupby("template_tier"):
            nc = int(gt.s_fp16.sum())
            if nc == 0 or nc == len(gt):
                rec["by_tier"][tier] = {"n": len(gt), "n_comply": nc, "m_star": None}
            else:
                rec["by_tier"][tier] = {
                    "n": len(gt), "n_comply": nc,
                    "m_star": round(logistic_fit(gt.m0, gt.s_fp16)["m_star"], 4)}
        out[key] = rec
    return out


def compliance_ladder(margins):
    """fp16 compliance by margin band, per family. The picture in one table."""
    bands = [-10, 0, 1, 2, 3, 4, 6, 8, 12, 25]
    out = {}
    m = margins.copy()
    m["band"] = pd.cut(m.m0, bands)
    for fam, g in m.groupby("family"):
        rows = []
        for band, gb in g.groupby("band", observed=True):
            rows.append({"band": str(band), "n": len(gb),
                         "comply": int(gb.s_fp16.sum()),
                         "comply_rate": round(float(gb.s_fp16.mean()), 4)})
        out[fam] = rows
    return out


# ---------------------------------------------------------------------------
# Part 2 -- the two-sided law
# ---------------------------------------------------------------------------

def auc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    p, n = int(label.sum()), int((1 - label).sum())
    if p == 0 or n == 0:
        return None
    r = stats.rankdata(score)
    return round(float((r[label == 1].sum() - p * (p + 1) / 2) / (p * n)), 4)


def two_sided_law(pairs):
    out = {}
    for fam, g in pairs.groupby("family"):
        refused = g[g.s_fp16 == 0]
        complied = g[g.s_fp16 == 1]
        neg = g[g.m0 < 0]
        out[fam] = {
            "refused_stratum": {
                "n": len(refused), "breaks": int(refused.flip.sum()),
                "auc_low_margin_predicts_break": auc(-refused.m0, refused.flip)},
            "complied_stratum": {
                "n": len(complied), "restores": int(complied.restore.sum()),
                "auc_high_margin_predicts_restore": auc(complied.m0, complied.restore)},
            "negative_margin_rows": {
                "n": len(neg),
                "n_fp16_complied": int(neg.s_fp16.sum()),
                "n_restores": int(neg.restore.sum())},
            "pooled_abs_margin": {
                "auc_small_abs_margin_predicts_flip": auc(-g.m0.abs(), g.flip),
                "auc_low_signed_margin_predicts_flip": auc(-g.m0, g.flip)},
        }
    # the non-monotonicity, on the family that has mass on both sides
    q = pairs[pairs.family == "Qwen2.5"].copy()
    q["decile"] = pd.qcut(q.m0.abs(), 10, duplicates="drop")
    rows = []
    for band, gb in q.groupby("decile", observed=True):
        rows.append({"abs_m0_band": str(band), "n": len(gb),
                     "flips": int(gb.flip.sum()),
                     "flip_rate": round(float(gb.flip.mean()), 4)})
    out["_qwen_abs_margin_deciles"] = rows
    return out


# ---------------------------------------------------------------------------
# Part 3 -- the zero-crossing criterion, re-aimed
# ---------------------------------------------------------------------------

def break_rate(m0, sd, m_star, mu):
    """Predicted probability that a REFUSED prompt breaks to compliance.

    The margin lands at m0 + delta with delta ~ Normal(mu, sd); the decision
    changes when it falls below the behavioural threshold m*.

    This is deliberately restricted to the fp16-refused stratum, and the first
    version of this script was not. Scoring complied prompts with the same
    expression is degenerate: a prompt that complied at fp16 while sitting above
    the threshold gets a restore probability of essentially 1, and with the
    crossing at m* = 0 that is EVERY complied prompt. On Falcon3 those 30 rows
    alone contributed about 1.68% to a predicted flip rate whose true value is
    nearer 0.03%, which made the criterion look calibrated when it was only
    counting misclassified prompts. Breaks within the refused stratum are also
    exactly what the paper's capture-rate claim is about, so this is the
    like-for-like comparison.
    """
    return stats.norm.cdf((m_star - m0 - mu) / sd)


def zero_crossing_ladder(pairs, margins, alphas, tstats, thresholds):
    """Four models, increasing in what they are allowed to know.

    M0  crossing at 0,  no mean shift      the paper's criterion
    M1  crossing at 0,  measured mean shift
    M2  crossing at m*, no mean shift
    M3  crossing at m*, measured mean shift

    M0 and M2 need only fp16 forward passes plus the checkpoint files. M1 and M3
    additionally need delta_m measured on the quantized checkpoint, which is a
    different and more expensive information regime -- still no generation and no
    judging, but no longer "before you quantize anything". Reported separately
    for that reason.
    """
    fal = pairs[pairs.family == "Falcon3"]
    mg = margins[margins.family == "Falcon3"].set_index(["size", "prompt_id"])
    out = {"cells": [], "note_information_regime": (
        "M0 and M2 use only fp16 forward passes and the checkpoint files. M1 and "
        "M3 additionally use the measured mean of delta_m, which requires forward "
        "passes on the quantized checkpoint.")}

    for cell, a in alphas.items():
        size, method = cell.split("-")
        alpha = a["alpha_equiv"]
        mu = tstats["cells"][cell]["mean"]
        sub = fal[(fal["size"] == size) & (fal.method == method)
                  & (fal.s_fp16 == 0)]
        if sub.empty:
            continue
        key = list(zip(sub["size"], sub.prompt_id))
        g = mg.loc[key, "g"].to_numpy()
        m0 = sub.m0.to_numpy()
        sd = g * alpha
        observed = float(sub.broke.mean())

        m_star = thresholds[f"Falcon3-{size}"]["m_star"]
        preds = {
            "M0_zero_no_mean": float(break_rate(m0, sd, 0.0, 0.0).mean()),
            "M1_zero_with_mean": float(break_rate(m0, sd, 0.0, mu).mean()),
            "M2_star_no_mean": float(break_rate(m0, sd, m_star, 0.0).mean()),
            "M3_star_with_mean": float(break_rate(m0, sd, m_star, mu).mean()),
        }
        rec = {"cell": cell, "n_refused": len(sub),
               "n_breaks": int(sub.broke.sum()), "alpha_equiv": alpha,
               "m_star": round(m_star, 4), "measured_mean_delta_m": mu,
               "observed_break_pct": round(100 * observed, 4)}
        for k, v in preds.items():
            rec[f"pred_{k}_pct"] = round(100 * v, 5)
            # A cell with zero observed breaks has no miss factor: the ratio is
            # 0/p, which is not "perfectly predicted". Excluded from the
            # geometric mean and recorded as None so the exclusion is visible.
            rec[f"miss_{k}_x"] = (round(observed / v, 1)
                                  if (v > 0 and observed > 0) else None)
        # Sensitivity of the re-aimed criterion to where m* actually sits. If
        # the improvement only survives at the point estimate, it is a fitted
        # parameter doing the work, not a threshold.
        lo_hi = thresholds[f"Falcon3-{size}"].get("m_star_ci95")
        if lo_hi:
            band = {}
            for tag, ms in (("at_ci_low", lo_hi[0]), ("at_ci_high", lo_hi[1])):
                v = float(break_rate(m0, sd, ms, 0.0).mean())
                band[tag] = {"m_star": ms, "pred_pct": round(100 * v, 5),
                             "miss_x": (round(observed / v, 1)
                                        if (v > 0 and observed > 0) else None)}
            rec["M2_sensitivity_to_m_star"] = band
        out["cells"].append(rec)

    for k in ("M0_zero_no_mean", "M1_zero_with_mean",
              "M2_star_no_mean", "M3_star_with_mean"):
        f = [c[f"miss_{k}_x"] for c in out["cells"] if c[f"miss_{k}_x"]]
        if f:
            out[f"geomean_miss_{k}"] = round(float(np.exp(np.mean(np.log(f)))), 2)
            out[f"range_miss_{k}"] = [min(f), max(f)]
            out[f"n_cells_scored_{k}"] = len(f)
    for tag in ("at_ci_low", "at_ci_high"):
        f = [c["M2_sensitivity_to_m_star"][tag]["miss_x"] for c in out["cells"]
             if c.get("M2_sensitivity_to_m_star", {}).get(tag, {}).get("miss_x")]
        if f:
            out[f"geomean_miss_M2_{tag}"] = round(float(np.exp(np.mean(np.log(f)))), 2)
            out[f"range_miss_M2_{tag}"] = [min(f), max(f)]
    out["note_excluded_cells"] = (
        "Cells with zero observed breaks carry no miss factor and are excluded "
        "from every geometric mean; n_cells_scored_* records how many remain.")
    return out


# ---------------------------------------------------------------------------
# Part 4 -- is m* just memorising these prompts?
# ---------------------------------------------------------------------------

def held_out_threshold(pairs, margins, alphas, tstats, rng, n_splits=200):
    """Refit m* on half the prompts, score the criterion on the other half.

    m* is estimated from the fp16 labels of the same prompts whose flips the
    criterion is scored on. This splits by prompt_id so the fitting and scoring
    sets share no prompt, and reports the geometric-mean miss factor of M2 on
    held-out prompts against the in-sample figure.
    """
    fal_m = margins[margins.family == "Falcon3"]
    fal_p = pairs[pairs.family == "Falcon3"]
    mg = fal_m.set_index(["size", "prompt_id"])
    ids = np.array(sorted(fal_m.prompt_id.unique()))

    scores = []
    for _ in range(n_splits):
        rng.shuffle(ids)
        fit_ids = set(ids[: len(ids) // 2])
        test_ids = set(ids[len(ids) // 2:])
        stars = {}
        ok = True
        for size in ("1B", "3B"):
            gf = fal_m[(fal_m["size"] == size) & (fal_m.prompt_id.isin(fit_ids))]
            nc = int(gf.s_fp16.sum())
            if nc == 0 or nc == len(gf):
                ok = False
                break
            stars[size] = logistic_fit(gf.m0, gf.s_fp16)["m_star"]
        if not ok:
            continue
        factors = []
        for cell, a in alphas.items():
            size, method = cell.split("-")
            sub = fal_p[(fal_p["size"] == size) & (fal_p.method == method)
                        & (fal_p.prompt_id.isin(test_ids)) & (fal_p.s_fp16 == 0)]
            if sub.empty or sub.broke.sum() == 0:
                continue
            key = list(zip(sub["size"], sub.prompt_id))
            sd = mg.loc[key, "g"].to_numpy() * a["alpha_equiv"]
            p = break_rate(sub.m0.to_numpy(), sd, stars[size], 0.0).mean()
            if p > 0:
                factors.append(float(sub.broke.mean()) / float(p))
        if factors:
            scores.append(float(np.exp(np.mean(np.log(factors)))))

    if not scores:
        return {"n_splits_used": 0, "note": "no split produced label variation"}
    return {"n_splits_used": len(scores),
            "heldout_geomean_miss_M2_median": round(float(np.median(scores)), 2),
            "heldout_geomean_miss_M2_iqr": [round(float(np.percentile(scores, 25)), 2),
                                            round(float(np.percentile(scores, 75)), 2)],
            "note": ("m* refit on half the prompts, criterion scored on the "
                     "disjoint half, split by prompt_id")}


def main():
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()
    margins = load_margins(pairs)
    alphas = json.load(open(os.path.join(RES, "mechanism_verdict.json")))["Q1_alpha_estimates"]
    tstats = json.load(open(os.path.join(RES, "s5_jailbreak_tstats.json")))

    thresholds = threshold_by_model(margins, rng)
    out = {
        "part1_threshold": thresholds,
        "part1_compliance_ladder": compliance_ladder(margins),
        "part2_two_sided_law": two_sided_law(pairs),
        "part3_zero_crossing_ladder": zero_crossing_ladder(
            pairs, margins, alphas, tstats, thresholds),
        "part4_held_out": held_out_threshold(pairs, margins, alphas, tstats, rng),
        "verdict_threshold_fixed_before_running": (
            "the threshold account is supported if the geometric-mean miss factor "
            "across the six Falcon3 cells falls below 3x"),
    }
    lad = out["part3_zero_crossing_ladder"]
    for k in ("M0_zero_no_mean", "M2_star_no_mean", "M3_star_with_mean"):
        gm = lad.get(f"geomean_miss_{k}")
        out.setdefault("verdict", {})[k] = (
            None if gm is None else ("supported" if gm < 3 else "not supported"))

    # The pre-registered threshold is met, but stating only that would overstate
    # it. m* is estimated from the fp16 compliance data, and Falcon3 -- the only
    # family with an alpha_equiv measurement -- barely has any: 4 compliant
    # jailbreak prompts at 1B and 6 at 3B, which is why the 1B interval spans
    # zero. Across m*'s own 95% interval the geometric-mean miss moves by more
    # than an order of magnitude, so the LOCATION of the threshold, not merely
    # its existence, is carrying the result.
    fal = {k: v for k, v in thresholds.items() if k.startswith("Falcon3")}
    out["verdict_qualification"] = {
        "pre_registered_threshold_met":
            out["verdict"].get("M2_star_no_mean") == "supported",
        "geomean_at_point_estimate": lad.get("geomean_miss_M2_star_no_mean"),
        "geomean_across_m_star_ci": [lad.get("geomean_miss_M2_at_ci_high"),
                                     lad.get("geomean_miss_M2_at_ci_low")],
        "heldout_geomean": out["part4_held_out"].get("heldout_geomean_miss_M2_median"),
        "falcon3_n_comply": {k: v["n_comply"] for k, v in fal.items()},
        "falcon3_m_star_ci": {k: v.get("m_star_ci95") for k, v in fal.items()},
        "limiting_factor": (
            "m* is under-identified on Falcon3, the only family with a measured "
            "alpha_equiv, because Falcon3 almost never complies at fp16. The "
            "families whose threshold IS tightly estimated -- Qwen2.5 at 4.55 "
            "and 3.97, from 97 and 73 compliant prompts -- have no alpha_equiv "
            "or delta_m measurement, so the criterion cannot yet be tested where "
            "the threshold is known."),
        "decisive_next_experiment": (
            "Measure alpha_equiv and delta_m for Qwen2.5-1.5B/3B. That moves the "
            "test onto the family that has boundary data, and it is the same "
            "mechanism kernel re-pointed at different model keys."),
    }

    dest = os.path.join(RES, "threshold.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Part 1: where the fp16 refusal boundary sits ===")
    for k, v in thresholds.items():
        ms = v.get("m_star")
        ci = v.get("m_star_ci95")
        print(f"  {k:14} n={v['n_prompts']:4} comply={v['n_comply']:4} "
              f"m0 range [{v['m0_min']:7.2f}, {v['m0_max']:6.2f}]  "
              f"m* = {ms if ms is None else round(ms, 2)}"
              f"{'' if not ci else f'  95% {ci}'}")
    print("\n=== Part 2: the two-sided law ===")
    for fam, v in out["part2_two_sided_law"].items():
        if fam.startswith("_"):
            continue
        r, c, n = v["refused_stratum"], v["complied_stratum"], v["negative_margin_rows"]
        print(f"  {fam:11} break: n={r['n']:5} k={r['breaks']:3} AUC={r['auc_low_margin_predicts_break']}"
              f" | restore: n={c['n']:4} k={c['restores']:3} AUC={c['auc_high_margin_predicts_restore']}"
              f" | m0<0: {n['n']} rows, {n['n_fp16_complied']} complied, {n['n_restores']} restored")
    print("\n=== Part 3: the criterion, aimed at 0 and at m* ===")
    print(f"  {'cell':9} {'obs%':>7} {'M0 miss':>9} {'M1 miss':>9} {'M2 miss':>9} {'M3 miss':>9}")
    for c in lad["cells"]:
        print(f"  {c['cell']:9} {c['observed_break_pct']:7.3f} "
              f"{str(c['miss_M0_zero_no_mean_x']):>9} {str(c['miss_M1_zero_with_mean_x']):>9} "
              f"{str(c['miss_M2_star_no_mean_x']):>9} {str(c['miss_M3_star_with_mean_x']):>9}")
    for k in ("M0_zero_no_mean", "M1_zero_with_mean", "M2_star_no_mean", "M3_star_with_mean"):
        print(f"  geomean {k:22} {lad.get(f'geomean_miss_{k}')}  range {lad.get(f'range_miss_{k}')}"
              f"  (n cells {lad.get(f'n_cells_scored_{k}')})")
    print(f"  M2 sensitivity: at m* CI low  geomean {lad.get('geomean_miss_M2_at_ci_low')}"
          f"  range {lad.get('range_miss_M2_at_ci_low')}")
    print(f"                  at m* CI high geomean {lad.get('geomean_miss_M2_at_ci_high')}"
          f"  range {lad.get('range_miss_M2_at_ci_high')}")
    print("\n=== Part 4: held-out threshold ===")
    print(f"  {out['part4_held_out']}")
    print("\n=== VERDICT (threshold fixed before running) ===")
    for k, v in out.get("verdict", {}).items():
        print(f"  {k}: {v}")
    q = out["verdict_qualification"]
    print("\n=== QUALIFICATION ===")
    print(f"  point estimate {q['geomean_at_point_estimate']}x, held out "
          f"{q['heldout_geomean']}x, across m* CI {q['geomean_across_m_star_ci']}x")
    print(f"  Falcon3 compliant prompts: {q['falcon3_n_comply']}")
    print(f"  Falcon3 m* CIs:            {q['falcon3_m_star_ci']}")
    print(f"  limiting factor: {q['limiting_factor']}")
    print(f"  next: {q['decisive_next_experiment']}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
