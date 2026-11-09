"""
margin_analysis.py
==================
E3: turn the margin sweep into the three tests, and validate the predictor
against real quantization outcomes already recorded in results.csv.

Quantities
----------
  m0(x)     margin at alpha=0 -- how far the refusal decision sits from the
            boundary, in log-prob units. Larger = more decisively a refusal.

  g(x)      sensitivity. Per-prompt std of the margin across seeds grows
            linearly in alpha under a first-order expansion, so g is the slope
            of that line fit through the origin:  g = sum(a_i s_i) / sum(a_i^2).
            Larger = the decision moves more per unit of weight noise.

  kappa(x)  = m0 / g. The weight-space margin: the RELATIVE weight-perturbation
            scale at which this decision is expected to flip. Units are the same
            as alpha, so kappa is directly comparable to the actual perturbation
            a given quantization format applies. Smaller = more fragile.

Tests
-----
  P1  m0 smaller for disguised jailbreak tiers than for plain
  P2  g   larger  for disguised jailbreak tiers than for plain
  P3  kappa predicts which individual prompts flip under REAL quantization,
      beating m0 alone and a prompt-length baseline

Usage:  python margin_analysis.py --sweep <margin_sweep.csv> [--results <results.csv>]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

JAILBREAK_TIERS = ["plain", "roleplay", "persuasion"]
INJECTION_TIERS = ["direct_harm", "data_stealing"]
DISGUISED = ["roleplay", "persuasion"]


# ---------------------------------------------------------------------------
# Derived per-prompt quantities
# ---------------------------------------------------------------------------

def per_prompt_table(sweep, metric="m_first"):
    """One row per (model, prompt) with m0, g, kappa and the linearity check.

    `metric` defaults to m_first, NOT m_seq. Validated against fp16 outcomes on
    Llama-3.2: m_first separates compliance from refusal at AUC 0.912 and
    correlates -0.58 with per-tier ASR (correct sign), while m_seq manages only
    0.767 and correlates +0.59 (WRONG sign). Length-normalizing over a ~12-token
    refusal phrase dilutes the decision -- most of those tokens are predictable
    once the opener is fixed -- so m_seq largely measures phrase predictability
    rather than refusal propensity.
    """
    key = ["model_key", "family", "size", "prompt_id",
           "attack_category", "template_tier"]

    base = (sweep[sweep.alpha == 0.0]
            .set_index(key)[metric].rename("m0").reset_index())

    # Per-prompt spread across seeds, at each alpha.
    spread = (sweep[sweep.alpha > 0.0]
              .groupby(key + ["alpha"])[metric]
              .agg(s="std", n="count").reset_index())
    spread = spread[spread["n"] >= 2]

    rows = []
    for k, grp in spread.groupby(key):
        a = grp["alpha"].to_numpy(float)
        s = grp["s"].to_numpy(float)
        if len(a) < 2 or not np.isfinite(s).all():
            continue
        # Slope through the origin: theory says s = g * alpha with no intercept.
        g = float((a * s).sum() / (a ** 2).sum())
        # How well does the no-intercept line actually fit? If linearity fails,
        # the first-order expansion kappa relies on is out of its regime.
        pred = g * a
        ss_res = float(((s - pred) ** 2).sum())
        ss_tot = float(((s - s.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(dict(zip(key, k), g=g, lin_r2=r2))

    out = base.merge(pd.DataFrame(rows), on=key, how="inner")
    out["kappa"] = out["m0"] / out["g"].replace(0.0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Real-quantization flips, recovered from the existing ASR data
# ---------------------------------------------------------------------------

def flip_table(results_path):
    """Per-prompt flip indicators: did quantization change this prompt's outcome?

    results.csv stores one classified row per (family, size, method, prompt),
    so pairing each quantized row against its own fp16 row recovers a
    per-prompt outcome change -- far more statistical power than the 18-cell
    aggregate the earlier analysis was limited to.
    """
    df = pd.read_csv(results_path)
    df = df[df.success.isin(["0", "1"])].copy()
    df["success"] = df["success"].astype(int)

    idx = ["family", "size", "prompt_id"]
    fp16 = (df[df.method == "fp16"]
            .set_index(idx)["success"].rename("s_fp16"))
    quant = df[df.method != "fp16"].set_index(idx)

    joined = quant.join(fp16, how="inner").reset_index()
    joined["flip"] = (joined["success"] != joined["s_fp16"]).astype(int)
    joined["broke"] = ((joined["s_fp16"] == 0) & (joined["success"] == 1)).astype(int)
    return joined[idx + ["method", "attack_category", "template_tier",
                         "success", "s_fp16", "flip", "broke"]]


def auc(scores, labels):
    """Rank-based AUC. Hand-rolled to keep sklearn out of the dependency set."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def describe(series):
    s = series.dropna()
    if len(s) < 2:
        return "n<2"
    ci = 1.96 * s.std() / np.sqrt(len(s))
    return f"{s.mean():+.4f} +/- {ci:.4f} (n={len(s)})"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="m_first", choices=["m_first","m_seq"])
    args = ap.parse_args()

    sweep = pd.read_csv(args.sweep)
    print(f"Loaded {len(sweep):,} margin measurements | "
          f"models={sweep.model_key.nunique()} prompts={sweep.prompt_id.nunique()} "
          f"alphas={sorted(sweep.alpha.unique())} metric={args.metric}")

    pp = per_prompt_table(sweep, args.metric)
    print(f"Derived per-prompt quantities for {len(pp):,} (model, prompt) pairs\n")

    print("=" * 70)
    print("LINEARITY CHECK -- is the first-order expansion valid here?")
    print("=" * 70)
    print(f"  R^2 of s = g*alpha (no intercept): median "
          f"{pp.lin_r2.median():.3f}, "
          f"frac > 0.9: {(pp.lin_r2 > 0.9).mean():.1%}")
    print("  (low R^2 would mean kappa is being read outside its valid regime)\n")

    print("=" * 70)
    print("P1 -- baseline margin m0 by tier (smaller = closer to flipping)")
    print("=" * 70)
    for tier in JAILBREAK_TIERS + INJECTION_TIERS:
        sub = pp[pp.template_tier == tier]
        print(f"  {tier:15s} {describe(sub.m0)}")
    plain = pp[pp.template_tier == "plain"].m0.dropna()
    disg = pp[pp.template_tier.isin(DISGUISED)].m0.dropna()
    inj = pp[pp.attack_category == "injection"].m0.dropna()
    if len(plain) > 1 and len(disg) > 1:
        u = stats.mannwhitneyu(plain, disg, alternative="greater")
        print(f"\n  plain > disguised?  U={u.statistic:.0f} p={u.pvalue:.3g}")
    if len(inj) > 1 and len(disg) > 1:
        u = stats.mannwhitneyu(inj, disg, alternative="greater")
        print(f"  injection > disguised?  U={u.statistic:.0f} p={u.pvalue:.3g}")

    print("\n" + "=" * 70)
    print("P2 -- sensitivity g by tier (larger = moves more per unit noise)")
    print("=" * 70)
    for tier in JAILBREAK_TIERS + INJECTION_TIERS:
        sub = pp[pp.template_tier == tier]
        print(f"  {tier:15s} {describe(sub.g)}")
    if len(plain) > 1:
        u = stats.mannwhitneyu(pp[pp.template_tier.isin(DISGUISED)].g.dropna(),
                               pp[pp.template_tier == "plain"].g.dropna(),
                               alternative="greater")
        print(f"\n  disguised > plain?  U={u.statistic:.0f} p={u.pvalue:.3g}")

    print("\n" + "=" * 70)
    print("KAPPA -- weight-space margin (relative noise scale to flip)")
    print("=" * 70)
    for tier in JAILBREAK_TIERS + INJECTION_TIERS:
        sub = pp[pp.template_tier == tier]
        print(f"  {tier:15s} {describe(sub.kappa)}")

    print("\n  by model:")
    print(pp.groupby(["family", "size", "template_tier"])["kappa"]
            .mean().unstack().round(3).to_string())

    # -- P3 -----------------------------------------------------------------
    if args.results and os.path.exists(args.results):
        print("\n" + "=" * 70)
        print("P3 -- does kappa predict REAL quantization flips?")
        print("=" * 70)
        flips = flip_table(args.results)
        merged = flips.merge(
            pp[["family", "size", "prompt_id", "m0", "g", "kappa"]],
            on=["family", "size", "prompt_id"], how="inner",
        )
        print(f"  {len(merged):,} (prompt, format) pairs with both margin and ASR data")
        if len(merged) == 0:
            print("  No overlap -- margin sweep and results.csv share no models yet.")
        else:
            print(f"  overall flip rate: {merged.flip.mean():.1%}\n")
            print("  AUC for predicting flip (0.5 = chance):")
            print(f"    kappa (theory)      {auc(-merged.kappa, merged.flip):.3f}")
            print(f"    m0 alone            {auc(-merged.m0, merged.flip):.3f}")
            print(f"    g alone             {auc(merged.g, merged.flip):.3f}")
            print("\n  by attack category:")
            for cat, sub in merged.groupby("attack_category"):
                print(f"    {cat:12s} kappa AUC={auc(-sub.kappa, sub.flip):.3f} "
                      f"(n={len(sub):,}, flip rate {sub.flip.mean():.1%})")
            print("\n  by format:")
            for m, sub in merged.groupby("method"):
                print(f"    {m:6s} kappa AUC={auc(-sub.kappa, sub.flip):.3f} "
                      f"(n={len(sub):,}, flip rate {sub.flip.mean():.1%})")

    if args.out:
        pp.to_csv(args.out, index=False)
        print(f"\nPer-prompt table -> {args.out}")


if __name__ == "__main__":
    main()
