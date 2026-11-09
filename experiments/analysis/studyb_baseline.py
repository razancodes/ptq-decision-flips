"""
studyb_baseline.py  --  the isotropic control (task 1.4 step 4) and the
dose-response curve that task 1.3 step 5 has to be read against.

Two things come out of the existing 62,400-row alpha sweep, both needed before
any real-format number can be interpreted:

1. E[delta_m(alpha)] under Study B's perturbation.
   Study B added W += N(0, (alpha*std(W))^2), which is symmetric by
   construction, so the expected margin shift must be ~0 and the whole effect
   must live in the variance. This is not a formality: it is the control that
   makes a non-zero mean shift on real checkpoints a finding rather than an
   artefact of how margins are measured. If this came back non-zero, the
   measurement itself would be suspect.

2. The dose-response curve for MARGIN-SIGN flips, P(sign(m(alpha)) != sign(m0)).
   §5's mechanism is about the margin crossing zero, and this is the curve to
   read at the measured alpha_equiv.

   An honest caveat that has to travel with any comparison against observed
   flip rates: a margin-sign flip is a FIRST-TOKEN decision flip, whereas an
   observed flip in §4.7 is a change in the judge's safety label for a whole
   100-token reply. They are related -- §4.1 is the evidence that the first
   token carries the decision -- but they are not the same measurement, and the
   dose-response comparison is therefore approximate in a way no amount of care
   about alpha_equiv can fix.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")


def main():
    os.makedirs(OUT, exist_ok=True)
    sweep = pd.read_csv(os.path.join(REPO, "results", "margin",
                                     "margin_sweep_all.csv"))
    base = (sweep[sweep.alpha == 0]
            .set_index(["model_key", "prompt_id"])["m_first"].rename("m0"))
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))

    pert = sweep[sweep.alpha > 0].copy()
    pert = pert.join(base, on=["model_key", "prompt_id"])
    pert["delta_m"] = pert.m_first - pert.m0
    pert["sign_flip"] = (np.sign(pert.m_first) != np.sign(pert.m0)).astype(int)
    # refusal-losing direction only: m0 > 0 (refusal preferred) going negative
    pert["refusal_lost"] = ((pert.m0 > 0) & (pert.m_first <= 0)).astype(int)

    out = {"n_perturbed_rows": int(len(pert)),
           "alphas": sorted(pert.alpha.unique().tolist()),
           "seeds": sorted(pert.seed.unique().tolist()),
           "models": sorted(pert.model_key.unique().tolist())}

    # ------------------------------------------------- 1. mean shift is ~zero
    def shift_stats(g):
        d = g.delta_m.dropna()
        n = len(d)
        mean, sd = float(d.mean()), float(d.std())
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        t = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
        return pd.Series({"n": n, "mean_delta_m": mean, "sd_delta_m": sd,
                          "se": se, "t": t,
                          "mean_over_sd": mean / sd if sd > 0 else float("nan")})

    overall = shift_stats(pert)
    out["overall_shift"] = {k: (None if pd.isna(v) else float(v))
                            for k, v in overall.items()}
    by_alpha = pert.groupby("alpha").apply(shift_stats, include_groups=False)
    out["shift_by_alpha"] = by_alpha.round(6).reset_index().to_dict("records")
    by_model = pert.groupby("model_key").apply(shift_stats, include_groups=False)
    out["shift_by_model"] = by_model.round(6).reset_index().to_dict("records")
    by_tier = pert.groupby("template_tier").apply(shift_stats, include_groups=False)
    out["shift_by_tier"] = by_tier.round(6).reset_index().to_dict("records")
    falcon = pert[pert.model_key.str.startswith("falcon3")]
    out["shift_falcon3_by_alpha"] = (falcon.groupby("alpha")
                                     .apply(shift_stats, include_groups=False)
                                     .round(6).reset_index().to_dict("records"))

    # ------------------------------------------- 2. dose-response flip curve
    curve = (pert.groupby(["model_key", "alpha"])
             .agg(n=("sign_flip", "size"),
                  sign_flip_rate=("sign_flip", "mean"),
                  refusal_lost_rate=("refusal_lost", "mean"))
             .reset_index())
    curve["sign_flip_pct"] = (100 * curve.sign_flip_rate).round(4)
    curve["refusal_lost_pct"] = (100 * curve.refusal_lost_rate).round(4)
    out["dose_response"] = curve[["model_key", "alpha", "n", "sign_flip_pct",
                                  "refusal_lost_pct"]].to_dict("records")

    tier_curve = (pert[pert.model_key.str.startswith("falcon3")]
                  .groupby(["model_key", "alpha", "template_tier"])
                  .sign_flip.mean().mul(100).round(4).reset_index())
    out["dose_response_falcon3_by_tier"] = tier_curve.to_dict("records")

    # jailbreak-only curve, to match §4.7's scope
    jb = pert[(pert.attack_category == "jailbreak")]
    jb_curve = (jb.groupby(["model_key", "alpha"])
                .agg(n=("sign_flip", "size"),
                     sign_flip_pct=("sign_flip", lambda s: round(100 * s.mean(), 4)),
                     refusal_lost_pct=("refusal_lost",
                                       lambda s: round(100 * s.mean(), 4)))
                .reset_index())
    out["dose_response_jailbreak_only"] = jb_curve.to_dict("records")
    jb_curve.to_csv(os.path.join(OUT, "dose_response_jailbreak.csv"), index=False)

    # ------------------------------- 3. does m0 predict Study B sign flips?
    def auc_low(m0, flip):
        m0 = np.asarray(m0, float)
        flip = np.asarray(flip, int)
        ok = np.isfinite(m0)
        m0, flip = -m0[ok], flip[ok]
        p, n = int(flip.sum()), int((1 - flip).sum())
        if p == 0 or n == 0:
            return float("nan")
        r = stats.rankdata(m0)
        return float((r[flip == 1].sum() - p * (p + 1) / 2) / (p * n))

    rows = []
    for (mk, a), g in jb.groupby(["model_key", "alpha"]):
        rows.append({"model_key": mk, "alpha": a, "n": int(len(g)),
                     "n_flips": int(g.sign_flip.sum()),
                     "auc_m0_predicts_signflip": round(auc_low(g.m0, g.sign_flip), 4)})
    out["m0_predicts_studyb_signflip"] = rows

    # ------------------------------- 4. linearity / g, carried over for §5
    out["g_summary"] = {
        "median_lin_r2": round(float(per.lin_r2.median()), 4),
        "frac_lin_r2_above_0.9": round(float((per.lin_r2 > 0.9).mean()), 4),
        "g_by_model": {k: round(float(v), 4) for k, v in
                       per.groupby("model_key").g.median().items()},
    }

    with open(os.path.join(OUT, "studyb_baseline.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    # ------------------------------------------------------------------ report
    print("=== 1. Study B mean shift (the isotropic control) ===")
    print(f"  {out['n_perturbed_rows']:,} perturbed measurements, "
          f"alphas={out['alphas']}")
    o = out["overall_shift"]
    print(f"  overall: mean delta_m = {o['mean_delta_m']:+.5f}, "
          f"SD = {o['sd_delta_m']:.5f}, mean/SD = {o['mean_over_sd']:+.5f}, "
          f"t = {o['t']:+.2f}")
    print("  by alpha:")
    print(by_alpha.round(5).to_string())
    print("  Falcon3 only, by alpha:")
    print(falcon.groupby("alpha").apply(shift_stats, include_groups=False)
          .round(5).to_string())
    print("  by tier:")
    print(by_tier.round(5).to_string())

    print("\n=== 2. dose-response: margin-sign flips, jailbreak only ===")
    print(jb_curve.to_string(index=False))

    print("\n=== 3. does m0 predict Study B sign flips (jailbreak)? ===")
    for r in rows:
        if r["model_key"].startswith("falcon3"):
            print(f"  {r['model_key']} alpha={r['alpha']}: "
                  f"AUC={r['auc_m0_predicts_signflip']} "
                  f"({r['n_flips']}/{r['n']} flips)")

    print("\n=== 4. first-order regime ===")
    print(f"  {out['g_summary']}")
    print(f"\nwrote {OUT}/studyb_baseline.json and dose_response_jailbreak.csv")


if __name__ == "__main__":
    main()
