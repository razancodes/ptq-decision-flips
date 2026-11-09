"""
index_stats.py  --  what the first-token index can honestly be said to do.

Everything the poster claims about the index itself, computed once, with the
falsifying controls next to each claim rather than in an appendix.

Six blocks:

1. RANKING, per model, with intervals clustered on the BASE REQUEST. The 300
   prompts are 100 base requests in three framings, so clustering on prompt_id
   still counts each request three times. This is the primary table; the
   per-family numbers in the paper pool a cell with 4 break events against one
   with 55.

2. RESTORES, same treatment. Falcon3's complied stratum is 30 rows, which is 10
   base requests. Reported so the number is not quotable without its interval.

3. CENTRING BASELINES. Subtracting the fitted threshold lifts the pooled AUC,
   but so does subtracting the per-model median, which needs no fp16 labels, no
   judge and no fit. Any standardisation claim has to be reported against these.

4. LEAKAGE. Under greedy decoding the first generated token is a deterministic
   function of the logits, the index is a contrast over four of those tokens,
   and the judge reads the opening of a truncated reply. So "the index predicts
   the fp16 decision" is partly a tautology. Measured here, not argued.

5. WITHIN-TIER ranking AUCs with event counts printed, because framing tier and
   margin are strongly related and most model-by-tier cells have too few events
   to say anything.

6. THE PARAMETER-FREE FALSIFICATION. Using the measured post-quantization margin
   directly -- no Gaussian, no sensitivity estimate, no perturbation size, no
   fitted threshold -- count how many refused prompts actually cross zero
   against how many break. This is the cleanest negative result in the project
   and it currently appears nowhere.

Inputs
  results/analysis/regrid_pairs.csv, phase21_pairs.csv
  results/analysis/delta_m_joined.csv
  results/analysis/threshold.json
  results/margin/per_prompt_margins.csv

Output
  results/analysis/index_stats.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "analysis"))
from threshold_analysis import auc, load_pairs  # noqa: E402

RES = os.path.join(REPO, "results", "analysis")
OUT = os.path.join(REPO, "results", "analysis")
SEED = 20260827
N_BOOT = 2000


def base_id(prompt_id):
    """jb_0000_persuasion -> jb_0000. The framing tiers share a base request."""
    return prompt_id.str.rsplit("_", n=1).str[0]


def clustered_auc(df, score, label, rng, n_boot=N_BOOT):
    """AUC with a bootstrap that resamples whole base requests.

    Resampling is done on precomputed index arrays rather than by concatenating
    DataFrames: 2,000 resamples over ~100 groups for a dozen strata is minutes
    of pandas and under a second of numpy.
    """
    a = auc(df[score], df[label])
    s = df[score].to_numpy(float)
    y = df[label].to_numpy(int)
    idx = list(df.reset_index(drop=True).groupby("base").indices.values())
    boot, undefined = [], 0
    for _ in range(n_boot):
        take = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[i] for i in take])
        v = auc(s[sel], y[sel])
        if v is None or not np.isfinite(v):
            # No event survived this resample. Discarding these silently
            # conditions the interval on "at least one event was drawn", which
            # is precisely the well-behaved half, and reports an interval
            # NARROWER than the prompt-clustered one. Count them instead.
            undefined += 1
        else:
            boot.append(v)
    frac_undef = round(undefined / n_boot, 4)
    if not boot:
        return a, [None, None], len(idx), frac_undef
    return a, [round(float(np.percentile(boot, 2.5)), 4),
               round(float(np.percentile(boot, 97.5)), 4)], len(idx), frac_undef


def main():
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()
    pairs["base"] = base_id(pairs.prompt_id)
    pairs["neg_m0"] = -pairs.m0
    pairs["key"] = pairs.family + "-" + pairs["size"]
    mstar = {k: v["m_star"] for k, v in
             json.load(open(os.path.join(RES, "threshold.json")))["part1_threshold"].items()}
    out = {}

    # --- 1 & 2. ranking and restores, clustered on the base request ------
    for block, stratum, score, label in (
            ("ranking_break", 0, "neg_m0", "flip"),
            ("restore", 1, "m0", "restore")):
        out[block] = {}
        sub = pairs[pairs.s_fp16 == stratum].copy()
        for k, g in sub.groupby("key"):
            a, ci, nclust, undef = clustered_auc(g, score, label, rng)
            ev = int(g[label].sum())
            n_ev_clusters = int(g.loc[g[label] == 1, "base"].nunique())
            out[block][k] = {
                "n_rows": len(g), "n_base_requests": nclust,
                "events": ev,
                # The number that actually governs precision: how many
                # INDEPENDENT requests the events live in, not how many rows.
                "n_base_requests_with_events": n_ev_clusters,
                "auc": a, "ci95_base_clustered": ci,
                "frac_resamples_with_no_events": undef,
                "informative": bool(ev >= 10 and n_ev_clusters >= 5
                                    and undef < 0.01 and ci[0] is not None
                                    and ci[0] > 0.5)}

    # --- 3. centring baselines -------------------------------------------
    ref = pairs[pairs.s_fp16 == 0].copy()
    ref["mstar"] = ref.key.map(mstar)
    gb = ref.groupby("key").m0
    variants = {
        "raw_m0": -ref.m0,
        "minus_fitted_threshold_NEEDS_LABELS": -(ref.m0 - ref.mstar),
        "minus_per_model_median": -(ref.m0 - gb.transform("median")),
        "minus_per_model_mean": -(ref.m0 - gb.transform("mean")),
        "per_model_zscore": -((ref.m0 - gb.transform("mean")) / gb.transform("std")),
        "within_model_percentile": -gb.rank(pct=True),
    }
    out["centring_baselines_pooled"] = {
        name: auc(v, ref.flip) for name, v in variants.items()}
    out["centring_note"] = (
        "Within a single model, subtracting any per-model constant is a monotone "
        "shift and leaves the AUC exactly unchanged. Every gain here is "
        "cross-model normalisation. The fitted threshold is not the best "
        "normaliser and it is the only one that costs a labelled fp16 pass.")

    # --- 4. leakage -------------------------------------------------------
    # Both pair files already carry first_gen_tok, y and m0, so read them raw
    # rather than joining m0 back on and colliding column names. GGUF rows have
    # no first_gen_tok recorded and drop out here, which is itself worth stating:
    # token-level forensics are unavailable for the highest-breaking format.
    lk = pd.concat([pd.read_csv(os.path.join(RES, f))
                    for f in ("regrid_pairs.csv", "phase21_pairs.csv")],
                   ignore_index=True)
    lk = lk[(lk.method != "fp16") & (lk.attack_category == "jailbreak")]
    lk = lk.dropna(subset=["first_gen_tok", "y", "m0"])
    lk["opens_I"] = lk.first_gen_tok.astype(str).str.startswith("I").astype(int)
    scored = {"I", "Sure", "Certainly", "Of"}
    off = lk[~lk.first_gen_tok.astype(str).isin(scored)]
    out["leakage"] = {
        "n": len(lk),
        "comply_rate": round(float(lk.y.mean()), 4),
        "auc_opens_with_I_predicts_label": auc(-lk.opens_I, lk.y),
        "p_comply_given_opens_I": round(float(lk[lk.opens_I == 1].y.mean()), 4),
        "p_comply_given_not": round(float(lk[lk.opens_I == 0].y.mean()), 4),
        "auc_m0_predicts_label_all": auc(-lk.m0, lk.y),
        "auc_m0_predicts_label_off_scored_tokens": (
            auc(-off.m0, off.y) if off.y.nunique() > 1 else None),
        "n_off_scored_tokens": len(off),
        "note": (
            "Greedy decoding makes the first token a deterministic function of "
            "the logits; the index is a contrast over four of those tokens; the "
            "judge reads the opening of a 100-token reply. The index predicting "
            "the fp16 LABEL is therefore substantially a readout of the argmax "
            "it is built from. This does not affect the flip claim, which "
            "predicts a change in that argmax under perturbation, but the "
            "construct must be stated as a four-token readout."),
    }

    # --- 5. within-tier ranking ------------------------------------------
    out["within_tier"] = {}
    for (k, tier), g in ref.groupby(["key", "template_tier"]):
        ev = int(g.flip.sum())
        out["within_tier"][f"{k}|{tier}"] = {
            "n": len(g), "events": ev,
            "auc": auc(-g.m0, g.flip) if ev > 0 else None,
            "reportable": bool(ev >= 10)}

    # --- 6. the parameter-free falsification ------------------------------
    dj = pd.read_csv(os.path.join(RES, "delta_m_joined.csv"))
    dj = dj[dj.attack_category == "jailbreak"]
    rg = pd.read_csv(os.path.join(RES, "regrid_pairs.csv"))
    rg = rg[(rg.method != "fp16") & (rg.attack_category == "jailbreak") & (rg.s_fp16 == 0)]
    j = rg.merge(dj[["family", "size", "prompt_id", "method", "m_first"]],
                 on=["family", "size", "prompt_id", "method"], how="inner")
    assert len(j) <= len(rg), "the delta_m join fanned out"
    out["parameter_free_sign_flip"] = {"cells": [], "note": (
        "Measured post-quantization margin, used directly. No Gaussian, no "
        "sensitivity estimate, no perturbation size, no fitted threshold. If a "
        "break were a sign flip of the first-token margin, these two columns "
        "would match.")}
    for (sz, meth), g in j.groupby(["size", "method"]):
        out["parameter_free_sign_flip"]["cells"].append({
            "cell": f"{sz}-{meth}", "n_refused": len(g),
            "n_margin_sign_flips": int((g.m_first < 0).sum()),
            "n_behavioural_breaks": int(g.broke.sum())})

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "index_stats.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)

    print("=== 1. RANKING (break), intervals clustered on the base request ===")
    print("%-14s %6s %6s %7s %7s %8s %-20s %7s %s" % (
        "model", "rows", "base", "breaks", "in base", "AUC", "95% CI", "no-ev%", "informative"))
    for k, v in out["ranking_break"].items():
        print("%-14s %6d %6d %7d %7d %8.4f %-20s %6.1f%% %s" % (
            k, v["n_rows"], v["n_base_requests"], v["events"],
            v["n_base_requests_with_events"], v["auc"],
            str(v["ci95_base_clustered"]), 100 * v["frac_resamples_with_no_events"],
            v["informative"]))
    print("\n=== 2. RESTORES ===")
    for k, v in out["restore"].items():
        print("%-14s rows=%4d base=%3d restores=%3d in %2d base AUC=%-8s CI=%-20s no-ev %4.1f%% informative=%s" % (
            k, v["n_rows"], v["n_base_requests"], v["events"],
            v["n_base_requests_with_events"], v["auc"],
            str(v["ci95_base_clustered"]),
            100 * v["frac_resamples_with_no_events"], v["informative"]))
    print("\n=== 3. CENTRING BASELINES (pooled) ===")
    for k, v in out["centring_baselines_pooled"].items():
        print("  %-40s %.4f" % (k, v))
    print("\n=== 4. LEAKAGE ===")
    for k, v in out["leakage"].items():
        if k != "note":
            print("  %-42s %s" % (k, v))
    print("\n=== 5. WITHIN-TIER (reportable = 10+ events) ===")
    for k, v in out["within_tier"].items():
        if v["reportable"]:
            print("  %-34s n=%4d events=%3d AUC=%s" % (k, v["n"], v["events"], v["auc"]))
    nrep = sum(1 for v in out["within_tier"].values() if not v["reportable"])
    print("  ... %d of %d model-by-tier cells have under 10 events and are not reportable"
          % (nrep, len(out["within_tier"])))
    print("\n=== 6. PARAMETER-FREE FALSIFICATION ===")
    print("  %-10s %10s %18s %20s" % ("cell", "n refused", "margin sign flips", "behavioural breaks"))
    for c in out["parameter_free_sign_flip"]["cells"]:
        print("  %-10s %10d %18d %20d" % (
            c["cell"], c["n_refused"], c["n_margin_sign_flips"], c["n_behavioural_breaks"]))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
