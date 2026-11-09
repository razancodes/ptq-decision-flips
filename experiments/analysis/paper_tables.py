"""
paper_tables.py -- the paper numbers that no earlier analysis script writes.

1. Table 2: m0 as a flip predictor within the fp16-refused stratum, per family
   (both sizes pooled), with a prompt-clustered bootstrap interval. Each
   bootstrap draw resamples prompt_id with replacement and keeps every row of a
   drawn prompt (both sizes, all three formats). Seed 20260827, 2,000 draws,
   draws with a single class are skipped.

   The interval in the paper came from an earlier run of this procedure whose
   code was not kept. This script reproduces the point estimates exactly and the
   interval endpoints to within 0.004; verify_paper.py checks them at that
   tolerance and says so.

2. Section 3.3 / Appendix B.4: the pooled comparison. On every Falcon3
   fp16-versus-quantized jailbreak pair (1,788 pairs, 69 flips in either
   direction), rank by -m0 and by -delta_m. Same pool on both sides.

3. Appendix B.4: mean delta_m of the flipped prompts on that pool.

Writes results/analysis/paper_tables.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from threshold_analysis import auc  # noqa: E402

RES = os.path.join(REPO, "results", "analysis")
SEED, N_BOOT = 20260827, 2000
KEY = ["prompt_id", "family", "size", "method"]


def clustered_ci(g, score, label, cluster="prompt_id"):
    rng = np.random.default_rng(SEED)
    g = g.reset_index(drop=True)
    groups = {k: v.index.to_numpy() for k, v in g.groupby(cluster)}
    ids = np.array(list(groups))
    vals = []
    for _ in range(N_BOOT):
        d = g.iloc[np.concatenate([groups[k] for k in rng.choice(ids, len(ids), replace=True)])]
        if d[label].nunique() == 2:
            vals.append(auc(-d[score], d[label]))
    return [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]


def main():
    pairs = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in ("regrid_pairs.csv", "phase21_pairs.csv")],
                      ignore_index=True)
    pairs = pairs[(pairs.method != "fp16") & (pairs.attack_category == "jailbreak")].dropna(subset=["m0"])
    out = {"table2_fp16_refused_stratum": {}}

    for fam, g in pairs[pairs.s_fp16 == 0].groupby("family"):
        out["table2_fp16_refused_stratum"][fam] = {
            "n_pairs": len(g), "n_prompts": int(g.prompt_id.nunique()), "breaks": int(g.broke.sum()),
            "auc_m0": round(auc(-g.m0, g.broke), 4), "ci95_prompt_clustered": clustered_ci(g, "m0", "broke")}

    f = pairs[pairs.family == "Falcon3"].merge(
        pd.read_csv(os.path.join(RES, "delta_m_joined.csv"))[KEY + ["delta_m"]], on=KEY, how="inner")
    assert len(f) == (pairs.family == "Falcon3").sum(), "delta_m join dropped Falcon3 pairs"
    out["pooled_falcon3_all_pairs"] = {
        "n_pairs": len(f), "flips": int(f.flip.sum()),
        "auc_m0": round(auc(-f.m0, f.flip), 4), "auc_delta_m": round(auc(-f.delta_m, f.flip), 4),
        "mean_delta_m_of_flips": round(float(f.loc[f.flip == 1, "delta_m"].mean()), 3),
        "mean_delta_m_not_flipped": round(float(f.loc[f.flip == 0, "delta_m"].mean()), 3)}

    # Table 7, "floor effect on the plain tier": flips per framing tier, judge
    # labels on both sides. Sampling = Study A fp16 (sampled at each repo's
    # declared settings) against stated-greedy fp16, Llama-3.2 and Qwen2.5 only
    # (Falcon3's fp16 was greedy in both). Quantization = stated-decoding pairs.
    sa = pd.read_csv(os.path.join(REPO, "results", "results.csv"))
    sa = sa[(sa.method == "fp16") & (sa.attack_category == "jailbreak") & sa.success.isin(["0", "1"])].copy()
    sa["size"] = sa["size"].astype(str)
    pairs["size"] = pairs["size"].astype(str)
    k = ["family", "size", "prompt_id"]
    s = pairs[pairs.family != "Falcon3"].drop_duplicates(k)[k + ["template_tier", "s_fp16"]].merge(sa[k + ["success"]], on=k)
    s["changed"] = (s.s_fp16 != s.success.astype(int)).astype(int)
    out["tier_flips_judge_labels"] = {
        "sampling_only": {f"{fa}|{t}": {"n": len(x), "flips": int(x.changed.sum())}
                          for (fa, t), x in s.groupby(["family", "template_tier"])},
        "quantization": {f"{fa}|{t}": {"n": len(x), "flips": int(x.flip.sum())}
                         for (fa, t), x in pairs.groupby(["family", "template_tier"])}}

    json.dump(out, open(os.path.join(RES, "paper_tables.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
