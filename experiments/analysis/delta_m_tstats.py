"""
delta_m_tstats.py -- per-cell mean and SD of the measured margin shift delta_m
on the 300 jailbreak prompts, with a one-sample t against zero.

Read the t-values with care: a quantized checkpoint is ONE draw of the
perturbation, and the 300 prompts all see that same draw, so they are not 300
independent samples of the shift. The paper therefore does not claim a
systematic bias; review_audit.py (section F) compares each cell's mean/SD with
single synthetic isotropic draws instead. Writes s5_jailbreak_tstats.json,
which threshold_analysis.py reads.
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


def main():
    dm = pd.read_csv(os.path.join(OUT, "delta_m_joined.csv"))
    jb = dm[dm.attack_category == "jailbreak"]
    rows, tvals, sig, nonsig = {}, [], [], []
    for (size, method), g in jb.groupby(["size", "method"]):
        d = g.delta_m.dropna()
        t, p = stats.ttest_1samp(d, 0.0)
        key = f"{size}-{method}"
        rows[key] = {"n": int(len(d)), "mean": round(float(d.mean()), 4),
                     "sd": round(float(d.std(ddof=1)), 4),
                     "t": round(float(t), 2), "p": float(p)}
        tvals.append(abs(float(t)))
        (sig if p < 0.05 else nonsig).append(key)
    with open(os.path.join(OUT, "s5_jailbreak_tstats.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"cells": rows, "significant": sig,
                   "not_significant": nonsig}, fh, indent=2)
    for k, v in rows.items():
        print(f"  {k}: n={v['n']} mean={v['mean']:+.4f} sd={v['sd']:.4f} "
              f"t={v['t']:+.2f} p={v['p']:.3g}")
    print(f"  significant: {sig}")
    print(f"  NOT significant: {nonsig}")


if __name__ == "__main__":
    main()
