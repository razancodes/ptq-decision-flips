"""
decoding_confound.py  --  re-analyse §4.7 stratified by what each cell's decoder
actually did.

decoding_audit.py established that Falcon3's six quantized cells were not all
greedy:

  fp16  (reference)   greedy      -- no do_sample in generation_config
  gptq  1B / 3B       greedy      -- no generation_config.json in the repo at all
                                     (404), so do_sample falls back to False
  awq   1B / 3B       SAMPLED     -- the repo declares "do_sample": true with no
                                     temperature, so transformers' defaults apply
                                     (T=1.0, top_k=50, top_p=1.0)
  gguf  1B / 3B       SAMPLED     -- generate_reply_gguf passes only max_tokens,
                                     and llama-cpp-python's create_chat_completion
                                     defaults to T=0.2, top_p=0.95, top_k=40

So §4.7's premise -- "because Falcon3 decodes greedily, every fp16->quantized
flip is 100% attributable to quantization" -- holds only for the GPTQ cells. In
the AWQ and GGUF cells the quantized side was sampled while the fp16 side was
greedy, which is precisely the confound §4.6 identified for Qwen2.5 and
Llama-3.2.

This script therefore computes:

  1. the confound-free core: greedy fp16 vs greedy GPTQ;
  2. the same quantities for the sampled cells, reported separately;
  3. whether the per-format ordering the paper reads as quantization severity is
     instead the sampling-temperature ordering.

It does not attempt to rescue anything. The numbers are what they are.
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

# From decoding_audit.py. Temperatures are the EFFECTIVE ones after library
# defaults are applied.
DECODING = {
    ("1B", "gptq"): ("greedy", 0.0), ("3B", "gptq"): ("greedy", 0.0),
    ("1B", "gguf"): ("sampled", 0.2), ("3B", "gguf"): ("sampled", 0.2),
    ("1B", "awq"): ("sampled", 1.0), ("3B", "awq"): ("sampled", 1.0),
}


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
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def wilson(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 4), round(100 * (c + h), 4)]


def block(sub, label):
    if not len(sub):
        return {"label": label, "n_pairs": 0}
    lm = sub[sub.template_tier.isin(["roleplay", "persuasion"])]
    ref = sub[sub.s_fp16 == 0]
    a = auc_low(sub.m0, sub.flip)
    return {
        "label": label,
        "n_pairs": int(len(sub)), "n_flips": int(sub.flip.sum()),
        "flip_rate_pct": round(100 * float(sub.flip.mean()), 4),
        "wilson95": wilson(int(sub.flip.sum()), int(len(sub))),
        "broke": int(sub.broke.sum()), "restore": int(sub.restore.sum()),
        "pct_degrading": (round(100 * sub.broke.sum() / sub.flip.sum(), 2)
                          if sub.flip.sum() else None),
        "m0_auc": round(a, 4) if np.isfinite(a) else None,
        "m0_ci": boot_ci(sub.m0, sub.flip),
        "m0_auc_length_matched": (round(auc_low(lm.m0, lm.flip), 4)
                                  if len(lm) and lm.flip.sum() else None),
        "m0_ci_length_matched": boot_ci(lm.m0, lm.flip) if len(lm) else None,
        "fp16_refused_auc": (round(auc_low(ref.m0, ref.flip), 4)
                             if len(ref) and ref.flip.sum() else None),
        "fp16_refused_n": int(len(ref)),
        "tier_flip_rate_pct": {str(t): round(100 * float(g.flip.mean()), 4)
                               for t, g in sub.groupby("template_tier")},
        "tier_n": {str(t): int(len(g)) for t, g in sub.groupby("template_tier")},
        "by_cell_flip_rate_pct": {f"{s}-{m}": round(100 * float(g.flip.mean()), 4)
                                  for (s, m), g in sub.groupby(["size", "method"])},
    }


def main():
    results = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                          dtype={"success": str})
    results["success"] = results["success"].fillna("").str.strip()
    per = pd.read_csv(os.path.join(REPO, "results", "margin",
                                   "per_prompt_margins.csv"))
    d = results[results.success.isin(["0", "1"])].copy()
    d["y"] = d.success.astype(int)

    idx = ["family", "size", "prompt_id"]
    fp16 = d[d.method == "fp16"].set_index(idx)["y"].rename("s_fp16")
    quant = d[d.method != "fp16"].set_index(idx)
    j = quant.join(fp16, how="inner").reset_index()
    j["flip"] = (j.y != j.s_fp16).astype(int)
    j["broke"] = ((j.s_fp16 == 0) & (j.y == 1)).astype(int)
    j["restore"] = ((j.s_fp16 == 1) & (j.y == 0)).astype(int)
    j = j.merge(per[["family", "size", "prompt_id", "m0", "g", "kappa"]],
                on=["family", "size", "prompt_id"], how="left")

    fal = j[(j.family == "Falcon3") & (j.attack_category == "jailbreak")].copy()
    fal["decoding"] = [DECODING.get((s, m), ("unknown", None))[0]
                       for s, m in zip(fal["size"], fal.method)]
    fal["temperature"] = [DECODING.get((s, m), ("unknown", None))[1]
                          for s, m in zip(fal["size"], fal.method)]

    out = {"decoding_map": {f"{s}-{m}": {"regime": v[0], "temperature": v[1]}
                            for (s, m), v in DECODING.items()}}
    out["all_cells"] = block(fal, "all six cells (the current §4.7 scope)")
    out["greedy_only"] = block(fal[fal.decoding == "greedy"],
                               "confound-free: greedy fp16 vs greedy GPTQ")
    out["sampled_only"] = block(fal[fal.decoding == "sampled"],
                                "sampled quantized side (AWQ T=1.0, GGUF T=0.2)")
    for method in ("gptq", "awq", "gguf"):
        out[f"method_{method}"] = block(fal[fal.method == method], method)

    # 3. is the format ordering the temperature ordering?
    cell = (fal.groupby(["size", "method"])
            .agg(n=("flip", "size"), flips=("flip", "sum"),
                 rate=("flip", "mean"), temp=("temperature", "first"))
            .reset_index())
    cell["rate_pct"] = (100 * cell.rate).round(4)
    rho, p_rho = stats.spearmanr(cell.temp, cell.rate)
    out["format_ordering_vs_temperature"] = {
        "cells": cell[["size", "method", "n", "flips", "rate_pct",
                       "temp"]].to_dict("records"),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "note": ("within each size the flip-rate ordering awq > gguf > gptq is "
                 "the same as the effective-temperature ordering 1.0 > 0.2 > 0, "
                 "so the per-format ordering cannot be read as quantization "
                 "severity from this data"),
    }
    # within-size ordering check, which is the fair one (size dominates)
    per_size = {}
    for size, g in cell.groupby("size"):
        g = g.sort_values("temp")
        per_size[size] = {
            "by_temperature": g[["method", "temp", "rate_pct"]].to_dict("records"),
            "monotone_in_temperature": bool(
                all(np.diff(g.rate_pct.values) > 0)),
        }
    out["within_size_ordering"] = per_size

    with open(os.path.join(OUT, "decoding_confound.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    def show(b):
        if not b.get("n_pairs"):
            print(f"  {b['label']}: no pairs")
            return
        print(f"  {b['label']}")
        print(f"    n={b['n_pairs']:,} flips={b['n_flips']} "
              f"({b['flip_rate_pct']:.2f}%, Wilson {b['wilson95']})  "
              f"break/restore {b['broke']}/{b['restore']}")
        print(f"    m0 AUC {b['m0_auc']} {b['m0_ci']}  "
              f"length-matched {b['m0_auc_length_matched']} "
              f"{b['m0_ci_length_matched']}  "
              f"fp16-refused {b['fp16_refused_auc']} (n={b['fp16_refused_n']})")
        print(f"    tiers {b['tier_flip_rate_pct']}")
        print(f"    cells {b['by_cell_flip_rate_pct']}")

    print("=== §4.7 stratified by what the decoder actually did ===")
    for k in ("all_cells", "greedy_only", "sampled_only", "method_gptq",
              "method_awq", "method_gguf"):
        show(out[k])
        print()
    print("=== is the format ordering the temperature ordering? ===")
    print(pd.DataFrame(out["format_ordering_vs_temperature"]["cells"])
          .to_string(index=False))
    print(f"  Spearman rho(temperature, flip rate) over 6 cells = "
          f"{out['format_ordering_vs_temperature']['spearman_rho']} "
          f"(p={out['format_ordering_vs_temperature']['spearman_p']:.4f})")
    for size, v in per_size.items():
        print(f"  {size}: monotone increasing in temperature = "
              f"{v['monotone_in_temperature']}  {v['by_temperature']}")
    print(f"\nwrote {OUT}/decoding_confound.json")


if __name__ == "__main__":
    main()
