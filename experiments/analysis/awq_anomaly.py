"""
awq_anomaly.py  --  why is AWQ's alpha_equiv ~7x GGUF's, and why do its norm
tensors differ from fp16 at all?

The measurement came back with AWQ as a clear outlier:

    format   median alpha_equiv (targets)   norm tensors
    gguf     0.063                          identical to fp16
    gptq     0.122-0.149                    identical to fp16
    awq      0.454-0.480, max 19.6          median alpha_equiv ~8

Norm weights are not quantized by any of these formats -- GPTQ and GGUF leave
them bit-identical, which is what validation V1 was for. So AWQ's checkpoint
contains norm weights that are *deliberately different* from the fp16 model's,
and that has a standard explanation: AWQ's activation-aware scaling migrates a
per-channel scale s out of a linear layer and into the preceding norm, dividing
the norm by s and multiplying the linear's input columns by s. The composition
is mathematically equivalent -- the model computes the same function -- but every
tensor involved changes.

If that is what is happening, then alpha_equiv for AWQ is NOT a quantization
error measure: it is dominated by an equivalence-preserving reparameterization,
and it cannot be compared against GGUF's or GPTQ's, nor read against Study B's
alpha, which perturbs weights without any compensating change elsewhere.

This script tests that hypothesis against the per-tensor data already collected,
without another GPU session:

  H1  the norm discrepancy is structured, not noise -- if it were quantization
      error the norms would be untouched, and if it were corruption the error
      would not concentrate in the tensors AWQ's scaling touches.
  H2  the affected target tensors are the ones scale migration acts on
      (q/k/v take the attention-norm scale, gate/up take the ffn-norm scale),
      while o_proj and down_proj -- which AWQ scales differently or not at all
      in the same way -- should look more like the other formats.
  H3  the error is multiplicative in character: a migrated tensor's error should
      scale with the tensor's own magnitude, so its Frobenius ratio stays high
      while a purely quantized tensor's does not.

None of this is a fix. It is a diagnosis, and the conclusion it supports is that
AWQ's alpha_equiv must be reported as not comparable rather than averaged in.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")


def main():
    a = pd.read_csv(os.path.join(OUT, "alpha_equiv_tensors.csv"))
    out = {}

    # ------------------------------------------------------------------ H1
    norms = a[a.kind == "norm"]
    out["H1_norm_tensors"] = {
        f"{s}-{m}": {
            "n": int(len(g)),
            "n_exactly_zero": int((g.alpha_equiv == 0).sum()),
            "median_alpha": round(float(g.alpha_equiv.median()), 5),
            "max_alpha": round(float(g.alpha_equiv.max()), 5),
            "median_frobenius": round(float(g.frobenius_ratio.median()), 5),
        }
        for (s, m), g in norms.groupby(["size", "method"])
    }

    # ------------------------------------------------------------------ H2
    tgt = a[a.group == "target"]
    out["H2_by_projection"] = {}
    for (s, m), g in tgt.groupby(["size", "method"]):
        out["H2_by_projection"][f"{s}-{m}"] = {
            k: round(float(gg.alpha_equiv.median()), 5)
            for k, gg in g.groupby("kind")}

    # ------------------------------------------------------------------ H3
    out["H3_frobenius_vs_alpha"] = {}
    for (s, m), g in tgt.groupby(["size", "method"]):
        out["H3_frobenius_vs_alpha"][f"{s}-{m}"] = {
            "median_alpha_equiv": round(float(g.alpha_equiv.median()), 5),
            "median_frobenius_ratio": round(float(g.frobenius_ratio.median()), 5),
            "median_mean_error_over_rms": round(float(
                (g.mean_error.abs() / g.rms_error.replace(0, np.nan)).median()), 5),
        }

    # which tensors are the extreme outliers?
    ext = tgt[tgt.alpha_equiv > 0.6].copy()
    out["outliers_above_0.6"] = {
        f"{s}-{m}": {"n": int(len(g)),
                     "kinds": g.kind.value_counts().to_dict(),
                     "layers": sorted(set(int(x) for x in g.layer))[:12],
                     "max": round(float(g.alpha_equiv.max()), 4)}
        for (s, m), g in ext.groupby(["size", "method"])}

    # what a clean 4-bit quantizer should look like, for reference: the
    # quantization step of a k-bit uniform quantizer over a Gaussian with unit
    # std is about 2*3.5/2^k, so RMS error ~ step/sqrt(12).
    for bits in (4, 5, 6, 8):
        step = 7.0 / (2 ** bits)
        out.setdefault("reference_uniform_quantizer_alpha", {})[f"{bits}bit"] = \
            round(step / np.sqrt(12), 5)

    with open(os.path.join(OUT, "awq_anomaly.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("=== H1: norm tensors, which no format quantizes ===")
    for k, v in out["H1_norm_tensors"].items():
        print(f"  {k}: n={v['n']} exactly-zero={v['n_exactly_zero']} "
              f"median_alpha={v['median_alpha']} max={v['max_alpha']} "
              f"median_frobenius={v['median_frobenius']}")
    print("\n=== H2: alpha_equiv by projection type ===")
    for k, v in out["H2_by_projection"].items():
        print(f"  {k}: {v}")
    print("\n=== H3: alpha_equiv vs Frobenius ratio ===")
    for k, v in out["H3_frobenius_vs_alpha"].items():
        print(f"  {k}: {v}")
    print("\n=== target tensors with alpha_equiv > 0.6 ===")
    for k, v in out["outliers_above_0.6"].items():
        print(f"  {k}: {v}")
    print("\n=== what a clean uniform quantizer would give ===")
    print(f"  {out['reference_uniform_quantizer_alpha']}")
    print(f"\nwrote {OUT}/awq_anomaly.json")


if __name__ == "__main__":
    main()
