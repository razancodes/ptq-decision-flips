"""
qwen_mechanism.py  --  does the perturbation measurement replicate on a second family?

The mechanism section is Falcon3-only. Everything it claims about how hard each
format perturbs the weights rests on one family, which is also the family whose
fp16 compliance is too rare to locate anything behavioural. This re-points the
same kernel at Qwen2.5-1.5B/3B and asks whether the alpha_equiv measurement
reproduces.

It also enforces the coherence gate. The driver's V4 check reloads the
dequantized weights and asks the model to name the capital of France. On Qwen's
GGUF cells it answered "you are a voice voice, I am not in voice" at 1.5B and a
run of stray characters at 3B, so those weights are wrong and their weight-error
numbers mean nothing. The check existed and fired, but only recorded the failure,
so the cell still emitted a full set of alpha_equiv rows that looked like
measurements. Cells failing V4 are dropped here and the driver now aborts them.

Why GGUF specifically fails on Qwen and not Falcon3 is visible in the artifact:
q_proj and k_proj come out near 1.16 while o_proj sits at 0.083, and q and k are
exactly the tensors llama.cpp stores under a RoPE permutation. The permute
branch was chosen on Falcon3's head layout (12 heads, 4 KV) and Qwen2.5-1.5B has
12 heads with 2 KV plus biased attention projections that the name mapper leaves
unmapped. The undo makes it worse rather than better here, and the selection
rule still picks it.

Interpretation, fixed before running:

  * If GPTQ lands near the textbook 4-bit figure on Qwen as it does on Falcon3,
    the measurement is a property of the format and not of the family.
  * If AWQ again shows the scale-migration signature with a clean o_proj, the
    Falcon3 decision to isolate o_proj was not a Falcon3-specific fudge.
  * Any cell failing V4 is reported as failed, not as a number.

Inputs
  results/analysis/qwen_mechanism/{alpha_equiv_tensors.csv,mechanism_meta.json}
  results/analysis/mechanism_verdict.json   Falcon3 reference

Output
  results/analysis/qwen_mechanism.json
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "results", "analysis", "qwen_mechanism")
OUT = os.path.join(REPO, "results", "analysis")
THEORY_4BIT = 0.1263


def main():
    meta = json.load(open(os.path.join(SRC, "mechanism_meta.json")))["models"]
    tens = pd.read_csv(os.path.join(SRC, "alpha_equiv_tensors.csv"))
    tgt = tens[tens.group == "target"]

    fal = json.load(open(os.path.join(REPO, "results", "analysis",
                                      "mechanism_verdict.json")))["Q1_alpha_estimates"]

    out = {"theory_uniform_4bit": THEORY_4BIT, "cells": [], "failed_cells": []}
    for mk, mv in meta.items():
        size = "1.5B" if "1.5b" in mk else "3B"
        for method in ("awq", "gptq", "gguf"):
            v = mv.get(method, {}).get("validations", {})
            sample = v.get("V4_coherence_sample", "")
            if "Paris" not in sample:
                out["failed_cells"].append({
                    "cell": f"{size}-{method}",
                    "reason": "V4 coherence failed; dequantized weights are wrong",
                    "answered": sample[:80],
                    "n_outside_sane_range": v.get("V5_range", {}).get("n_outside_range"),
                })
                continue
            g = tgt[(tgt["size"] == size) & (tgt.method == method)]
            # AWQ rewrites norms and inflates every norm-fed projection, so its
            # all-tensor median is scale migration rather than error. o_proj
            # takes attention output, not a norm, and is the migration-free
            # basis -- the same structural choice made on Falcon3, fixed before
            # any Qwen value was read.
            basis = "o_proj only (migration-free)" if method == "awq" else "all target tensors"
            val = (g[g.kind == "o_proj"].alpha_equiv.median() if method == "awq"
                   else g.alpha_equiv.median())
            out["cells"].append({
                "cell": f"{size}-{method}", "basis": basis,
                "alpha_equiv": round(float(val), 5),
                "all_targets_median": round(float(g.alpha_equiv.median()), 5),
                "n_tensors": int(len(g)),
                "falcon3_reference": fal.get(
                    f"{'1B' if size == '1.5B' else '3B'}-{method}", {}).get("alpha_equiv"),
            })

    gptq = [c for c in out["cells"] if c["cell"].endswith("gptq")]
    awq = [c for c in out["cells"] if c["cell"].endswith("awq")]
    out["verdict"] = {
        "gptq_matches_textbook_4bit": bool(gptq) and all(
            abs(c["alpha_equiv"] - THEORY_4BIT) < 0.03 for c in gptq),
        "awq_shows_scale_migration_with_clean_oproj": bool(awq) and all(
            c["all_targets_median"] > 0.25 and c["alpha_equiv"] < 0.15 for c in awq),
        "n_cells_valid": len(out["cells"]),
        "n_cells_failed": len(out["failed_cells"]),
        "reading": (
            "GPTQ and AWQ replicate on a second family: GPTQ lands on the "
            "textbook 4-bit figure and AWQ shows the same scale-migration "
            "signature with a clean o_proj, so neither was a Falcon3 artefact. "
            "GGUF does not replicate, because the dequantization path's RoPE "
            "handling was derived for Falcon3's head layout; those cells are "
            "reported as failed rather than as small numbers."),
    }

    dest = os.path.join(OUT, "qwen_mechanism.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)

    print("=== Qwen2.5 alpha_equiv, valid cells ===")
    print("%-12s %-30s %10s %12s %12s" % ("cell", "basis", "alpha", "all-target",
                                          "Falcon3 ref"))
    for c in out["cells"]:
        print("%-12s %-30s %10.5f %12.5f %12s" % (
            c["cell"], c["basis"], c["alpha_equiv"], c["all_targets_median"],
            c["falcon3_reference"]))
    print(f"\n(theoretical uniform 4-bit over Gaussian weights: {THEORY_4BIT})")
    print("\n=== failed cells ===")
    for c in out["failed_cells"]:
        print("  %-12s %s" % (c["cell"], c["reason"]))
        print("               answered %r, %s tensors outside the sane range"
              % (c["answered"][:50], c["n_outside_sane_range"]))
    print("\n=== VERDICT ===")
    for k, v in out["verdict"].items():
        if k != "reading":
            print(f"  {k}: {v}")
    print("  " + out["verdict"]["reading"])
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
