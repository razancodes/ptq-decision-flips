"""
human_review_analysis.py  --  the 267-pair blind human review, recomputed.

The paper's labeling defence rests on a human reviewer independently re-labeling
every fp16<->quantized pair the judge scored as a flip, blind to the judge's own
labels, scoring each side as refuse/comply. Until now those statistics existed
only as numbers in prose: the instrument and its scored counterpart sat untracked
in the working tree and no script derived the figures from them. This script
closes that gap, so 96.44% / kappa 0.929 / 248 of 267 are reproducible from the
repository instead of taken on trust.

It also reports two properties of the check's DESIGN that change how the numbers
should be read, and that the paper currently omits. Both are consequences of the
sampling rule, not of the reviewer:

  1. The sample is selected on the outcome. Every row is a judge-flagged flip, so
     the judge's 534 side-labels are forced to exactly 267 zeros and 267 ones.
     Chance agreement is therefore ~0.50 here against a higher value in the
     population the labels are drawn from, and Cohen's kappa does not transport
     between the two. Raw agreement does not have this problem; kappa does.

  2. The two sides were shown together. fp16_reply and quantized_reply sit in one
     row, so the 534 side comparisons are paired, not independent, and a Wilson
     interval over n=534 is anticonservative. The pair-level confirmation rate
     (n=267) is the interval that does not have this problem.

Neither point weakens what the check establishes -- that the judge's POSITIVE
flip calls are overwhelmingly real, with zero double-sided errors. What neither
the check nor this script can bound is the judge's false-negative rate, because
non-flip pairs were never sampled. That is the error term that would move the
measured flip rate, and it remains open.

Inputs
  human_review_flip_pairs_SCORED.csv       the scored review instrument
  results/analysis/regrid_pairs.csv   Falcon3 stated-decoding pairs
  results/analysis/phase21_pairs.csv  Llama-3.2 / Qwen2.5 stated-decoding pairs

Output
  results/analysis/human_review.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCORED = os.path.join(REPO, "results", "analysis", "human_review_flip_pairs_SCORED.csv")
OUT = os.path.join(REPO, "results", "analysis")


def wilson(k, n, z=1.96):
    """Wilson score interval on a proportion, returned as percentages."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(100 * max(0.0, centre - half)), 2),
            round(float(100 * min(1.0, centre + half)), 2)]


def cohen_kappa(a, b):
    """Two-rater kappa on binary labels. Returns (kappa, observed, chance)."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    po = float((a == b).mean())
    pe = float((a == 1).mean() * (b == 1).mean()
               + (a == 0).mean() * (b == 0).mean())
    return round((po - pe) / (1 - pe), 4), round(po, 6), round(pe, 6)


def load_population_positive_rate():
    """Judge positive rate over the stated-decoding jailbreak grid.

    This is the base rate the review sample was drawn from, and the reason its
    kappa does not transport.
    """
    frames = []
    for name in ("regrid_pairs.csv", "phase21_pairs.csv"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path, usecols=["attack_category", "y"]))
    if not frames:
        return None, 0
    pop = pd.concat(frames, ignore_index=True)
    pop = pop[pop.attack_category == "jailbreak"].dropna(subset=["y"])
    return float(pop.y.mean()), int(len(pop))


def main():
    d = pd.read_csv(SCORED)

    # Long form: one row per SIDE, because a side is what the reviewer scored.
    sides = pd.concat([
        pd.DataFrame({"family": d.family, "side": "fp16",
                      "human": d.your_label_fp16,
                      "judge": d.judge_label_fp16__DO_NOT_PEEK}),
        pd.DataFrame({"family": d.family, "side": "quant",
                      "human": d.your_label_quant,
                      "judge": d.judge_label_quant__DO_NOT_PEEK}),
    ], ignore_index=True).dropna(subset=["human", "judge"])
    sides["human"] = sides.human.astype(int)
    sides["judge"] = sides.judge.astype(int)

    kappa, po, pe = cohen_kappa(sides.human, sides.judge)
    n_side = len(sides)
    n_agree = int((sides.human == sides.judge).sum())

    per_family = {}
    for fam, g in sides.groupby("family"):
        kf, pof, _ = cohen_kappa(g.human, g.judge)
        per_family[fam] = {"n_sides": len(g),
                           "agreement_pct": round(100 * pof, 2),
                           "kappa": kf}

    # Pair level: a judge-flagged flip is human-confirmed when the reviewer's two
    # labels also differ from each other.
    hf = d.your_label_fp16.astype(int) != d.your_label_quant.astype(int)
    confirmed = int(hf.sum())
    un = d[~hf]
    double_sided = int((
        (un.your_label_fp16.astype(int) != un.judge_label_fp16__DO_NOT_PEEK.astype(int))
        & (un.your_label_quant.astype(int) != un.judge_label_quant__DO_NOT_PEEK.astype(int))
    ).sum())

    pop_pos, pop_n = load_population_positive_rate()
    pe_population = None if pop_pos is None else round(
        pop_pos ** 2 + (1 - pop_pos) ** 2, 6)

    out = {
        "per_side": {
            "n_sides": n_side,
            "n_agree": n_agree,
            "agreement_pct": round(100 * po, 4),
            "agreement_wilson95": wilson(n_agree, n_side),
            "cohen_kappa": kappa,
            "chance_agreement_in_sample": pe,
            "per_family": per_family,
        },
        "pair_level": {
            "n_pairs": int(len(d)),
            "confirmed": confirmed,
            "confirmed_pct": round(100 * confirmed / len(d), 4),
            "confirmed_wilson95": wilson(confirmed, len(d)),
            "unconfirmed_one_sided": int(len(un)) - double_sided,
            "unconfirmed_double_sided": double_sided,
        },
        "design_caveats": {
            "selected_on_outcome": True,
            "judge_positive_rate_in_sample": round(float((sides.judge == 1).mean()), 4),
            "judge_positive_rate_in_population": None if pop_pos is None else round(pop_pos, 4),
            "population_n": pop_n,
            "chance_agreement_in_sample": pe,
            "chance_agreement_at_population_base_rate": pe_population,
            "kappa_transports": False,
            "note_kappa": (
                "Every sampled row is a judge-flagged flip, which forces the "
                "judge's side-label marginal to 50/50 and drives chance agreement "
                "to about 0.50. At the population base rate chance agreement is "
                "higher, so this kappa is an in-stratum figure and must not be "
                "quoted as a dataset-wide one. Raw agreement and the pair-level "
                "confirmation rate are the figures to report."
            ),
            "sides_shown_together": True,
            "note_pairing": (
                "fp16_reply and quantized_reply appear in one row, so the two side "
                "comparisons per pair are not independent and the n=534 Wilson "
                "interval is anticonservative. The n=267 pair-level interval does "
                "not rely on independence between sides."
            ),
            "false_negative_rate_bounded": False,
            "note_false_negatives": (
                "Non-flip pairs were never sampled, so this check bounds the "
                "judge's false-POSITIVE rate on flips and says nothing about false "
                "negatives, which is the error term that would move the measured "
                "flip rate."
            ),
        },
    }

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "human_review.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    ps, pl, dc = out["per_side"], out["pair_level"], out["design_caveats"]
    print("=== per-side agreement (the reviewer's scoring unit) ===")
    print(f"  {ps['n_agree']}/{ps['n_sides']} = {ps['agreement_pct']}% "
          f"Wilson95 {ps['agreement_wilson95']}")
    print(f"  Cohen's kappa {ps['cohen_kappa']}  "
          f"(chance agreement {ps['chance_agreement_in_sample']})")
    for fam, v in ps["per_family"].items():
        print(f"    {fam:11} n={v['n_sides']:4}  {v['agreement_pct']}%  kappa={v['kappa']}")
    print("\n=== pair-level confirmation ===")
    print(f"  {pl['confirmed']}/{pl['n_pairs']} = {pl['confirmed_pct']}% "
          f"Wilson95 {pl['confirmed_wilson95']}")
    print(f"  unconfirmed: {pl['unconfirmed_one_sided']} one-sided, "
          f"{pl['unconfirmed_double_sided']} double-sided")
    print("\n=== design caveats (report these alongside the numbers) ===")
    print(f"  judge positive rate: {dc['judge_positive_rate_in_sample']} in sample "
          f"vs {dc['judge_positive_rate_in_population']} in population (n={dc['population_n']})")
    print(f"  chance agreement:    {dc['chance_agreement_in_sample']} in sample "
          f"vs {dc['chance_agreement_at_population_base_rate']} at population base rate")
    print(f"  kappa transports:    {dc['kappa_transports']}")
    print(f"  false negatives bounded: {dc['false_negative_rate_bounded']}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
