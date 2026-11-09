"""
null_is_it_real.py  --  is the 0.00% numerical null measuring a perturbation, or
is it a restatement of judge determinism?

The challenge, stated precisely: the null compares each original Study A fp16 row
against its three batched left-padded re-seed counterparts, and reports 0 label
flips in 1,791 comparisons. That is only meaningful if the two sides differ as
TEXT. If the re-seed reproduced the original string, the judge was handed the
same input twice and 0.00% just restates the 190/190 judge-reproducibility
figure -- the null would be circular and would have to be withdrawn.

Section 6 reports 91.9% exact reproduction for the Falcon3 re-seed, i.e. 8.1% of
strings differ. But that figure is over some scope of its own, and the null runs
on a specific subset: Falcon3, jailbreak only, scorable on both sides. The
divergence rate has to be computed on exactly those 1,791 comparisons, not
inherited from a headline number computed over a different set.

So: for each of the 1,791 comparisons, is the re-seed reply byte-identical to the
original reply? Reported overall, per model, per tier, per seed, and -- most
importantly -- split by whether the pair's labels agreed, since the null's whole
content is "text changed, verdict did not".
"""

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "analysis")


def norm(s):
    return "" if pd.isna(s) else str(s).strip()


def main():
    rows = pd.read_csv(os.path.join(OUT, "numerical_null_rows.csv"))
    reseed = pd.read_csv(os.path.join(REPO, "results", "margin",
                                      "reseed_runs_all.csv"), dtype=str)
    orig = pd.read_csv(os.path.join(REPO, "results", "results.csv"),
                       dtype={"success": str})

    print(f"null comparisons on file: {len(rows):,}")

    # original fp16 reply per (family, size, prompt)
    o = orig[(orig.family == "Falcon3") & (orig.method == "fp16")][
        ["family", "size", "prompt_id", "model_reply"]].rename(
        columns={"model_reply": "reply_orig"})
    # re-seed reply per (family, size, seed, prompt)
    r = reseed[reseed.family == "Falcon3"][
        ["family", "size", "seed", "prompt_id", "model_reply"]].rename(
        columns={"model_reply": "reply_reseed"})
    r["seed"] = r["seed"].astype(int)

    m = rows.merge(o, on=["family", "size", "prompt_id"], how="left")
    m = m.merge(r, on=["family", "size", "seed", "prompt_id"], how="left")
    missing = int(m.reply_orig.isna().sum() + m.reply_reseed.isna().sum())
    print(f"unmatched replies after join: {missing}")

    m["same_text"] = [norm(a) == norm(b)
                      for a, b in zip(m.reply_orig, m.reply_reseed)]
    m["differs"] = ~m.same_text

    n = len(m)
    d = int(m.differs.sum())
    print("\n=== THE NUMBER THAT WAS MISSING ===")
    print(f"  of the {n:,} comparisons in the null, the re-seed reply differs")
    print(f"  from the original in **{d:,} ({100*d/n:.2f}%)**")
    print(f"  and is byte-identical in {n-d:,} ({100*(n-d)/n:.2f}%)")

    out = {"n_comparisons": n, "n_text_differs": d,
           "pct_text_differs": round(100 * d / n, 4),
           "pct_text_identical": round(100 * (n - d) / n, 4),
           "n_label_flips": int(m.flip.sum())}

    print("\n=== by model / tier / seed ===")
    for by in ("size", "tier", "seed"):
        if by not in m:
            continue
        t = m.groupby(by).differs.agg(["size", "sum", "mean"])
        t.columns = ["n", "n_differ", "frac_differ"]
        print(f"  by {by}:")
        print(t.round(4).to_string())
        out[f"by_{by}"] = {str(k): {"n": int(v["n"]),
                                    "n_differ": int(v["n_differ"]),
                                    "pct_differ": round(100 * v["frac_differ"], 3)}
                           for k, v in t.iterrows()}

    print("\n=== the claim the null actually makes ===")
    diff = m[m.differs]
    print(f"  on the {len(diff):,} comparisons where the TEXT CHANGED,")
    print(f"  the safety label changed on {int(diff.flip.sum())} of them "
          f"({100*diff.flip.mean() if len(diff) else 0:.2f}%)")
    same = m[m.same_text]
    print(f"  on the {len(same):,} where the text was identical, "
          f"{int(same.flip.sum())} label changes (this part IS just judge "
          f"determinism)")
    out["flips_where_text_changed"] = int(diff.flip.sum())
    out["n_where_text_changed"] = int(len(diff))
    out["flips_where_text_identical"] = int(same.flip.sum())
    out["n_where_text_identical"] = int(len(same))

    verdict = ("REAL: the null rests on a majority-changed-text sample"
               if 100 * d / n >= 5 else
               "THIN: too few comparisons changed text for the null to say much"
               if d > 0 else
               "CIRCULAR: no text changed, so the null restates judge determinism")
    out["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    if d:
        print(f"  The load-bearing statement is: on {len(diff):,} comparisons "
              f"where Falcon3\n  produced DIFFERENT text under a different "
              f"numerical regime, the safety label\n  changed "
              f"{int(diff.flip.sum())} times.")

    with open(os.path.join(OUT, "null_is_it_real.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    m[["family", "size", "seed", "prompt_id", "tier", "y0", "y", "flip",
       "same_text"]].to_csv(os.path.join(OUT, "null_text_divergence.csv"),
                            index=False)
    print(f"\nwrote {OUT}/null_is_it_real.json and null_text_divergence.csv")


if __name__ == "__main__":
    main()
