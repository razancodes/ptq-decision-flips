"""
verify_paper.py -- check every number the paper quotes against the artifact that
produces it. Run after reproduce.py (or straight away: results/ is committed).

    python verify_paper.py

Each row is (where in the paper, the value as printed, the artifact field it
comes from). A value passes when the artifact rounds to it at the precision the
paper prints, or sits within the stated tolerance where one is given (only for
the Table 2 bootstrap intervals; see paper_tables.py for why).
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "analysis")
_cache = {}


def J(name, *path):
    if name not in _cache:
        _cache[name] = json.load(open(os.path.join(R, name), encoding="utf-8"))
    o = _cache[name]
    for p in path:
        o = o[p]
    return o


def sweep_rows():
    return len(pd.read_csv(os.path.join(HERE, "results", "margin", "margin_sweep_all.csv"), usecols=["alpha"]))


V = lambda c, k: J("mechanism_verdict.json", "Q2_variance_model", c, k)           # noqa: E731
T2 = lambda f, *k: J("paper_tables.json", "table2_fp16_refused_stratum", f, *k)    # noqa: E731
POOL = lambda k: J("paper_tables.json", "pooled_falcon3_all_pairs", k)             # noqa: E731
DR = J  # alias for readability below
TS = lambda c: J("s5_jailbreak_tstats.json", "cells", c)                           # noqa: E731
A_F = lambda k: J("review_audit.json", "A_first_token_event", "Falcon3", k)        # noqa: E731
RANK = lambda m: J("index_stats.json", "ranking_break", m)                         # noqa: E731
sig = lambda: [c for c in J("s5_jailbreak_tstats.json", "significant")]            # noqa: E731
TIER = lambda kind, k: (lambda d: 100 * d["flips"] / d["n"])(J("paper_tables.json", "tier_flips_judge_labels", kind, k))  # noqa: E731

# (where, value as printed, computed value, tolerance or None)
CLAIMS = [
    # ---- abstract, introduction
    ("abstract: under-predicts, low end", "3.5", lambda: DR("dose_response_recheck.json", "range_vs_repaired")[0], None),
    ("abstract: under-predicts, high end", "337", lambda: DR("dose_response_recheck.json", "range_vs_repaired")[1], None),
    ("abstract: m0 AUC", "0.95", lambda: POOL("auc_m0"), None),
    ("abstract: delta_m AUC", "0.71", lambda: POOL("auc_delta_m"), None),
    # ---- section 2
    ("S2: synthetic sweep measurements", "62400", sweep_rows, None),
    ("S2: median R^2 of the linear fit", "0.997", lambda: J("studyb_baseline.json", "g_summary", "median_lin_r2"), None),
    # ---- section 3.1, Table 1
    ("S3.1: median obs/pred ratio", "0.62", lambda: J("mechanism_verdict.json", "Q2_verdict", "median_ratio"), None),
    ("S3.1: cells within 2x", "6", lambda: sum(0.5 <= V(c, "ratio_obs_over_pred") <= 2 for c in J("mechanism_verdict.json", "Q2_variance_model")), None),
    ("Table 1: 1B GPTQ predicted", "1.877", lambda: V("1B-gptq", "predicted_sd"), None),
    ("Table 1: 1B GPTQ observed", "0.991", lambda: V("1B-gptq", "observed_sd"), None),
    ("Table 1: 1B GPTQ ratio", "0.53", lambda: V("1B-gptq", "ratio_obs_over_pred"), None),
    ("Table 1: 3B GPTQ predicted", "2.782", lambda: V("3B-gptq", "predicted_sd"), None),
    ("Table 1: 3B GPTQ observed", "1.557", lambda: V("3B-gptq", "observed_sd"), None),
    ("Table 1: 3B GPTQ ratio", "0.56", lambda: V("3B-gptq", "ratio_obs_over_pred"), None),
    ("Table 1: 3B GGUF predicted", "1.301", lambda: V("3B-gguf", "predicted_sd"), None),
    ("Table 1: 3B GGUF observed", "1.309", lambda: V("3B-gguf", "observed_sd"), None),
    ("Table 1: 3B GGUF ratio", "1.01", lambda: V("3B-gguf", "ratio_obs_over_pred"), None),
    # ---- section 3.2 and Appendix B.4: the boundary
    ("S3.2: Falcon3 breaks", "59", lambda: A_F("breaks"), None),
    ("S3.2: breaks whose first token changes", "58", lambda: A_F("breaks_opener_left_I"), None),
    ("S3.2: breaks with measured margin still positive", "57",
     lambda: A_F("breaks") - sum(J("review_audit.json", "B_post_quant_margin", m, "breaks_with_mq_below_0") for m in ("Falcon3-1B", "Falcon3-3B")), None),
    ("S3.2: miss by format, GPTQ", "7.2", lambda: DR("dose_response_recheck.json", "by_method")[0]["factor"], None),
    ("S3.2: miss by format, AWQ", "36", lambda: DR("dose_response_recheck.json", "by_method")[1]["factor"], None),
    ("S3.2: miss by format, GGUF", "302", lambda: DR("dose_response_recheck.json", "by_method")[2]["factor"], None),
    ("S3.2/B.4: alpha needed, low end", "0.16", lambda: min(c["alpha_required"] for c in DR("dose_response_recheck.json", "alpha_shortfall")), None),
    ("S3.2/B.4: alpha needed, high end", "0.37", lambda: max(c["alpha_required"] for c in DR("dose_response_recheck.json", "alpha_shortfall")), None),
    ("S3.2/B.4: times the largest dose (0.04), high end", "9", lambda: max(c["alpha_required"] for c in DR("dose_response_recheck.json", "alpha_shortfall")) / 0.04, None),
    ("S3.2/B.4: cells with |mean/SD| >= 0.4", "5", lambda: sum(abs(V(c, "mean_over_sd")) >= 0.4 for c in J("mechanism_verdict.json", "Q2_variance_model")), None),
    ("S3.2: cells with negative mean shift", "4", lambda: sum(TS(c)["mean"] < 0 for c in J("s5_jailbreak_tstats.json", "cells")), None),
    ("S3.2/B.4: most affected mean/SD", "-2.11", lambda: V("3B-awq", "mean_over_sd"), None),
    # ---- section 3.3, Table 2
    ("S3.3: pooled delta_m AUC", "0.715", lambda: POOL("auc_delta_m"), None),
    ("S3.3: pooled m0 AUC", "0.948", lambda: POOL("auc_m0"), None),
    ("Table 2: Falcon3 n", "1758", lambda: T2("Falcon3", "n_pairs"), None),
    ("Table 2: Falcon3 AUC", "0.952", lambda: T2("Falcon3", "auc_m0"), None),
    ("Table 2: Falcon3 CI low", "0.926", lambda: T2("Falcon3", "ci95_prompt_clustered")[0], 0.004),
    ("Table 2: Falcon3 CI high", "0.971", lambda: T2("Falcon3", "ci95_prompt_clustered")[1], 0.004),
    ("Table 2: Llama-3.2 n", "1701", lambda: T2("Llama-3.2", "n_pairs"), None),
    ("Table 2: Llama-3.2 AUC", "0.949", lambda: T2("Llama-3.2", "auc_m0"), None),
    ("Table 2: Llama-3.2 CI low", "0.909", lambda: T2("Llama-3.2", "ci95_prompt_clustered")[0], 0.004),
    ("Table 2: Llama-3.2 CI high", "0.976", lambda: T2("Llama-3.2", "ci95_prompt_clustered")[1], 0.004),
    ("Table 2: Qwen2.5 n", "1290", lambda: T2("Qwen2.5", "n_pairs"), None),
    ("Table 2: Qwen2.5 AUC", "0.902", lambda: T2("Qwen2.5", "auc_m0"), None),
    ("Table 2: Qwen2.5 CI low", "0.855", lambda: T2("Qwen2.5", "ci95_prompt_clustered")[0], 0.004),
    ("Table 2: Qwen2.5 CI high", "0.942", lambda: T2("Qwen2.5", "ci95_prompt_clustered")[1], 0.004),
    ("S3.3: corr(m0, m0+dm), Falcon3 low", "0.90", lambda: J("review_audit.json", "B_post_quant_margin", "Falcon3-3B", "corr_m0_mq"), None),
    ("S3.3: corr(m0, m0+dm), Falcon3 high", "0.95", lambda: J("review_audit.json", "B_post_quant_margin", "Falcon3-1B", "corr_m0_mq"), None),
    # ---- Appendix A (limitations)
    ("App A: m0 ranks sampling-only flips, low", "0.87", lambda: J("review_audit.json", "E_sampling_specificity", "Qwen2.5: fp16 sampled vs fp16 greedy (pure sampling)", "auc_m0_sampling"), None),
    ("App A: m0 ranks sampling-only flips, high", "0.95", lambda: J("review_audit.json", "E_sampling_specificity", "Falcon3-gguf: sampler-only breaks (Study A flip, not in repaired)", "auc_m0"), None),
    # ---- Appendix B.2 / B.3: judge and human calibration
    ("B.3: judge rows re-sent", "192", lambda: J("numerical_null.json", "judge_reproducibility", "n_sent"), None),
    ("B.3: judge rows identical", "190", lambda: J("numerical_null.json", "judge_reproducibility", "n_agree"), None),
    ("B.3: flip pairs reviewed", "267", lambda: J("human_review.json", "pair_level", "n_pairs"), None),
    ("B.3: side comparisons", "534", lambda: J("human_review.json", "per_side", "n_sides"), None),
    ("B.3: per-side agreement %", "96.44", lambda: J("human_review.json", "per_side", "agreement_pct"), None),
    ("B.3: agreement CI low", "94.5", lambda: J("human_review.json", "per_side", "agreement_wilson95")[0], None),
    ("B.3: agreement CI high", "97.7", lambda: J("human_review.json", "per_side", "agreement_wilson95")[1], None),
    ("B.3: Cohen's kappa", "0.929", lambda: J("human_review.json", "per_side", "cohen_kappa"), None),
    ("B.3: confirmed flips %", "92.88", lambda: J("human_review.json", "pair_level", "confirmed_pct"), None),
    ("B.3: confirmed flips", "248", lambda: J("human_review.json", "pair_level", "confirmed"), None),
    ("B.3: one-sided disagreements", "19", lambda: J("human_review.json", "pair_level", "unconfirmed_one_sided"), None),
    # ---- Appendix B.4: alpha_equiv, AWQ, bias
    ("Table 6: GPTQ alpha 1B", "0.122", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "1B-gptq", "alpha_equiv"), None),
    ("Table 6: GPTQ alpha 3B", "0.149", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "3B-gptq", "alpha_equiv"), None),
    ("Table 6: 4-bit uniform reference", "0.126", lambda: J("mechanism_verdict.json", "Q1_reference", "uniform_4bit_over_gaussian"), None),
    ("Table 6: GGUF alpha 1B", "0.063", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "1B-gguf", "alpha_equiv"), None),
    ("Table 6: GGUF alpha 3B", "0.070", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "3B-gguf", "alpha_equiv"), None),
    ("Table 6: AWQ o_proj alpha 1B", "0.084", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "1B-awq", "alpha_equiv"), None),
    ("Table 6: AWQ o_proj alpha 3B", "0.100", lambda: J("mechanism_verdict.json", "Q1_alpha_estimates", "3B-awq", "alpha_equiv"), None),
    ("B.4: AWQ norm tensors rewritten, 1B", "36", lambda: J("awq_anomaly.json", "H1_norm_tensors", "1B-awq", "n") - J("awq_anomaly.json", "H1_norm_tensors", "1B-awq", "n_exactly_zero"), None),
    ("B.4: AWQ norm tensors, 1B total", "37", lambda: J("awq_anomaly.json", "H1_norm_tensors", "1B-awq", "n"), None),
    ("B.4: AWQ norm tensors rewritten, 3B", "44", lambda: J("awq_anomaly.json", "H1_norm_tensors", "3B-awq", "n") - J("awq_anomaly.json", "H1_norm_tensors", "3B-awq", "n_exactly_zero"), None),
    ("B.4: AWQ norm median Frobenius ratio (1B; 3B is 1.04)", "1.03", lambda: J("awq_anomaly.json", "H1_norm_tensors", "1B-awq", "median_frobenius"), None),
    ("B.4: |t| range, low", "7.5", lambda: min(abs(TS(c)["t"]) for c in sig()), None),
    ("B.4: |t| range, high", "36.6", lambda: max(abs(TS(c)["t"]) for c in sig()), None),
    ("B.4: pooled symmetric control mean/SD", "-0.061", lambda: J("studyb_baseline.json", "overall_shift", "mean_over_sd"), None),
    ("B.4: synthetic draws pooled", "30", lambda: J("review_audit.json", "F_bias_and_magnitude", "synthetic_single_draws_alpha_0.04", "n_draws"), None),
    ("B.4: single draws |mean/SD|, Falcon3 max", "1.75", lambda: J("review_audit.json", "F_bias_and_magnitude", "synthetic_single_draws_alpha_0.04", "max_abs_mean_over_sd_falcon3"), None),
    ("B.4: single draws |mean/SD|, six-model max", "3.26", lambda: J("review_audit.json", "F_bias_and_magnitude", "synthetic_single_draws_alpha_0.04", "max_abs_mean_over_sd"), None),
    ("B.4: mean delta_m, flipped", "-1.82", lambda: POOL("mean_delta_m_of_flips"), None),
    ("B.4: mean delta_m, not flipped", "-0.49", lambda: POOL("mean_delta_m_not_flipped"), None),
    ("B.4: fp16 boundary, Falcon3-1B", "2.5", lambda: J("threshold.json", "part1_threshold", "Falcon3-1B", "m_star"), None),
    ("B.4: fp16 boundary, Falcon3-3B", "3.8", lambda: J("threshold.json", "part1_threshold", "Falcon3-3B", "m_star"), None),
    ("B.4: geometric-mean miss at zero", "40", lambda: J("threshold.json", "part3_zero_crossing_ladder", "geomean_miss_M0_zero_no_mean"), None),
    ("B.4: geometric-mean miss at boundary", "1.5", lambda: J("threshold.json", "part3_zero_crossing_ladder", "geomean_miss_M2_star_no_mean"), None),
    ("B.4: miss across the boundary CI, low", "0.6", lambda: J("threshold.json", "part3_zero_crossing_ladder", "geomean_miss_M2_at_ci_high"), None),
    ("B.4: miss across the boundary CI, high", "13.7", lambda: J("threshold.json", "part3_zero_crossing_ladder", "geomean_miss_M2_at_ci_low"), None),
    # ---- Appendix B.4 per-model results (base-request clustered elsewhere; point values here)
    ("B.4: Falcon3-1B AUC", "0.789", lambda: RANK("Falcon3-1B")["auc"], None),
    ("B.4: Falcon3-1B breaks", "4", lambda: RANK("Falcon3-1B")["events"], None),
    ("B.4: Falcon3-3B AUC", "0.981", lambda: RANK("Falcon3-3B")["auc"], None),
    ("B.4: Falcon3-3B breaks", "55", lambda: RANK("Falcon3-3B")["events"], None),
    ("B.4: Llama-3.2-1B AUC", "0.942", lambda: RANK("Llama-3.2-1B")["auc"], None),
    ("B.4: Llama-3.2-1B breaks", "7", lambda: RANK("Llama-3.2-1B")["events"], None),
    ("B.4: Llama-3.2-3B AUC", "0.950", lambda: RANK("Llama-3.2-3B")["auc"], None),
    ("B.4: Llama-3.2-3B breaks", "24", lambda: RANK("Llama-3.2-3B")["events"], None),
    ("B.4: Qwen2.5-1.5B AUC", "0.917", lambda: RANK("Qwen2.5-1.5B")["auc"], None),
    ("B.4: Qwen2.5-1.5B breaks", "8", lambda: RANK("Qwen2.5-1.5B")["events"], None),
    ("B.4: Qwen2.5-3B AUC", "0.893", lambda: RANK("Qwen2.5-3B")["auc"], None),
    ("B.4: Qwen2.5-3B breaks", "40", lambda: RANK("Qwen2.5-3B")["events"], None),
    # ---- generation reliability, Table 7
    ("B.4.2: residual empty replies", "7", lambda: J("regrid_diagnosis.json", "repaired_config_removes_empty", "n_still_empty"), None),
    ("B.4.2: original empties diagnosed", "29", lambda: J("regrid_diagnosis.json", "n_failing_diagnosed"), None),
    ("B.4.2: matched controls", "40", lambda: J("regrid_diagnosis.json", "n_controls"), None),
    ("B.4.2: original Falcon3-1B-AWQ empties", "23", lambda: J("forensics.json", "falcon3_1b_awq_jb", "n_empty"), None),
    ("Table 7: length-matched m0 AUC", "0.930", lambda: J("regrid_summary.json", "repaired_s47", "all_cells", "m0_auc_length_matched"), None),
    ("Table 7: stratified AUC", "0.952", lambda: J("regrid_summary.json", "repaired_s47", "all_cells", "fp16_refused_auc"), None),
    ("Table 7: pooled AUC", "0.948", lambda: J("regrid_summary.json", "repaired_s47", "all_cells", "m0_auc"), None),
    ("Table 7: plain flips under sampling %, Llama-3.2", "0.5", lambda: TIER("sampling_only", "Llama-3.2|plain"), None),
    ("Table 7: plain flips under sampling %, Qwen2.5", "0.5", lambda: TIER("sampling_only", "Qwen2.5|plain"), None),
    ("Table 7: plain flips under quantization %, low (Qwen2.5)", "0.3", lambda: TIER("quantization", "Qwen2.5|plain"), None),
    ("Table 7: plain flips under quantization %, high (Falcon3)", "1.0", lambda: TIER("quantization", "Falcon3|plain"), None),
    # ---- compute and re-seed (Appendix B.5)
    ("B.5: re-seed comparisons", "1791", lambda: J("null_is_it_real.json", "n_comparisons"), None),
    ("B.5: re-seed comparisons with different text", "93", lambda: J("null_is_it_real.json", "n_text_differs"), None),
]

# Printed numbers that no shipped script produces. Listed so they are not
# mistaken for verified ones.
UNVERIFIED = []  # every printed number now has a producing script


def rounds_to(v, printed):
    """True if v rounds half-up to the printed value, as a paper rounds (0.945 -> 0.95).
    Decimal(repr(v)) avoids binary-float artefacts such as 0.945 being 0.94499..."""
    q = Decimal(printed)
    return Decimal(repr(v)).quantize(q if "." in printed else Decimal(1), rounding=ROUND_HALF_UP) == q


def main():
    fails = 0
    for where, printed, fn, tol in CLAIMS:
        try:
            v = float(fn())
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {where}: {type(e).__name__}: {e}")
            fails += 1
            continue
        target = float(printed)
        ok = abs(v - target) <= tol if tol else rounds_to(v, printed)
        fails += not ok
        note = f" (tolerance {tol})" if tol else ""
        print(f"{'ok  ' if ok else 'FAIL'}  {where:58s} paper {printed:>7}  artifact {v:.4g}{note}")
    print()
    for where, why in UNVERIFIED:
        print(f"UNVERIFIED  {where}\n            {why}")
    print(f"\n{len(CLAIMS) - fails} of {len(CLAIMS)} numbers verified, {fails} failed, {len(UNVERIFIED)} unverifiable.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
