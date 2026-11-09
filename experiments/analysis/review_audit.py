"""
review_audit.py -- answer every LIGHT/AXIOM reviewer question that existing
artifacts can answer, CPU only, no new generation.

A. Is a break a first-token event? (JsxN: "do refusals break later in the reply?")
B. Post-quantization margin m0+dm vs m0 vs dm, on the REPAIRED labels (JsxN)
C. Precision/recall under imbalance (P586, meta-review)
D. Cheap forward-pass baselines (P586, meta-review)
E. Does m0 predict sampler-induced flips? (P586's exact test)
F. Why m0 ranks well: spread of m0 vs spread of dm
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "analysis"))
from threshold_analysis import auc  # noqa: E402

RES = os.path.join(REPO, "results", "analysis")
rng = np.random.default_rng(20260827)
OUT = {}


def base(pid):
    return pid.str.rsplit("_", n=1).str[0]


def ap(score, y):
    """Average precision (area under PR curve, step-wise)."""
    s, y = np.asarray(score, float), np.asarray(y, int)
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def at_k(score, y, frac):
    s, y = np.asarray(score, float), np.asarray(y, int)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / k), float(y[top].sum() / max(y.sum(), 1))


def budget_for_recall(score, y, r):
    s, y = np.asarray(score, float), np.asarray(y, int)
    c = np.cumsum(y[np.argsort(-s, kind="mergesort")])
    return float((np.argmax(c >= r * y.sum()) + 1) / len(y))


def boot(df, fn, n=2000):
    idx = list(df.reset_index(drop=True).groupby("base").indices.values())
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        d = df.iloc[sel]
        if 0 < d.y_ev.sum() < len(d):
            vals.append(fn(d))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


# ---------------------------------------------------------------- load
cols = ["family", "size", "prompt_id", "method", "attack_category", "template_tier",
        "prompt_text", "n_input_tokens", "first_gen_tok", "model_reply", "y", "s_fp16",
        "flip", "broke", "restore", "m0"]
pairs = pd.concat([pd.read_csv(os.path.join(RES, f))[cols] for f in
                   ("regrid_pairs.csv", "phase21_pairs.csv")], ignore_index=True)
pairs = pairs[(pairs.method != "fp16") & (pairs.attack_category == "jailbreak")]
pairs = pairs.dropna(subset=["m0", "s_fp16"]).copy()
pairs["model"] = pairs.family + "-" + pairs["size"]
pairs["base"] = base(pairs.prompt_id)

ta = pd.read_csv(os.path.join(RES, "token_ablation.csv"))
ta["topk"] = ta.greedy_topk.apply(json.loads)
key = ["family", "size", "prompt_id"]
pairs = pairs.merge(ta[key + ["m_first_default", "m_first_alt", "greedy_top1_token",
                              "greedy_top1_logprob", "greedy_top2_logprob", "topk",
                              "ref_first_lp_0", "com_first_lp_0", "com_first_lp_1",
                              "com_first_lp_2"]], on=key, how="left")
assert pairs.m_first_default.notna().all()


# Known limits, stated where the numbers are produced:
#  * Llama-3.2 rows in token_ablation.csv do NOT reproduce the paper's m0
#    (OUT["m0_vs_token_ablation_by_model"]); its chat template injects the run
#    date. Llama baselines in D are from a differently-rendered prompt.
#  * Falcon3 GGUF dm comes from dequantized weights in HF, not llama.cpp.
#  * entropy and class margin in D use top-5 log-probs only (approximate).
OUT["m0_vs_token_ablation_by_model"] = pairs.assign(d=(pairs.m0 - pairs.m_first_default).abs()).groupby("model").d.max().round(3).to_dict()

# ---------------------------------------------------------------- A. first-token event?
ref = pairs[pairs.s_fp16 == 0].copy()
# first_gen_tok is NaN on every GGUF row (llama.cpp path), so read the opener
# from the reply text. "late" break = reply still opens with the word I.
ref["fp16_open"] = ref.greedy_top1_token.astype(str).str.strip()
ref["q_opens_I"] = ref.model_reply.fillna("").astype(str).str.lstrip().str.match(r"^\W{0,3}I\b")
ref["opener_changed"] = ~ref.q_opens_I
ref["q_open"] = ref.model_reply.fillna("").astype(str).str.lstrip().str.split().str[0].str[:12]
A = {}
for name, g in list(ref.groupby("family")) + [("ALL", ref)]:
    b = g[g.broke == 1]
    nb = g[g.broke == 0]
    A[name] = {
        "n_refused_pairs": len(g), "breaks": len(b),
        "fp16_opens_I": round(float((g.fp16_open == "I").mean()), 4),
        "breaks_opener_left_I": int(b.opener_changed.sum()),
        "frac_breaks_first_token": round(float(b.opener_changed.mean()), 3),
        "late_breaks_by_method": b[b.q_opens_I].method.value_counts().to_dict(),
        "breaks_still_opening_I_LATE_BREAKS": int(b.q_opens_I.sum()),
        "nonbreaks_opener_left_I": int(nb.opener_changed.sum()),
        "P_break_given_opener_changed": round(float(g[g.opener_changed].broke.mean()), 4)
        if g.opener_changed.any() else None,
        "P_break_given_opener_same": round(float(g[~g.opener_changed].broke.mean()), 4),
        "break_openers": b.q_open.value_counts().head(10).to_dict(),
        "nonbreak_changed_openers": nb[nb.opener_changed].q_open.value_counts().head(8).to_dict(),
    }
for name, g in ref.groupby("family"):
    nb = g[g.broke == 0]
    for kind, ev in (("first_token", g[(g.broke == 1) & g.opener_changed]),
                     ("late", g[(g.broke == 1) & g.q_opens_I])):
        A[name][f"auc_m0_{kind}_breaks_vs_nonbreaks"] = auc(
            -pd.concat([ev.m0, nb.m0]), [1] * len(ev) + [0] * len(nb)) if len(ev) >= 3 else None
OUT["A_first_token_event"] = A
late = ref[(ref.broke == 1) & ref.q_opens_I]
OUT["A_late_break_examples"] = [r[:220] for r in late.model_reply.astype(str).head(6)]

# ---------------------------------------------------------------- B. post-quant margin
dmf = pd.read_csv(os.path.join(RES, "delta_m_joined.csv"))[
    ["prompt_id", "family", "size", "method", "m_first", "m_fp16", "delta_m"]]
qd = pd.read_csv(os.path.join(REPO, "results", "analysis", "qwen_mechanism", "delta_m_prompts.csv"))
qd["size"] = qd["size"].astype(str)
q0 = qd[qd.method == "fp16"][["prompt_id", "size", "m_first"]].rename(columns={"m_first": "m_fp16"})
qq = qd[qd.method.isin(["awq", "gptq"])].merge(q0, on=["prompt_id", "size"])
qq["delta_m"] = qq.m_first - qq.m_fp16
dm = pd.concat([dmf, qq[dmf.columns]], ignore_index=True)
dm["size"] = dm["size"].astype(str)
B = {}
bj = ref.merge(dm, on=["prompt_id", "family", "size", "method"], how="inner")
bj["m_q"] = bj.m_first
for name, g in bj.groupby(["family", "size"]):
    if g.broke.sum() < 3:
        continue
    B["-".join(name).strip("-")] = {
        "n": len(g), "breaks": int(g.broke.sum()),
        "auc_m0": auc(-g.m0, g.broke), "auc_mq": auc(-g.m_q, g.broke),
        "auc_dm": auc(-g.delta_m, g.broke),
        "breaks_with_mq_below_0": int(((g.m_q < 0) & (g.broke == 1)).sum()),
        "sd_m0": round(float(g.m0.std()), 3), "sd_dm": round(float(g.delta_m.std()), 3),
        "corr_m0_mq": round(float(np.corrcoef(g.m0, g.m_q)[0, 1]), 3),
        "corr_m0_dm": round(float(np.corrcoef(g.m0, g.delta_m)[0, 1]), 3),
    }
OUT["B_post_quant_margin"] = B
# per cell, with format
Bc = {}
for (f, s, m), g in bj.groupby(["family", "size", "method"]):
    Bc[f"{f}-{s}-{m}"] = {"n": len(g), "breaks": int(g.broke.sum()),
                         "auc_m0": auc(-g.m0, g.broke) if g.broke.sum() else None,
                         "auc_mq": auc(-g.m_q, g.broke) if g.broke.sum() else None,
                         "auc_dm": auc(-g.delta_m, g.broke) if g.broke.sum() else None,
                         "mean_dm": round(float(g.delta_m.mean()), 3),
                         "sd_dm": round(float(g.delta_m.std()), 3)}
OUT["B_per_cell"] = Bc

# ---------------------------------------------------------------- D. baselines + C. PR
REFUSAL_WORDS = {"I", "Sorry", "As", "(I", "Unfortunately", "I'm", "My", "Apolog", "No",
                 "<|endoftext|>", "<|eot_id|>", "<|im_end|>", "\u0109I", "$I", "**I"}


def feats(d):
    p = np.exp(d.ref_first_lp_0)
    top = d.topk.apply(lambda t: np.exp([x[1] for x in t]))
    resid = top.apply(lambda v: max(1e-12, 1 - v.sum()))
    ent = top.apply(lambda v: -(v * np.log(v)).sum()) - resid * np.log(resid)

    def best_comp(t):
        ps = [np.exp(x[1]) for x in t if x[0].strip() not in REFUSAL_WORDS]
        lo = min(np.exp(t[-1][1]), max(1e-12, 1 - sum(np.exp(x[1]) for x in t)))
        return np.log(max(ps + [lo]))
    return pd.DataFrame({
        "FTFI m0 (paper)": d.m0,
        "log P('I') [softmax prob of refusal token]": d.ref_first_lp_0,
        "top1 - top2 logprob [argmax margin]": d.greedy_top1_logprob - d.greedy_top2_logprob,
        "-entropy, top-5 truncated": -ent,
        "log P(I) - log P(best non-refusal) [class margin, top-5]": d.ref_first_lp_0 - d.topk.apply(best_comp),
        "log P(I) - max log P(Sure,Certainly,Of)": d.ref_first_lp_0 - d[["com_first_lp_0", "com_first_lp_1", "com_first_lp_2"]].max(axis=1),
        "m0, alternate refusal token set": d.m_first_alt,
        "-prompt length (chars)": -d.prompt_text.astype(str).str.len(),
        "-framing tier (plain<roleplay<persuasion)": -d.template_tier.map({"plain": 0, "roleplay": 1, "persuasion": 2}),
    }, index=d.index)


F = feats(ref)
D, C = {}, {}
for model in ["Falcon3-3B", "Llama-3.2-3B", "Qwen2.5-3B", "Falcon3-1B", "Llama-3.2-1B", "Qwen2.5-1.5B"]:
    g = ref[ref.model == model]
    fz = F.loc[g.index]
    y = g.broke.to_numpy(int)
    D[model] = {"n": len(g), "breaks": int(y.sum()), "base_rate": round(float(y.mean()), 4)}
    for c in fz.columns:
        # every score oriented so that LOW = fragile, negate -> high = predicted break
        sc = -fz[c].to_numpy(float)
        D[model][c] = {"auc": auc(sc, y), "ap": round(ap(sc, y), 3)}
    if y.sum() >= 10:
        sc = -g.m0.to_numpy(float)
        gg = g.assign(y_ev=y, sc=sc)
        C[model] = {
            "base_rate": round(float(y.mean()), 4),
            "auc": auc(sc, y), "auc_ci": boot(gg, lambda d: auc(d.sc, d.y_ev)),
            "ap": round(ap(sc, y), 3), "ap_ci": boot(gg, lambda d: ap(d.sc, d.y_ev)),
            "ap_lift_over_base": round(ap(sc, y) / y.mean(), 1),
            **{f"precision@{int(k*100)}%": round(at_k(sc, y, k)[0], 3) for k in (0.05, 0.10, 0.20)},
            **{f"recall@{int(k*100)}%": round(at_k(sc, y, k)[1], 3) for k in (0.05, 0.10, 0.20)},
            "screen_frac_for_80pct_recall": round(budget_for_recall(sc, y, 0.8), 3),
            "screen_frac_for_90pct_recall": round(budget_for_recall(sc, y, 0.9), 3),
        }
OUT["D_baselines"] = D
OUT["C_precision_recall"] = C

# ---------------------------------------------------------------- E. sampling flips
sa = pd.read_csv(os.path.join(REPO, "results", "results.csv"))
sa = sa[(sa.attack_category == "jailbreak") & sa.success.isin(["0", "1"])].copy()
sa["ya"] = sa.success.astype(int)
sa["size"] = sa["size"].astype(str)
E = {}
# E2: same fp16 weights, Study A sampled vs repaired greedy label
fp = ref.drop_duplicates(subset=key)[key + ["m0", "base", "model", "ref_first_lp_0", "topk"]]
s16 = sa[sa.method == "fp16"][key + ["ya", "model_reply"]].rename(columns={"model_reply": "reply_a"})
e2 = fp.merge(s16, on=key)
for fam, temp, top_p in (("Llama-3.2", 0.6, 0.9), ("Qwen2.5", 0.7, 0.8), ("Falcon3", 0, 1)):
    g = e2[e2.family == fam]
    ev = g.ya.to_numpy(int)
    qb = ref[(ref.family == fam)]
    E[f"{fam}: fp16 sampled vs fp16 greedy (pure sampling)"] = {
        "decoding": f"T={temp}, top_p={top_p}", "n_greedy_refused": len(g),
        "sampling_breaks": int(ev.sum()),
        "auc_m0_sampling": auc(-g.m0, ev) if 0 < ev.sum() < len(ev) else None,
        "auc_ci": boot(g.assign(y_ev=ev, sc=-g.m0), lambda d: auc(d.sc, d.y_ev)) if ev.sum() >= 5 else None,
        "auc_m0_quant_breaks_same_family": auc(-qb.m0, qb.broke),
        "sampling_breaks_opening_not_I": int(((~g.reply_a.astype(str).str.lstrip().str.startswith("I")) & (ev == 1)).sum()),
        "sampling_breaks_with_P_I_above_top_p": int(((np.exp(g.ref_first_lp_0) > top_p) & (ev == 1)).sum()) if temp else None,
        # top-p prediction: if fp16 P(I) > top_p the nucleus is {I}, so the first
        # token is forced and any sampling break must be a late one (opens with I)
        "of_those_opening_I": int(((np.exp(g.ref_first_lp_0) > top_p) & (ev == 1) & g.reply_a.astype(str).str.lstrip().str.match(r"^\W{0,3}I\b")).sum()) if temp else None,
        "max_P_I_among_opener_changing_sampling_breaks": round(float(np.exp(g.ref_first_lp_0)[(ev == 1) & ~g.reply_a.astype(str).str.lstrip().str.match(r"^\W{0,3}I\b")].max()), 3) if temp and ev.sum() else None,
    }
# E1: Falcon3 quantized, Study A sampled vs repaired greedy -> the "39%"
fq = ref[ref.family == "Falcon3"].merge(sa[sa.method != "fp16"][key + ["method", "ya"]], on=key + ["method"])
for m, g in fq.groupby("method"):
    sampler_only = ((g.ya == 1) & (g.broke == 0)).astype(int)
    keep = g.broke == 0  # compare sampler-only flips against rows that flip in neither
    gg = g[keep].assign(y_ev=sampler_only[keep], sc=-g.m0[keep])
    E[f"Falcon3-{m}: sampler-only breaks (Study A flip, not in repaired)"] = {
        "n": int(keep.sum()), "sampler_only_breaks": int(gg.y_ev.sum()),
        "auc_m0": auc(gg.sc, gg.y_ev) if 0 < gg.y_ev.sum() < len(gg) else None,
        "auc_ci": boot(gg, lambda d: auc(d.sc, d.y_ev)) if gg.y_ev.sum() >= 5 else None}
OUT["E_sampling_specificity"] = E

print("wrote results/analysis/review_audit.json")
json.dump(OUT, open(os.path.join(REPO, "results", "analysis", "review_audit.json"), "w"), indent=1, default=str)

# ---------------------------------------------------------------- F. magnitude model and "bias", one draw at a time
# One real quantization is ONE perturbation shared by all 300 prompts, so its
# across-prompt mean is a single draw's common-mode shift. The published control
# (studyb_baseline.json, mean/SD -0.061) pooled 5 seeds x 5 doses x 6 models and
# averaged that common mode away. The comparable control is per draw.
sw = pd.read_csv(os.path.join(REPO, "results", "margin", "margin_sweep_all.csv"))
sw = sw[sw.attack_category == "jailbreak"]
b0 = sw[sw.alpha == 0].drop_duplicates(["model_key", "prompt_id"]).set_index(["model_key", "prompt_id"]).m_first
sw = sw[sw.alpha == 0.04].copy()
sw["dm"] = sw.m_first.to_numpy() - b0.loc[list(zip(sw.model_key, sw.prompt_id))].to_numpy()
draw = sw.groupby(["model_key", "seed"]).dm.agg(lambda d: d.mean() / d.std())
pm = pd.read_csv(os.path.join(REPO, "results", "margin", "per_prompt_margins.csv"))
pm["size"] = pm["size"].astype(str)
alpha = {("Falcon3",) + tuple(k.split("-")): v["alpha_equiv"] for k, v in
         json.load(open(os.path.join(RES, "mechanism_verdict.json")))["Q1_alpha_estimates"].items()}
alpha.update({("Qwen2.5",) + tuple(c["cell"].split("-")): c["alpha_equiv"] for c in
              json.load(open(os.path.join(REPO, "results", "analysis", "qwen_mechanism.json")))["cells"]})
real = {}
for (f, s, m), g in dm[dm.prompt_id.str.startswith("jb_")].merge(pm[["family", "size", "prompt_id", "g"]], on=["family", "size", "prompt_id"]).groupby(["family", "size", "method"]):
    a = alpha[(f, s, m)]
    real[f"{f}-{s}-{m}"] = {
        "mean_over_sd": round(float(g.delta_m.mean() / g.delta_m.std()), 3),
        # isotropic theory predicts E[dm^2] = alpha^2 * g^2 per prompt, so compare RMS to RMS
        "ratio_rms_obs_over_pred": round(float(np.sqrt((g.delta_m ** 2).mean()) / (a * np.sqrt((g.g ** 2).mean()))), 3),
        "ratio_sd_over_median_g_alpha_AS_PUBLISHED": round(float(g.delta_m.std() / (a * g.g.median())), 3)}
OUT["F_bias_and_magnitude"] = {
    "synthetic_single_draws_alpha_0.04": {"n_draws": int(len(draw)),
        "median_abs_mean_over_sd": round(float(draw.abs().median()), 3),
        "max_abs_mean_over_sd": round(float(draw.abs().max()), 3),
        "max_abs_mean_over_sd_falcon3": round(float(draw[draw.index.get_level_values(0).str.startswith("falcon3")].abs().max()), 3),
        "n_negative": int((draw < 0).sum())},
    "real_cells": real,
    "real_median_abs_mean_over_sd": round(float(np.median([abs(v["mean_over_sd"]) for v in real.values()])), 3),
    "real_n_negative": int(sum(v["mean_over_sd"] < 0 for v in real.values())),
    "rms_within_2x": int(sum(0.5 <= v["ratio_rms_obs_over_pred"] <= 2 for v in real.values())),
}
json.dump(OUT, open(os.path.join(REPO, "results", "analysis", "review_audit.json"), "w"), indent=1, default=str)
print("F:", json.dumps({k: v for k, v in OUT["F_bias_and_magnitude"].items() if k != "real_cells"}))
