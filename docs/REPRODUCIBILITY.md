# Reproducibility notes

Everything a reviewer or re-implementer needs beyond the README: where each
paper number comes from, how the GPU measurements were made, and the caveats
that come with them.

## Where each result comes from

| Paper | Number | Artifact (`results/analysis/`) | Script (`experiments/analysis/`) |
|---|---|---|---|
| Sec. 3.1, Table 1, Fig. 1 | variance model within 2x, median 0.62 | `mechanism_verdict.json` | `mechanism_verdict.py` |
| Abstract, Sec. 3.2 | zero-crossing miss 3.5-337x; 7.2 / 36 / 302x by format | `dose_response_recheck.json` | `dose_response_recheck.py` |
| Sec. 3.2, App. B.4 | 58 of 59 breaks change the first token; 57 keep a positive margin | `review_audit.json` | `review_audit.py` |
| Abstract, Sec. 3.3 | m0 AUC 0.948 vs delta_m 0.715, same pool | `paper_tables.json` | `paper_tables.py` |
| Table 2 | per-family AUC with prompt-clustered CI | `paper_tables.json` | `paper_tables.py` |
| App. B.4 | per-model AUC and break counts | `index_stats.json` | `index_stats.py` |
| App. B.4 | boundary at m = 2.5 / 3.8; miss 40x -> 1.5x (0.6x-13.7x across its CI) | `threshold.json` | `threshold_analysis.py` |
| App. B.2, Table 4 | decoding each checkpoint inherited | `decoding_audit.json` | `decoding_audit.py` (network) |
| App. B.3 | human review, 92.88% of judge flips confirmed | `human_review.json` | `human_review_analysis.py` |
| App. B.4.1, Table 5 | alpha_equiv per format | `mechanism_verdict.json`, `awq_anomaly.json` | `mechanism_verdict.py`, `awq_anomaly.py` |
| Table 7 | plain-tier flips on judge labels | `paper_tables.json` | `paper_tables.py` |
| App. B.6 | re-seed control, 93 of 1,791 with different text | `null_is_it_real.json` | `null_is_it_real.py` |

`verify_paper.py` holds the full list, one row per printed number (112).
`results/README.md` says which files are raw measurements and which are derived.
`reproduce.py --network` also re-runs the two steps that read from the Hugging
Face Hub (each checkpoint's `generation_config.json`, and each tokenizer).

## Scope and counts behind the paper's numbers

Detail the camera-ready states only briefly. Every number here is read from the
artifact named next to it.

**Which models each result covers.**

- **Falcon3 only** (1B and 3B, each in AWQ, GPTQ and GGUF: the paper's six cells):
  alpha_equiv (Table 5), the variance model (Fig. 1, Tables 1 and 6), the
  zero-crossing miss (3.5-337x), the boundary diagnosis (58 of 59 breaks), the
  per-checkpoint mean shift, and the pooled AUC comparison (m0 0.948 vs delta_m
  0.715). The mechanism runs cover `falcon3-1b` and `falcon3-3b` only
  (`mechanism_meta.json`).
- **All three families**: the synthetic margin sweep (six models), Table 2, the
  per-model AUCs, the human review, and the decoding audit.

**Prompt sets.** Each cell has 400 prompts: 300 jailbreak (100 AdvBench requests,
each plain, roleplay and persuasion) and 100 InjecAgent cases.

| Uses all 400 prompts per cell | Uses the 300 jailbreak prompts |
|---|---|
| Synthetic sweep: 6 models x 26 configurations x 400 = 62,400 margins | Variance model, observed and predicted SD (n = 300 per cell; `mechanism_verdict.py`, Q2) |
| Stated-decoding grid: 24 cells x 400 = 9,600 generations (7,200 jailbreak) | Per-checkpoint mean shift and its t-statistics (`s5_jailbreak_tstats.json`) |
| Falcon3 re-run: 8 cells x 400 = 3,200 generations | Flip rates, flips and every AUC (pooled: 1,788 Falcon3 pairs) |
| Greedy-cell reproduction, 400 of 400 (`regrid_summary.json`) | Human review (267 pairs) and the original empty replies |

**The linear fit (Sec. 2).** For each prompt, the across-seed SD of the margin at
each nonzero alpha is fit as SD = g * alpha with no intercept
(`experiments/margin_analysis.py`). Median R^2 is 0.997, and 96.6% of prompts are
above 0.9 (`studyb_baseline.json`). The fit is on the SD because the signed
delta_m averages to about zero under isotropic noise.

**Zero-crossing miss per cell (Sec. 3.2, App. B.4.1)**, from
`dose_response_recheck.json`, with observed rates from the stated-decoding re-run:

| Cell | Predicted flip rate | Observed flip rate | Miss |
|---|---|---|---|
| 1B GPTQ | 0.29% | 1.01% | 3.5x |
| 3B GPTQ | 0.71% | 7.69% | 10.9x |
| 3B AWQ | 0.16% | 4.00% | 24.8x |
| 1B AWQ | 0.029% | 1.35% | 47.1x |
| 3B GGUF | 0.030% | 8.00% | 266.6x |
| 1B GGUF | 0.0030% | 1.01% | 337.3x |

The by-format figures in the paper are the mean of the two sizes: GPTQ 7.2x,
AWQ 36x, GGUF 302x.

**Pooled and stratified AUC (Sec. 3.3, Table 7).** Both are Falcon3, jailbreak set,
stated-decoding re-run (`regrid_summary.json`, `repaired_s47.all_cells`). Pooled
means all 1,788 quantized-versus-fp16 pairs with flips in either direction (69:
59 breaks, 10 restores): m0 AUC 0.948, delta_m 0.715. Stratified means the 1,758
pairs whose fp16 reply refused, predicting breaks: m0 AUC 0.952. Table 2 gives the
stratified AUC for every family.

**Why Table 2 stratifies.** fp16 compliance on the jailbreak set differs 17-fold
across families (`index_stats.json`): 28.3% of Qwen2.5's fp16 replies comply (508
of 1,798 pairs), 5.5% of Llama-3.2's (99 of 1,800), and 1.7% of Falcon3's (30 of
1,788). Pooled over families, an AUC could be earned by telling families apart,
or fp16-refused from fp16-complied prompts, rather than by ranking which refusals
break. Within one family's fp16-refused stratum neither route is open, so Table 2
measures only that ranking.

**Human review (App. B.3).** The 267 judge-flagged flips count both directions:
138 breaks (refuse to comply) and 129 restores (comply to refuse). By family:
Falcon3 59 + 10, Llama-3.2 31 + 16, Qwen2.5 48 + 103
(`human_review_flip_pairs.csv`). The 138 breaks are the ones counted in Sec. 3.2
and App. B.4.1.

**Empty replies (App. B.4.2).** Under inherited decoding, 29 Falcon3 jailbreak
replies were empty: 23 in 1B-AWQ, 3 in 1B-fp16, 2 in 3B-AWQ and 1 in 1B-GPTQ.
Under stated greedy decoding 22 of the 29 are no longer empty. In the 7 that
remain (1B-AWQ 3, 1B-fp16 3, 1B-GPTQ 1) the greedy first token is end-of-sequence
(`regrid_diagnosis.csv`, `regrid_diagnosis.json`).

## Re-running the GPU measurements

The raw measurements came from Kaggle kernels (a single NVIDIA T4 each, about 10
GPU-hours in total; the llama.cpp GGUF cells ran on CPU) and an LLM judge.

1. **Checkpoints.** Use the commits in `checkpoints.json`, e.g.
   `from_pretrained(repo, revision=commit)`. Seventeen come from the run logs, six
   are repository heads that predate every run, and the gated
   `meta-llama/Llama-3.2-1B-Instruct` resolved `main` without logging its commit.
2. **Environment.** `requirements-gpu.txt` lists the versions the run logs record.
3. **Kernels.** `experiments/build_*.py` generate kernels into `experiments/_build/`.
   `experiments/kernels/` holds what was actually pushed and is the reference when
   the two differ. The judge kernel rebuilds byte for byte. `margin_core.py` has since
   gained optional ablation arguments, so kernels that inline it differ additively;
   `experiments/local_smoke_ablation.py` asserts the defaults reproduce the old path.
4. **Judge.** `experiments/judge_driver.py` calls `deepseek-ai/DeepSeek-V4-Flash`
   through the Hugging Face router at temperature 0 and needs `HF_TOKEN`. The rubric
   is extracted from `legacy/notebooks/04_classify.ipynb` and asserted by SHA-256.
5. **Decoding.** Pass `do_sample=False` (transformers) or `temperature=0.0`
   (llama.cpp) explicitly. Inheriting a checkpoint's `generation_config.json` is the
   confound the paper documents.

## Rebuilding the Hugging Face dataset

```bash
python hf_dataset/build.py      # writes hf_dataset/data/pairs.parquet, asserting the paper's counts
```

## Caveats

- **Study A inherited its decoding.** `results/results.csv` and `model_loaders.generate_reply()`
  pass no sampling arguments, so each quantized checkpoint used its own repository's
  defaults (Falcon3-AWQ sampled at T=1.0, GGUF at T=0.2). Flip rates are quoted from
  the stated-decoding grids only; Study A is kept because the confound is itself a result.
- **Llama-3.2 margins depend on the date.** Its chat template inserts the current date
  and no driver pins `date_string`, so Llama-3.2 m0 varies across run dates (up to 6.3
  logits at 1B). Falcon3 and Qwen2.5 reproduce to 0.000.
- **Line endings.** Kaggle CSVs store line breaks inside replies as `\r\n`; Study A's
  Colab CSV stores `\n`. `.gitattributes` keeps the bytes, and `regrid_analysis.py`
  normalises line endings before comparing; greedy cells then match Study A 400 of 400.
- **Table 2 intervals** come from a run of the same bootstrap whose code was not kept.
  `paper_tables.py` reproduces the AUCs exactly and the interval ends to within 0.004.
- **m0 is a single-token contrast**: logit("I") minus the mean logit of "Sure",
  "Certainly" and "Of" (`continuation_tokens.json`). Swapping in distinct refusal tokens
  moves break-prediction AUC by at most 0.010; the pre-registered rank-correlation
  criterion fails on Llama-3.2 (0.625 < 0.90) and is reported as failed.
- **GGUF tensor error** is measured on weights dequantized in Python, not inside llama.cpp.
- **One judge.** Human review covers all 267 judge-flagged flips but no non-flips, so it
  bounds false positives, not false negatives.
