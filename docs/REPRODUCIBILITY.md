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
| App. B.4, Table 6 | alpha_equiv per format | `mechanism_verdict.json`, `awq_anomaly.json` | `mechanism_verdict.py`, `awq_anomaly.py` |
| Table 7 | plain-tier flips on judge labels | `paper_tables.json` | `paper_tables.py` |
| App. B.5 | re-seed control, 93 of 1,791 with different text | `null_is_it_real.json` | `null_is_it_real.py` |

`verify_paper.py` holds the full list, one row per printed number (108).
`results/README.md` says which files are raw measurements and which are derived.
`reproduce.py --network` also re-runs the two steps that read from the Hugging
Face Hub (each checkpoint's `generation_config.json`, and each tokenizer).

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
