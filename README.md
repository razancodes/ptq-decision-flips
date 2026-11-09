# A First-Order Theory of Decision Flips Under Post-Training Quantization

**What It Predicts, and Where It Breaks**

Muhammed Razan, G Shivam (B.M.S. College of Engineering)
NeurIPS 2026 Workshop: AXIOM Foundations of Efficient Deep Learning

Code, data and every intermediate result behind the paper. All paper numbers
regenerate from the raw measurements on a laptop CPU in about a minute, and
`verify_paper.py` checks each one against the artifact that produces it.

> **Content warning.** `data/` holds harmful-instruction prompts (from AdvBench
> and InjecAgent) and `results/` holds model replies to them, some of which
> comply. They are released for safety research only.

## The result in one paragraph

We treat quantization as a weight perturbation and ask whether first-order
theory, estimated from forward passes on the unquantized model, predicts which
refusal decisions a real quantization format flips. It predicts the *size* of
the margin shift well: within a factor of two on all six Falcon3
model-by-format cells, median observed/predicted ratio 0.62. It does not
predict *which decisions flip*: the zero-crossing criterion under-predicts flip
rates by 3.5 to 337 times, because the behavioural boundary is not at zero (in
57 of 59 Falcon3 breaks the measured post-quantization margin is still
positive). The pre-quantization margin itself predicts flips far better than
the theory's own quantity: AUC 0.948 against 0.715 on the same pool.

## Reproduce the paper

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce.py      # regenerates results/analysis/ and results/margin/ from raw data (~1 min)
git status               # should be clean: the regenerated files equal the committed ones
python verify_paper.py   # checks every number the paper prints
```

`reproduce.py --network` also re-runs the two steps that read from the Hugging
Face Hub (each checkpoint's `generation_config.json`, and each tokenizer).
Their outputs are committed, so the default run does not need them.

## Where each result comes from

| Paper | Number | Artifact | Script |
|---|---|---|---|
| Sec. 3.1, Table 1, Fig. 1 | variance model within 2x, median 0.62 | `results/analysis/mechanism_verdict.json` | `experiments/analysis/mechanism_verdict.py` |
| Abstract, Sec. 3.2 | zero-crossing miss 3.5-337x; 7.2 / 36 / 302x by format | `dose_response_recheck.json` | `dose_response_recheck.py` |
| Sec. 3.2, App. B.4 | 58 of 59 breaks change the first token; 57 keep a positive margin | `review_audit.json` | `review_audit.py` |
| Abstract, Sec. 3.3 | m0 AUC 0.948 vs delta_m 0.715, same pool | `paper_tables.json` | `paper_tables.py` |
| Table 2 | per-family AUC with prompt-clustered CI | `paper_tables.json` | `paper_tables.py` |
| App. B.4 | per-model AUC and break counts | `index_stats.json` | `index_stats.py` |
| App. B.4 | boundary at m = 2.5 / 3.8; miss 40x -> 1.5x | `threshold.json` | `threshold_analysis.py` |
| App. B.2, Table 4 | decoding each checkpoint inherited | `decoding_audit.json` | `decoding_audit.py` (network) |
| App. B.3 | human review, 92.88% of judge flips confirmed | `human_review.json` | `human_review_analysis.py` |
| App. B.4, Table 6 | alpha_equiv per format | `mechanism_verdict.json`, `awq_anomaly.json` | `mechanism_verdict.py`, `awq_anomaly.py` |
| App. B.5 | re-seed control, 93 of 1,791 with different text | `null_is_it_real.json` | `null_is_it_real.py` |

`verify_paper.py` holds the full list, one row per printed number.

## Layout

```
reproduce.py, verify_paper.py   run the analysis; check the paper
checkpoints.json                Hugging Face commit of every checkpoint used
model_registry.py               model x format -> checkpoint repository
model_loaders.py                loaders (legacy Study A path; see Known issues)
data/                           attack prompts and the script that built them
experiments/                    GPU drivers: generation, margins, tensor error, judge
experiments/kernels/            each Kaggle kernel exactly as it ran (provenance; do not edit)
experiments/analysis/           CPU analysis, run in order by reproduce.py
results/results.csv             Study A, the original grid (decoding inherited; superseded)
results/margin/                 synthetic perturbation sweep and re-seed control
results/analysis/               stated-decoding generations, judge labels, every derived artifact
legacy/notebooks/               the Colab notebooks behind Study A, kept as a record
```

`results/README.md` lists which files are raw measurements and which are derived.

## Re-running the GPU experiments

The raw measurements came from Kaggle kernels (a single NVIDIA T4 each, about 10
GPU-hours in total; the llama.cpp GGUF cells ran on CPU) and an LLM judge. To
re-run them:

1. **Checkpoints.** Use the commits in `checkpoints.json`, e.g.
   `from_pretrained(repo, revision=commit)`. Seventeen are taken from the run logs;
   six are repository heads that predate every run; the gated
   `meta-llama/Llama-3.2-1B-Instruct` resolved `main` and its commit was not logged.
2. **Environment.** `requirements-gpu.txt` lists the versions the run logs record.
3. **Kernels.** `experiments/build_*.py` generate kernels into `experiments/_build/`.
   `experiments/kernels/` holds what was actually pushed, and is the reference when
   the two differ. The judge kernel rebuilds byte for byte. `margin_core.py` has since
   gained optional ablation arguments, so kernels that inline it differ additively
   (the defaults reproduce the old path, asserted by `experiments/local_smoke_ablation.py`).
4. **Judge.** `experiments/judge_driver.py` calls `deepseek-ai/DeepSeek-V4-Flash`
   through the Hugging Face router at temperature 0 and needs `HF_TOKEN`. The rubric
   text is extracted from `legacy/notebooks/04_classify.ipynb` and asserted by
   SHA-256 at build time. Rebuilding the judge kernel reproduces the pushed one byte
   for byte.

Generation must state its decoding. Every stated-decoding run passes
`do_sample=False` (transformers) or `temperature=0.0` (llama.cpp) explicitly. Do
not inherit a checkpoint's `generation_config.json`: that is the confound the
paper documents.

## Known issues, stated rather than hidden

- **Study A inherited its decoding.** `results/results.csv` and `model_loaders.generate_reply()`
  pass no sampling arguments, so each quantized checkpoint used its own repository's
  defaults (Falcon3-AWQ sampled at T=1.0, GGUF at T=0.2). Flip rates are quoted from
  the stated-decoding grids only. Study A is kept because the confound is itself a result.
- **Llama-3.2 margins depend on the date.** Its chat template inserts the current date
  and no driver pins `date_string`, so Llama-3.2 m0 differs across run dates (up to 6.3
  logits at 1B). Falcon3 and Qwen2.5 reproduce to 0.000.
- **Line endings.** Kaggle CSVs store line breaks inside replies as `\r\n`; Study A's
  Colab CSV stores `\n`. `.gitattributes` keeps the bytes, and the Study A comparison in
  `regrid_analysis.py` normalises line endings explicitly. Greedy cells then reproduce
  Study A at 400 of 400.
- **Table 2 intervals** come from a run of the same bootstrap whose code was not kept.
  `paper_tables.py` reproduces the AUCs exactly and the interval ends to within 0.004.
- **The m0 index is a single-token contrast**: logit("I") minus the mean logit of
  "Sure", "Certainly" and "Of". All three refusal continuations share the first token
  "I" on every model (`continuation_tokens.json`). Swapping in distinct refusal tokens
  moves break-prediction AUC by at most 0.010; the pre-registered rank-correlation
  criterion fails on Llama-3.2 (0.625 < 0.90) and is reported as failed.
- **GGUF tensor error** is measured on weights dequantized in Python, not inside llama.cpp.
- **One judge.** Human review covers all 267 judge-flagged flips but no non-flips, so it
  bounds false positives, not false negatives.
- **One table entry has no producer here.** Table 7's plain-tier sampling rate
  (0.17-0.56%) came from a classifier-labelled analysis whose code is not in this
  repository; `verify_paper.py` lists it as unverified and gives the judge-label figures.

## Citation

```bibtex
@inproceedings{razan2026firstorder,
  title     = {A First-Order Theory of Decision Flips Under Post-Training Quantization:
               What It Predicts, and Where It Breaks},
  author    = {Razan, Muhammed and Shivam, G},
  booktitle = {NeurIPS 2026 Workshop: AXIOM Foundations of Efficient Deep Learning},
  year      = {2026}
}
```

## License

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA`). The attack
prompts derive from AdvBench and InjecAgent, and the replies were produced by
Falcon3, Llama-3.2 and Qwen2.5; their own licenses apply to those parts.
