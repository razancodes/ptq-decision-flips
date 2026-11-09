# Results

**Raw** files are measurements from GPU runs or the LLM judge; nothing in this
repository regenerates them on CPU. **Derived** files are rewritten by
`reproduce.py`, and a clean run leaves them byte-identical. Labels everywhere
are from the LLM judge (`deepseek-ai/DeepSeek-V4-Flash`, temperature 0), except
`human_review_flip_pairs_SCORED.csv`, which is the blind human review.

## Study A: the original grid (decoding inherited, superseded)

| File | Kind | Contents |
|---|---|---|
| `results.csv` | raw | 11,200 rows: 6 models + SmolLM2 x 4 precisions x 400 prompts, Colab. Each quantized checkpoint used its own repository's decoding defaults. Kept as the record of the confound; flip rates are not quoted from it. |

## Synthetic perturbation sweep and re-seed control (`margin/`)

| File | Kind | Contents |
|---|---|---|
| `margin_sweep_all.csv` | raw | 62,400 rows: 6 models x 400 prompts x 26 (alpha, seed) configs. Use `m_first`; `m_seq` is a deprecated measure. The noise draw is seeded on `seed` alone, so dose a is exactly a times the unit draw. |
| `reseed_runs_all.csv` | raw | 7,200 rows: fp16 regenerated at seeds 1000-1002, batch 16, left-padded. |
| `per_prompt_margins.csv` | derived | m0, g, kappa, linear-fit R^2 per (model, prompt). |

## Stated-decoding grids (`analysis/`)

| File | Kind | Contents |
|---|---|---|
| `regrid_generations.csv`, `regrid_generations_gguf.csv`, `regrid_meta*.json` | raw | Falcon3, 3,200 generations, `do_sample=False` / `temperature=0.0` |
| `p21_generations_*.csv`, `p21_meta_*.json` | raw | Llama-3.2 and Qwen2.5, 6,400 generations, stated decoding |
| `judge_labels_regrid.csv`, `judge_labels_phase21.csv`, `judge_meta_*.json` | raw | judge labels for the two grids |
| `regrid_pairs.csv`, `phase21_pairs.csv` | derived | fp16-versus-quantized pairs with labels and m0. **The paper's flip data.** |
| `regrid_summary.json`, `phase21_analysis.json` | derived | flip rates, tier staircase, agreement with Study A |
| `regrid_diagnosis.{json,csv}`, `llama_awq_check.csv` | raw | empty-reply diagnosis (29 failures, 40 controls) |
| `decoding_audit.json` | raw* | decoding each checkpoint repository declares (`reproduce.py --network` refetches) |
| `decoding_confound.json` | derived | the format-ordering inversion |

## Mechanism (`analysis/`)

| File | Kind | Contents |
|---|---|---|
| `alpha_equiv_tensors.csv`, `mechanism_meta.json` | raw | measured per-tensor quantization error, Falcon3, by direct dequantization |
| `delta_m_prompts.csv` | raw | margins on each quantized Falcon3 checkpoint |
| `qwen_mechanism/` | raw | the same for Qwen2.5 (AWQ, GPTQ; GGUF failed its coherence probe and is excluded) |
| `delta_m_joined.csv`, `mechanism_analysis.json`, `mechanism_verdict.json` | derived | variance model and zero-crossing test |
| `dose_response_recheck.json` | derived | zero-crossing miss against the stated-decoding flip rates. Its stored Spearman p-value is not used: the unit is the format, n = 3. |
| `studyb_baseline.json`, `dose_response_jailbreak.csv` | derived | the pooled synthetic control |
| `s5_jailbreak_tstats.json` | derived | per-cell mean shift; t-values treat prompts as independent, see `delta_m_tstats.py` |
| `threshold.json` | derived | fp16 behavioural boundary and the re-aimed criterion |
| `awq_anomaly.json` | derived | why AWQ's error is reported on `o_proj` |
| `qwen_mechanism.json` | derived | Qwen2.5 variance model |

## Index, controls, validation (`analysis/`)

| File | Kind | Contents |
|---|---|---|
| `token_ablation.csv`, `token_ablation_meta*.json` | raw | m0 recomputed with alternative refusal tokens |
| `continuation_tokens.json` | raw* | first token of each refusal and compliance continuation, per tokenizer |
| `token_ablation.json` | derived | pre-registered ablation verdict (fails on Llama-3.2, as reported) |
| `index_stats.json` | derived | per-model AUC with base-request-clustered intervals |
| `review_audit.json` | derived | first-token breaks, post-quantization margin, baselines, precision/recall, sampling flips, single-draw control |
| `paper_tables.json` | derived | Table 2, the pooled m0-vs-delta_m comparison, tier flips on judge labels |
| `human_review_flip_pairs.csv`, `human_review_flip_pairs_SCORED.csv` | raw | all 267 judge-flagged flips, blind human scores |
| `human_review.json` | derived | agreement, kappa and its caveats |
| `judge_labels.csv`, `judge_meta.json` | raw | re-seed control labels and judge self-consistency |
| `numerical_null*.{json,csv}`, `null_is_it_real.json`, `null_text_divergence.csv` | derived | re-seed control; only 93 of 1,791 comparisons changed the text |
| `forensics.json`, `repro_empty_rows.csv`, `empty_reproducibility.json` | derived | empty and degenerate replies |

`*` fetched from the Hugging Face Hub rather than measured on a GPU.
