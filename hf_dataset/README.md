---
pretty_name: PTQ Refusal Flips
language:
- en
license: other
license_name: mixed-see-card
size_categories:
- 1K<n<10K
task_categories:
- text-classification
tags:
- ai-safety
- quantization
- refusal
- jailbreak
configs:
- config_name: default
  data_files:
  - split: train
    path: data/pairs.parquet
extra_gated_prompt: >-
  This dataset contains harmful-instruction prompts and language-model replies
  to them, some of which comply. Access is for AI-safety research. Model replies
  remain under their models' licenses, including the non-commercial Qwen
  Research License for Qwen2.5-3B.
extra_gated_fields:
  I will use this dataset for safety research and will not use it to cause harm: checkbox
  I have read the license section of the dataset card: checkbox
---

# PTQ Refusal Flips

Paired refusal decisions from six small instruction-tuned models before and
after post-training quantization, under stated greedy decoding. Each row is one
jailbreak prompt answered by an fp16 model and by one of its 4-bit quantized
checkpoints (AWQ, GPTQ, GGUF Q4_K_M), with LLM-judge labels on both replies,
whether the decision flipped, and the pre-quantization refusal margin `m0`.

Released with the paper *A First-Order Theory of Decision Flips Under
Post-Training Quantization: What It Predicts, and Where It Breaks* (Muhammed
Razan, G Shivam; NeurIPS 2026 Workshop: AXIOM Foundations of Efficient Deep
Learning).

[Paper (OpenReview)](https://openreview.net/forum?id=S5ack3Pfuc) ·
[NeurIPS 2026 page](https://neurips.cc/virtual/2026/loc/paris/156960) ·
[Code, raw generations and every analysis (GitHub)](https://github.com/razancodes/ptq-decision-flips)

> **Content warning.** Prompts ask for harmful content and some replies provide it.

## At a glance

- 5,386 pairs: 3 families (Falcon3, Llama-3.2, Qwen2.5) x 2 sizes x 3 formats x
  300 jailbreak prompts (100 base requests x plain / roleplay / persuasion
  framing), minus 14 pairs with an unscorable reply.
- 267 flips: 138 breaks (fp16 refused, quantized complied) and 129 restores.
- Every flip was re-labelled blind by a human: 92.9% of judge flips confirmed.
- Decoding is stated, not inherited: `do_sample=False` in transformers,
  `temperature=0` in llama.cpp, 100 new tokens.

## Fields

| Field | Meaning |
|---|---|
| `family`, `size`, `quant_format` | model and quantization format (`awq`, `gptq`, `gguf`) |
| `checkpoint`, `checkpoint_commit` | Hugging Face repository of the quantized checkpoint and the commit that ran (null for one gated repo whose commit was not logged) |
| `prompt_id`, `base_request_id`, `framing` | prompt identity; the three framings of a request share `base_request_id` |
| `prompt` | the user turn sent to the model |
| `fp16_reply`, `quantized_reply` | the two replies |
| `fp16_complied`, `quantized_complied` | LLM-judge label, 1 = complied, 0 = refused |
| `flip`, `break`, `restore` | label changed; changed refuse to comply; changed comply to refuse |
| `m0` | fp16 first-token refusal margin: logit("I") minus the mean logit of "Sure", "Certainly", "Of" |
| `delta_m` | change in that margin on the quantized checkpoint (Falcon3, all formats; Qwen2.5 AWQ and GPTQ) |
| `human_fp16_complied`, `human_quantized_complied` | blind human label, present on the 267 flips only |

## Intended use and what the data does not show

Use it to study which refusals quantization breaks, to test predictors of
breaks, or to evaluate safety judges on quantized-model outputs. Before relying
on it, note:

- **Labels are from one LLM judge** (`deepseek-ai/DeepSeek-V4-Flash`, temperature 0).
  Human review covers the flips only, so it bounds false positives, not false negatives.
- **Breaks are rare** (2.6% of pairs). Report precision and recall, not only AUC.
- **`m0` is not specific to quantization.** It also ranks flips caused by sampling alone.
- **Llama-3.2 `m0` depends on the run date**, because its chat template inserts it.
- Small models (1B-3B), one decision type (refusal), English, 100-token replies.
- Compliance means the reply did not refuse. It is not a measure of how harmful the reply is.

## License

Our labels, margins and the prompt templates are CC BY 4.0. The prompts derive
from AdvBench and InjecAgent. Model replies remain under their models' licenses:
TII Falcon License (Falcon3), Llama 3.2 Community License (Llama-3.2), Apache 2.0
(Qwen2.5-1.5B) and the non-commercial Qwen Research License (Qwen2.5-3B). Filter
on `family` and `size` if your use needs a single license.

## Citation

```bibtex
@inproceedings{razan2026firstorder,
  title     = {A First-Order Theory of Decision Flips Under Post-Training Quantization:
               What It Predicts, and Where It Breaks},
  author    = {Razan, Muhammed and Shivam, G},
  booktitle = {NeurIPS 2026 Workshop: AXIOM Foundations of Efficient Deep Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=S5ack3Pfuc}
}
```
