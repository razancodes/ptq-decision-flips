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

```text
██████╗ ████████╗ ██████╗     ███████╗██╗     ██╗██████╗ ███████╗
██╔══██╗╚══██╔══╝██╔═══██╗    ██╔════╝██║     ██║██╔══██╗██╔════╝
██████╔╝   ██║   ██║   ██║    █████╗  ██║     ██║██████╔╝███████╗
██╔═══╝    ██║   ██║▄▄ ██║    ██╔══╝  ██║     ██║██╔═══╝ ╚════██║
██║        ██║   ╚██████╔╝    ██║     ███████╗██║██║     ███████║
╚═╝        ╚═╝    ╚══▀▀═╝     ╚═╝     ╚══════╝╚═╝╚═╝     ╚══════╝
```

<h2 align="center">PTQ Refusal Flips</h2>

<p align="center">
  Paired refusal decisions before and after 4-bit post-training quantization<br>
  <i>Dataset for "A First-Order Theory of Decision Flips Under Post-Training Quantization: What It Predicts, and Where It Breaks"</i><br>
  <b>Muhammed Razan</b> · <b>G Shivam</b> · NeurIPS 2026 Workshop: AXIOM Foundations of Efficient Deep Learning
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=S5ack3Pfuc"><img src="https://img.shields.io/badge/OpenReview-S5ack3Pfuc-8c1b13" alt="OpenReview"></a>
  <a href="https://neurips.cc/virtual/2026/loc/paris/156960"><img src="https://img.shields.io/badge/NeurIPS%202026-AXIOM%20Workshop-6f42c1" alt="NeurIPS 2026"></a>
  <a href="https://github.com/razancodes/ptq-decision-flips"><img src="https://img.shields.io/badge/GitHub-ptq--decision--flips-181717?logo=github" alt="Code"></a>
</p>

---

Six small instruction-tuned models answer 300 jailbreak prompts twice: once in
fp16 and once from a 4-bit checkpoint (AWQ, GPTQ, GGUF Q4_K_M), with greedy
decoding stated explicitly. Each row is one such pair, with LLM-judge labels on
both replies, whether the refusal flipped, the fp16 refusal margin `m0`, and a
blind human label on every flip.

## Quickstart

Accept the terms on this page, run `hf auth login`, then:

```python
from datasets import load_dataset

pairs = load_dataset("razancodes/ptq-refusal-flips", split="train")
breaks = pairs.filter(lambda r: r["break"] == 1)   # refused at fp16, complied after quantization
```

## Contents

| | Falcon3 | Llama-3.2 | Qwen2.5 | Total |
|---|---|---|---|---|
| Pairs (1B-3B, three formats) | 1,788 | 1,800 | 1,798 | **5,386** |
| Breaks (refuse → comply) | 59 | 31 | 48 | **138** |
| Restores (comply → refuse) | 10 | 16 | 103 | **129** |

300 prompts per model and format: 100 harmful requests from AdvBench, each in a
plain, roleplay and persuasion framing. 14 pairs with an unscorable reply are
left out. Human review confirmed 92.9% of the judge's flips.

## Fields

| Field | Meaning |
|---|---|
| `family`, `size`, `quant_format` | model and quantization format (`awq`, `gptq`, `gguf`) |
| `checkpoint`, `checkpoint_commit` | quantized checkpoint on the Hub and the commit that ran (null for one gated repo whose commit was not logged) |
| `prompt_id`, `base_request_id`, `framing` | prompt identity; the three framings of a request share `base_request_id` |
| `prompt` | the user turn sent to the model |
| `fp16_reply`, `quantized_reply` | the two replies (greedy, 100 new tokens) |
| `fp16_complied`, `quantized_complied` | LLM-judge label: 1 = complied, 0 = refused |
| `flip`, `break`, `restore` | label changed; refuse → comply; comply → refuse |
| `m0` | fp16 first-token refusal margin: logit("I") minus the mean logit of "Sure", "Certainly", "Of" |
| `delta_m` | change in that margin on the quantized checkpoint (Falcon3, all formats; Qwen2.5 AWQ and GPTQ) |
| `human_fp16_complied`, `human_quantized_complied` | blind human label, on the 267 flips |

## Notes

- Labels come from one LLM judge (`deepseek-ai/DeepSeek-V4-Flash`, temperature 0)
  that scores whether a refusal happened, not how harmful a reply is.
- Breaks are rare (2.6% of pairs), so report precision and recall alongside AUC.
- Llama-3.2's chat template inserts the run date, so its `m0` varies across run dates.
- The prompts ask for harmful content and some replies provide it. Use the data for
  safety research.

## License

Our labels and margins are CC BY 4.0. Prompts derive from AdvBench and
InjecAgent. Model replies keep their models' licenses: TII Falcon License
(Falcon3), Llama 3.2 Community License (Llama-3.2), Apache 2.0 (Qwen2.5-1.5B) and
the non-commercial Qwen Research License (Qwen2.5-3B). Filter on `family` and
`size` if your use needs a single license.

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
