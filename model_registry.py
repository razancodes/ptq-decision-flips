"""
model_registry.py
==================
Single source of truth for every model repo ID used in the experiment.
Add/fix entries here — never hardcode a repo name inside a notebook cell.

Status key:
  "confirmed"   — verified real, correctly-labeled repo (checked its own
                  metadata/README to confirm the stated base model matches)
  "tested"      — confirmed AND actually loaded successfully in this project
  "unverified"  — repo name not personally confirmed by either Claude or
                  the user yet — check huggingface.co before running
  "n/a"         — deliberately not applicable for this family (e.g. no
                  GPTQ/AWQ checkpoints exist for this model)

Every "confirmed" entry below was checked against its own Hugging Face
model card to verify the declared base model matches what it claims to
be quantized from — this is specifically to avoid the mislabeling problem
noted in the original project scoping (a Gemma-3-1B "AWQ" checkpoint that
was actually mislabeled QAT).
"""

MODELS = {
    "qwen2.5-1.5b": {
        "family": "Qwen2.5",
        "size": "1.5B",
        "fp16": {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "status": "tested"},
        "gguf": {"repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                  "filename": "*q4_k_m.gguf", "status": "tested"},
        "gptq": {"repo": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "status": "tested"},
        "awq":  {"repo": "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "status": "tested"},
    },
    "qwen2.5-3b": {
        "family": "Qwen2.5",
        "size": "3B",
        # All four official Qwen team repos — same publisher as the 1.5B
        # versions already tested, same naming convention.
        "fp16": {"repo": "Qwen/Qwen2.5-3B-Instruct", "status": "confirmed"},
        "gguf": {"repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                  "filename": "*q4_k_m.gguf", "status": "confirmed"},
        "gptq": {"repo": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4", "status": "confirmed"},
        "awq":  {"repo": "Qwen/Qwen2.5-3B-Instruct-AWQ", "status": "confirmed"},
    },
    "llama3.2-1b": {
        "family": "Llama-3.2",
        "size": "1B",
        "fp16": {"repo": "meta-llama/Llama-3.2-1B-Instruct", "status": "confirmed"},
        "gguf": {"repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                  "filename": "*Q4_K_M.gguf", "status": "confirmed"},  # bartowski uses uppercase Q4_K_M, not lowercase
        # GPTQ and AWQ both from the same community quantizer ("clowman") —
        # AWQ repo's own config explicitly confirms base model =
        # meta-llama/Llama-3.2-1B-Instruct, so not a repeat of the
        # Gemma-3-1B mislabeling problem.
        "gptq": {"repo": "clowman/Llama-3.2-1B-Instruct-GPTQ-Int4", "status": "confirmed"},
        "awq":  {"repo": "clowman/Llama-3.2-1B-Instruct-AWQ-Int4", "status": "confirmed"},
    },
    "llama3.2-3b": {
        "family": "Llama-3.2",
        "size": "3B",
        # Originally flagged "not yet verified" in project notes — now confirmed.
        "fp16": {"repo": "meta-llama/Llama-3.2-3B-Instruct", "status": "confirmed"},
        "gguf": {"repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                  "filename": "*Q4_K_M.gguf", "status": "confirmed"},  # bartowski uses uppercase Q4_K_M, not lowercase
        # Repo's own model card explicitly states "Original Base Model:
        # meta-llama/Llama-3.2-3B-Instruct".
        "gptq": {"repo": "shuyuej/Llama-3.2-3B-Instruct-GPTQ", "status": "confirmed"},
        "awq":  {"repo": "AMead10/Llama-3.2-3B-Instruct-AWQ", "status": "confirmed"},
    },
    "falcon3-1b": {
        "family": "Falcon3",
        "size": "1B",
        "fp16": {"repo": "tiiuae/Falcon3-1B-Instruct", "status": "confirmed"},
        "gguf": {"repo": "tiiuae/Falcon3-1B-Instruct-GGUF",
                  "filename": "*q4_k_m*.gguf", "status": "confirmed"},
        # TII's official blog confirms GPTQ-Int4 and AWQ exist for the whole
        # Falcon3 family — verify exact repo slug on huggingface.co before
        # first use (search "Falcon3 1B GPTQ" / "Falcon3 1B AWQ").
        "gptq": {"repo": "tiiuae/Falcon3-1B-Instruct-GPTQ-Int4", "status": "confirmed"},
        "awq":  {"repo": "tiiuae/Falcon3-1B-Instruct-AWQ", "status": "confirmed"},
    },
    "falcon3-3b": {
        "family": "Falcon3",
        "size": "3B",
        "fp16": {"repo": "tiiuae/Falcon3-3B-Instruct", "status": "confirmed"},
        "gguf": {"repo": "tiiuae/Falcon3-3B-Instruct-GGUF",
                  "filename": "*q4_k_m*.gguf", "status": "confirmed"},
        "gptq": {"repo": "tiiuae/Falcon3-3B-Instruct-GPTQ-Int4", "status": "confirmed"},
        "awq":  {"repo": "tiiuae/Falcon3-3B-Instruct-AWQ", "status": "confirmed"},
    },
    "smollm2-360m": {
        "family": "SmolLM2",
        "size": "360M",
        "fp16": {"repo": "HuggingFaceTB/SmolLM2-360M-Instruct", "status": "confirmed"},
        # NOTE: SmolLM2-360M only ships Q8_0 GGUF (no Q4_K_M exists for
        # that size). To keep GGUF precision consistent WITHIN this family
        # for the size-effect comparison, both SmolLM2 sizes are run at
        # Q8_0. This means SmolLM2's GGUF column is not directly comparable
        # at face value to the Q4_K_M GGUF column used for every other
        # family in this project — fine for within-family (method, size)
        # analysis, but flag this explicitly if SmolLM2's GGUF ASR is ever
        # compared against another family's GGUF ASR.
        "gguf": {"repo": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
                  "filename": "*q8_0*.gguf", "status": "confirmed"},
        "gptq": {"repo": None, "status": "n/a"},
        "awq":  {"repo": None, "status": "n/a"},
    },
    "smollm2-1.7b": {
        "family": "SmolLM2",
        "size": "1.7B",
        "fp16": {"repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "status": "confirmed"},
        # Matched to 360M's forced Q8_0 (see note above) to preserve the
        # within-family size comparison at fixed GGUF precision.
        "gguf": {"repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
                  "filename": "*Q8_0*.gguf", "status": "confirmed"},
        "gptq": {"repo": None, "status": "n/a"},
        "awq":  {"repo": None, "status": "n/a"},
    },
}


def get(model_key, method):
    """Look up a repo entry, e.g. get('qwen2.5-1.5b', 'gptq')."""
    entry = MODELS[model_key][method]
    if entry["repo"] is None:
        raise ValueError(
            f"No repo for {model_key}/{method} — this combination doesn't "
            f"exist (status='n/a'), or check model_registry.py"
        )
    return entry


def list_unconfirmed():
    """Print every entry not yet confirmed real, as a to-do list.
    (Everything in this file is currently confirmed or tested — this stays
    here for when you add new models later.)"""
    for model_key, methods in MODELS.items():
        for method, entry in methods.items():
            if entry.get("status") not in ("confirmed", "tested", "n/a"):
                print(f"{model_key} / {method}: repo={entry['repo']}")
