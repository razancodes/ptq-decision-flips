"""
model_loaders.py
=================
Shared model-loading, generation, and cleanup utilities for the
quantization-vs-injection-robustness experiments.

Import this from any notebook:

    import sys
    sys.path.append('/content/drive/MyDrive/slm_security_paper')
    import model_loaders
    import importlib
    importlib.reload(model_loaders)  # picks up latest saved edits

Every loader returns objects ready to pass into generate_reply() /
generate_reply_gguf(). Always call unload() when you're done with a
model before loading the next one, or you WILL run out of T4 VRAM
partway through a multi-model run.
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from llama_cpp import Llama


# ---------------------------------------------------------------------------
# Loaders — one per quantization format
# ---------------------------------------------------------------------------

def load_fp16(model_name):
    """Load an unquantized (fp16) baseline model. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    return model, tokenizer


def load_gguf(repo_id, filename_pattern="*q4_k_m.gguf"):
    """Load a GGUF model via llama.cpp. Returns a llama_cpp.Llama object
    (not a (model, tokenizer) pair — llama.cpp bundles both)."""
    llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename_pattern,
        n_gpu_layers=-1,
        verbose=False,
    )
    return llm


def load_gptq(model_name, backend="triton"):
    """Load a GPTQ-quantized model. Returns (model, tokenizer).

    backend='triton' avoids gptqmodel's default Marlin kernel, which
    requires an Ampere+ GPU (compute capability 8.0+) and fails to
    compile on the T4 (compute capability 7.5) with a CUDA compile
    error mentioning 'dequant'/'dequant_fp8_scales'. If you ever run
    this on an A100/newer GPU, backend='marlin' will likely be faster.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = GPTQConfig(bits=4, backend=backend)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
        quantization_config=quant_config,
    )
    return model, tokenizer


def load_awq(model_name):
    """Load an AWQ-quantized model. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_reply(model, tokenizer, user_message, max_new_tokens=100):
    """Generation helper for fp16 / GPTQ / AWQ (transformers-based) models."""
    messages = [{"role": "user", "content": user_message}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    # Explicitly drop references before returning, so nothing outside
    # this function can accidentally keep the activation/KV tensors alive.
    del inputs, outputs
    return response


def generate_reply_gguf(llm, user_message, max_tokens=100):
    """Generation helper for GGUF (llama.cpp) models — different call pattern."""
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": user_message}],
        max_tokens=max_tokens,
    )
    return output["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Cleanup — call this between every model load
# ---------------------------------------------------------------------------

def unload():
    """Aggressively free GPU memory. Call this AFTER deleting your model
    variables yourself in the notebook — e.g.:

        del model, tokenizer   # or: del llm
        model_loaders.unload()

    Why the del has to happen in your notebook, not inside this function:
    Python passes objects by reference. If you instead call
    unload(model, tokenizer), this function only deletes its own LOCAL
    copies of those names — your notebook's global `model`/`tokenizer`
    variables still exist and still point at the same GPU tensors, so
    nothing actually becomes eligible for garbage collection. The del
    has to happen in the same scope the variable was created in.
    """
    # Run collection twice — the first pass can surface objects whose
    # __del__ drops the last reference to something else.
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[unload] allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")

    if allocated > 0.5:
        print(
            "[unload] warning: allocated memory did not drop near 0. "
            "Make sure you ran 'del model, tokenizer' (or 'del llm') "
            "in your notebook BEFORE calling unload(). Also check for "
            "any cell whose last line evaluates to the model object "
            "(e.g. just 'model' alone on the last line) — IPython caches "
            "that in Out[], which is a separate lingering reference. "
            "Run: %reset_selective -f model|tokenizer|llm"
        )


def print_memory():
    """Quick memory check without unloading anything."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")
