"""
reseed_driver.py
================
E4 generation: repeat the fp16 runs at several sampling seeds to measure the
EMPIRICAL NULL -- how much does ASR move between two runs that differ only by
sampling randomness, with no quantization involved at all?

Why this is needed
------------------
model_loaders.generate_reply() passes no sampling arguments, so the original
runs used each model's generation_config defaults -- and Qwen2.5-Instruct and
Llama-3.2-Instruct both ship do_sample=true. Every row in results.csv is
therefore a single stochastic draw. Without this control, a reviewer can argue
that the measured quantization deltas are just sampling noise, and the only
answer available is an analytic binomial model.

This produces the empirical distribution instead: the spread across these
fp16-only passes is the null that any claimed quantization effect must clear.

Design note on batching
-----------------------
Generation is batched with LEFT padding (required for decoder-only models);
the original ran batch=1. Batched and unbatched sampling can differ slightly in
numerics, so the valid comparison here is between THESE passes -- all produced
identically -- rather than against the original fp16 rows. That keeps the null
internally consistent, and the caveat is reported rather than hidden.

Writes /kaggle/working/reseed_runs.csv (same schema as results.csv plus `seed`).
"""

import json
import os
import sys
import time
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIG: build_reseed_kernel.py rewrites this block --------------------
CONFIG = {
    "models": ["qwen2.5-1.5b"],
    "seeds": [1000, 1001],
    "limit_prompts": 20,
    "max_new_tokens": 100,
    "batch_size": 16,
    "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
    "out_dir": "/kaggle/working",
}
# --- END CONFIG ------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_hf_token():
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            log("HF_TOKEN loaded from Kaggle Secrets.")
            return True
    except Exception as exc:
        log(f"No HF_TOKEN ({type(exc).__name__}); gated models will be skipped.")
    return False


def resolve_dataset_dir(preferred):
    if os.path.exists(os.path.join(preferred, "attack_prompts.csv")):
        return preferred
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "attack_prompts.csv" in files:
            log(f"Resolved dataset dir to {root}")
            return root
    raise FileNotFoundError("attack_prompts.csv not found under /kaggle/input")


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Batched chat generation, matching generate_reply()'s settings.

    No sampling arguments are passed, exactly as model_loaders.generate_reply
    does, so the model's own generation_config governs -- which is the whole
    point: we are measuring the variance that configuration produces.
    """
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False,
        )
        for p in prompts
    ]
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"          # required for decoder-only generation
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    tokenizer.padding_side = prev_side

    out = model.generate(**enc, max_new_tokens=max_new_tokens)
    gen = out[:, enc["input_ids"].shape[-1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def run_model(model_key, registry, prompts, cfg, out_path):
    entry = registry.MODELS[model_key]["fp16"]
    family = registry.MODELS[model_key]["family"]
    size = registry.MODELS[model_key]["size"]
    log(f"=== {model_key} ({entry['repo']}) ===")

    tokenizer = AutoTokenizer.from_pretrained(entry["repo"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        entry["repo"], dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    log(f"  generation_config: do_sample={model.generation_config.do_sample} "
        f"temperature={model.generation_config.temperature} "
        f"top_p={model.generation_config.top_p}")

    bs = cfg["batch_size"]
    for seed in cfg["seeds"]:
        t0 = time.time()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rows = []
        for start in range(0, len(prompts), bs):
            chunk = prompts.iloc[start:start + bs]
            replies = generate_batch(model, tokenizer,
                                     chunk["prompt_text"].tolist(),
                                     cfg["max_new_tokens"])
            for (_, r), reply in zip(chunk.iterrows(), replies):
                rows.append({
                    "family": family, "size": size, "method": "fp16",
                    "seed": seed,
                    "attack_category": r["attack_category"],
                    "template_tier": r["template_tier"],
                    "prompt_id": r["prompt_id"],
                    "prompt_text": r["prompt_text"],
                    "attacker_instruction": r.get("attacker_instruction", ""),
                    "model_reply": reply,
                    "success": "",
                })
        pd.DataFrame(rows).to_csv(
            out_path, mode="a" if os.path.exists(out_path) else "w",
            header=not os.path.exists(out_path), index=False,
        )
        log(f"  seed={seed}: {len(rows)} generations ({time.time()-t0:.0f}s)")

    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device.")
    if torch.cuda.get_device_capability(0) < (7, 0):
        raise RuntimeError("GPU below sm_70 -- re-push with --accelerator NvidiaTeslaT4.")
    log(f"gpu={torch.cuda.get_device_name(0)}")

    have_token = setup_hf_token()
    dataset_dir = resolve_dataset_dir(CONFIG["dataset_dir"])
    sys.path.insert(0, dataset_dir)
    import model_registry

    prompts = pd.read_csv(os.path.join(dataset_dir, "attack_prompts.csv"))
    if CONFIG["limit_prompts"]:
        prompts = (prompts.groupby("template_tier", group_keys=False)
                   .apply(lambda g: g.head(max(1, CONFIG["limit_prompts"] //
                                               prompts.template_tier.nunique())))
                   .reset_index(drop=True))
    log(f"{len(prompts)} prompts x {len(CONFIG['seeds'])} seeds x "
        f"{len(CONFIG['models'])} models")

    out_path = os.path.join(CONFIG["out_dir"], "reseed_runs.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    manifest = {"config": CONFIG, "models": {}, "started": time.time()}
    for model_key in CONFIG["models"]:
        try:
            if model_registry.MODELS[model_key]["family"].startswith("Llama") \
                    and not have_token:
                log(f"SKIP {model_key}: gated, no HF_TOKEN")
                manifest["models"][model_key] = {"status": "skipped_gated"}
                continue
            run_model(model_key, model_registry, prompts, CONFIG, out_path)
            manifest["models"][model_key] = {"status": "ok"}
        except Exception as exc:
            log(f"FAILED {model_key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            manifest["models"][model_key] = {"status": "failed", "error": str(exc)}

    manifest["elapsed_min"] = (time.time() - manifest["started"]) / 60
    with open(os.path.join(CONFIG["out_dir"], "reseed_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"DONE in {manifest['elapsed_min']:.1f} min")

    if os.path.exists(out_path):
        d = pd.read_csv(out_path)
        log(f"{len(d)} rows written")
        log("\n" + d.groupby(["family", "size", "seed"]).size().to_string())


if __name__ == "__main__":
    main()
