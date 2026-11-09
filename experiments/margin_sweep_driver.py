"""
margin_sweep_driver.py
======================
E2 driver: refusal-margin dose-response under Gaussian weight noise.

Not run directly -- build_kernel.py concatenates margin_core.py with this file
to produce a single self-contained Kaggle kernel script. Kaggle kernels are one
file, and keeping margin_core.py out of the attached dataset means iterating on
the measurement code doesn't require re-versioning the dataset.

Reads   /kaggle/input/<DATASET>/attack_prompts.csv
        /kaggle/input/<DATASET>/model_registry.py
Writes  /kaggle/working/margin_sweep.csv          (incremental, crash-safe)
        /kaggle/working/margin_sweep_meta.json    (run manifest + integrity checks)
"""

import json
import os
import sys
import time
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIG: build_kernel.py rewrites this block ---------------------------
CONFIG = {
    "models": ["qwen2.5-1.5b"],
    "limit_prompts": 20,
    "alphas": [0.0, 0.01, 0.04],
    "seeds_per_alpha": 2,
    "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
    "out_dir": "/kaggle/working",
}
# --- END CONFIG ------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_hf_token():
    """Llama-3.2 repos are gated. Pull the token from Kaggle Secrets if present.

    Absence is not fatal: open models still run and gated ones are recorded as
    skipped, so a missing secret costs coverage rather than the whole run.
    """
    if os.environ.get("HF_TOKEN"):
        log("HF_TOKEN already in environment.")
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
        log(f"No HF_TOKEN secret available ({type(exc).__name__}). "
            "Gated models (Llama-3.2) will be skipped.")
    return False


def resolve_dataset_dir(preferred):
    """Find the attached dataset, whatever Kaggle named its mount point.

    The configured path is tried first, then every directory under
    /kaggle/input is searched for attack_prompts.csv. A dataset that is still
    processing at kernel-launch time simply isn't mounted, and the resulting
    ModuleNotFoundError three call-frames later is far harder to read than an
    explicit listing of what actually is mounted.
    """
    if os.path.exists(os.path.join(preferred, "attack_prompts.csv")):
        return preferred

    # Kaggle's mount layout is not stable across API versions: datasets have
    # appeared at /kaggle/input/<slug>/ and at
    # /kaggle/input/datasets/<owner>/<slug>/. Walk the tree rather than guessing
    # a depth, and log what was found either way.
    log(f"'{preferred}' has no attack_prompts.csv; walking /kaggle/input ...")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "attack_prompts.csv" in files:
            log(f"Resolved dataset dir to {root} (contents: {sorted(files)[:6]})")
            return root

    listing = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        listing.append(f"  {root} -> {sorted(files)[:4]}")
        if len(listing) > 25:
            break
    raise FileNotFoundError(
        "attack_prompts.csv not found anywhere under /kaggle/input.\n"
        + "\n".join(listing)
    )


def load_prompts(dataset_dir, limit=None):
    df = pd.read_csv(os.path.join(dataset_dir, "attack_prompts.csv"))
    if limit:
        # Stratify the smoke subset so every tier is represented -- taking a
        # head() slice would sample a single tier and prove nothing.
        df = (df.groupby("template_tier", group_keys=False)
                .apply(lambda g: g.head(max(1, limit // df.template_tier.nunique())))
                .reset_index(drop=True))
    log(f"{len(df)} prompts: " +
        ", ".join(f"{k}={v}" for k, v in df.template_tier.value_counts().items()))
    return df


def sweep_model(model_key, registry, prompts, configs, device, out_path):
    entry = registry.MODELS[model_key]["fp16"]
    repo = entry["repo"]
    family = registry.MODELS[model_key]["family"]
    size = registry.MODELS[model_key]["size"]
    log(f"=== {model_key} ({repo}) ===")

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.float16, device_map=device,
    )
    model.eval()

    targets = target_linear_weights(model)          # noqa: F821 (from margin_core)
    log(f"  {len(targets)} target tensors, "
        f"{sum(p.numel() for _, p in targets)/1e6:.0f}M params perturbed")

    snap = snapshot_weights(targets)                # noqa: F821
    rows = []
    integrity = {}

    for cfg_i, (alpha, seed) in enumerate(configs):
        t0 = time.time()
        realized = apply_gaussian_noise(targets, alpha, seed, device)   # noqa: F821
        if realized:
            worst = max(abs(r - alpha) / alpha for r in realized.values())
            if worst > 0.05:
                raise RuntimeError(
                    f"Noise scale off by {worst:.1%} at alpha={alpha} -- aborting "
                    "rather than recording mis-scaled perturbations."
                )

        for _, row in prompts.iterrows():
            m_seq, m_first = margin(model, tokenizer,             # noqa: F821
                                    row["prompt_text"], device)
            rows.append({
                "model_key": model_key, "family": family, "size": size,
                "prompt_id": row["prompt_id"],
                "attack_category": row["attack_category"],
                "template_tier": row["template_tier"],
                "alpha": alpha, "seed": seed,
                "m_seq": m_seq, "m_first": m_first,
            })

        restore_weights(targets, snap)              # noqa: F821
        log(f"  [{cfg_i+1}/{len(configs)}] alpha={alpha} seed={seed} "
            f"({time.time()-t0:.0f}s)")

        # Flush after every config so a session timeout still leaves usable data.
        pd.DataFrame(rows).to_csv(
            out_path, mode="a" if os.path.exists(out_path) else "w",
            header=not os.path.exists(out_path), index=False,
        )
        rows = []

    # Integrity: re-measure the unperturbed baseline AFTER the sweep. Any drift
    # means the restore path leaked and every number above is suspect.
    log("  verifying restore fidelity...")
    baseline_after = [margin(model, tokenizer, r["prompt_text"], device)[0]   # noqa: F821
                      for _, r in prompts.head(10).iterrows()]
    saved = pd.read_csv(out_path)
    baseline_before = (saved[(saved.model_key == model_key) & (saved.alpha == 0.0)]
                       .head(10)["m_seq"].tolist())
    drift = max(abs(a - b) for a, b in zip(baseline_after, baseline_before))
    integrity["restore_drift"] = drift
    integrity["restore_ok"] = bool(drift < 1e-3)
    log(f"  restore drift = {drift:.3e} "
        f"({'OK' if integrity['restore_ok'] else 'FAILED'})")

    del model, tokenizer, snap, targets
    free_memory()                                   # noqa: F821
    return integrity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    if device == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} "
            f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    if device != "cuda":
        raise RuntimeError("No CUDA device -- enable a GPU accelerator.")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0):
        raise RuntimeError(
            f"GPU compute capability {capability} is below the sm_70 minimum this "
            "PyTorch build supports (a P100 is sm_60). Re-push with "
            "--accelerator NvidiaTeslaT4; running here would fail on the first "
            "matmul or, worse, return garbage."
        )

    have_token = setup_hf_token()
    dataset_dir = resolve_dataset_dir(CONFIG["dataset_dir"])
    sys.path.insert(0, dataset_dir)
    import model_registry

    prompts = load_prompts(dataset_dir, CONFIG["limit_prompts"])
    configs = build_configs(CONFIG["alphas"], CONFIG["seeds_per_alpha"])  # noqa: F821
    log(f"{len(configs)} configs x {len(prompts)} prompts "
        f"= {len(configs)*len(prompts)} measurements per model")

    out_path = os.path.join(CONFIG["out_dir"], "margin_sweep.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    manifest = {"config": CONFIG, "models": {}, "started": time.time()}
    for model_key in CONFIG["models"]:
        try:
            gated = model_registry.MODELS[model_key]["family"].startswith("Llama")
            if gated and not have_token:
                log(f"SKIP {model_key}: gated repo, no HF_TOKEN")
                manifest["models"][model_key] = {"status": "skipped_gated"}
                continue
            integrity = sweep_model(model_key, model_registry, prompts,
                                    configs, device, out_path)
            manifest["models"][model_key] = {"status": "ok", **integrity}
        except Exception as exc:
            log(f"FAILED {model_key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            manifest["models"][model_key] = {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
            free_memory()                           # noqa: F821

    manifest["finished"] = time.time()
    manifest["elapsed_min"] = (manifest["finished"] - manifest["started"]) / 60
    with open(os.path.join(CONFIG["out_dir"], "margin_sweep_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"DONE in {manifest['elapsed_min']:.1f} min")
    for k, v in manifest["models"].items():
        log(f"  {k}: {v.get('status')} "
            f"{'drift=%.2e' % v['restore_drift'] if 'restore_drift' in v else ''}")

    if os.path.exists(out_path):
        final = pd.read_csv(out_path)
        log(f"{len(final)} rows written to margin_sweep.csv")
        log("\n" + final.groupby(["template_tier", "alpha"])["m_seq"]
                        .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
