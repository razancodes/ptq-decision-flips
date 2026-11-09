"""
token_ablation_driver.py
========================
Is the First-Token Fragility Index a property of the refusal boundary, or of the
token "I"?

m_first is the metric every reported number uses, and on all six models its
refusal side collapses to a single token: the three refusal continuations share
the first token "I", so the refusal half of the contrast is one logit averaged
three times (asserted by experiments/presubmission/continuation_token_check.py).
The compliance half does give three distinct tokens. So the index is

    logit("I") - mean(logit("Sure"), logit("Certainly"), logit("Of"))

That is defensible on the evidence -- 99.3% of judged refusals in the repaired
grid really do open with "I" -- but it is not what the code comment or the
paper's measures table claims, and "I" is not refusal-specific. This kernel
re-measures the index with ALT_REFUSAL_CONTINUATIONS, whose three first tokens
are distinct ("Sorry", "Unfortunately", "I"), on the same prompts and the same
fp16 checkpoints.

There is no perturbation here and no generation: this is the alpha = 0 margin
only, twice per prompt. It is far cheaper than the sweep.

The question the analysis then answers, on the stated-decoding grid:

  * Spearman(m_first_default, m_first_alt) across prompts. High means the token
    choice is a labelling issue and the index is measuring the boundary.
  * AUC of the alt index predicting breaks within the fp16-refused stratum,
    against the published 0.952 / 0.949 / 0.902. If it holds, the single-token
    finding is a documentation defect. If it moves, the index is an artifact of
    one token and the paper's central quantity needs re-specifying.

Not run directly -- build_token_ablation.py concatenates margin_core.py with
this file into one self-contained Kaggle kernel script. The helper functions
below are verbatim copies from margin_sweep_driver.py, which is the established
pattern in this repository: Kaggle kernels are a single file and each driver
carries its own copy rather than importing across them.

Reads   /kaggle/input/<DATASET>/attack_prompts.csv
        /kaggle/input/<DATASET>/model_registry.py
Writes  /kaggle/working/token_ablation.csv        (incremental, crash-safe)
        /kaggle/working/token_ablation_meta.json  (run manifest + token ids)
"""

import json
import os
import sys
import time
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIG: build_token_ablation.py rewrites this block -------------------
CONFIG = {
    "models": ["qwen2.5-1.5b"],
    "limit_prompts": 20,
    "jailbreak_only": True,
    "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
    "out_dir": "/kaggle/working",
}
# --- END CONFIG ------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_hf_token():
    """Llama-3.2 repos are gated. Pull the token from Kaggle Secrets if present.

    Attaching a secret in Kaggle's UI does NOT export it as an environment
    variable; the kernel has to fetch it and set HF_TOKEN itself. A 401 from an
    unread secret is indistinguishable from a 401 from no secret at all, which
    is why this is explicit rather than assumed.

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
    """Find the attached dataset, whatever Kaggle named its mount point."""
    if os.path.exists(os.path.join(preferred, "attack_prompts.csv")):
        return preferred
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
        + "\n".join(listing))


def load_prompts(dataset_dir, limit=None, jailbreak_only=True):
    df = pd.read_csv(os.path.join(dataset_dir, "attack_prompts.csv"))
    if jailbreak_only:
        # The paper's scope. Injection was collected but is not reported, and
        # refusal-versus-compliance is the wrong outcome axis for it.
        df = df[df.attack_category == "jailbreak"].reset_index(drop=True)
    if limit:
        df = (df.groupby("template_tier", group_keys=False)
                .apply(lambda g: g.head(max(1, limit // df.template_tier.nunique())))
                .reset_index(drop=True))
    log(f"{len(df)} prompts: " +
        ", ".join(f"{k}={v}" for k, v in df.template_tier.value_counts().items()))
    return df


def ablate_model(model_key, registry, prompts, device, out_path):
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

    # Record what each set's first tokens actually are on THIS tokenizer, so the
    # kernel's own output evidences the premise rather than relying on the
    # offline check having been run against the same revision.
    ids_default = first_token_ids(tokenizer, REFUSAL_CONTINUATIONS)   # noqa: F821
    ids_alt = first_token_ids(tokenizer, ALT_REFUSAL_CONTINUATIONS)   # noqa: F821
    meta = {
        "repo": repo,
        "default_refusal_first_ids": ids_default,
        "default_refusal_first_tokens": tokenizer.convert_ids_to_tokens(ids_default),
        "default_refusal_n_distinct": len(set(ids_default)),
        "alt_refusal_first_ids": ids_alt,
        "alt_refusal_first_tokens": tokenizer.convert_ids_to_tokens(ids_alt),
        "alt_refusal_n_distinct": len(set(ids_alt)),
    }
    log(f"  default refusal first tokens {meta['default_refusal_first_tokens']} "
        f"({meta['default_refusal_n_distinct']} distinct)")
    log(f"  alt     refusal first tokens {meta['alt_refusal_first_tokens']} "
        f"({meta['alt_refusal_n_distinct']} distinct)")
    if meta["alt_refusal_n_distinct"] < 3:
        raise RuntimeError(
            f"The alt refusal set collapses to {meta['alt_refusal_n_distinct']} "
            "distinct first tokens on this tokenizer, so it does not ablate the "
            "token choice. Aborting rather than recording a null comparison.")

    rows = []
    t0 = time.time()
    for _, row in prompts.iterrows():
        text = row["prompt_text"]
        # Per-continuation values, not just the two margins, because the whole
        # point is to see what the contrast is made of.
        rn_d, cn_d, rf_d, cf_d = score_continuations(model, tokenizer, text, device)   # noqa: F821
        rn_a, cn_a, rf_a, cf_a = score_continuations(                                  # noqa: F821
            model, tokenizer, text, device,
            refusals=ALT_REFUSAL_CONTINUATIONS,                                        # noqa: F821
            compliances=ALT_COMPLIANCE_CONTINUATIONS)                                  # noqa: F821
        ids, toks, lps = first_token_topk(model, tokenizer, text, device, k=5)         # noqa: F821

        rec = {
            "model_key": model_key, "family": family, "size": size,
            "prompt_id": row["prompt_id"],
            "attack_category": row["attack_category"],
            "template_tier": row["template_tier"],
            "m_first_default": sum(rf_d) / len(rf_d) - sum(cf_d) / len(cf_d),
            "m_first_alt": sum(rf_a) / len(rf_a) - sum(cf_a) / len(cf_a),
            "m_seq_default": sum(rn_d) / len(rn_d) - sum(cn_d) / len(cn_d),
            "m_seq_alt": sum(rn_a) / len(rn_a) - sum(cn_a) / len(cn_a),
            # The index averages the compliance side, but a greedy argmax
            # switches at its MAX. This gap is the size of that discrepancy and
            # is the quantity the apparent non-zero threshold may just be.
            "compliance_max_minus_mean": max(cf_d) - sum(cf_d) / len(cf_d),
            "greedy_top1_token": toks[0], "greedy_top1_logprob": lps[0],
            "greedy_top2_token": toks[1], "greedy_top2_logprob": lps[1],
            "greedy_topk": json.dumps(list(zip(toks, [round(v, 4) for v in lps]))),
        }
        for i, v in enumerate(rf_d):
            rec[f"ref_first_lp_{i}"] = v
        for i, v in enumerate(cf_d):
            rec[f"com_first_lp_{i}"] = v
        for i, v in enumerate(rf_a):
            rec[f"alt_ref_first_lp_{i}"] = v
        rows.append(rec)
    log(f"  {len(rows)} prompts in {time.time()-t0:.0f}s")

    pd.DataFrame(rows).to_csv(
        out_path, mode="a" if os.path.exists(out_path) else "w",
        header=not os.path.exists(out_path), index=False)

    del model, tokenizer
    free_memory()                                   # noqa: F821
    return meta


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    if device != "cuda":
        raise RuntimeError("No CUDA device -- enable a GPU accelerator.")
    log(f"gpu={torch.cuda.get_device_name(0)}")
    # Verbatim from margin_sweep_driver.py. Kaggle hands out P100s (sm_60) and
    # the current torch build has no kernels for them; without this the run
    # burns a session and every model comes back as
    # "CUDA error: no kernel image is available for execution on the device",
    # which is what the first push of this kernel did.
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0):
        raise RuntimeError(
            f"GPU compute capability {capability} is below the sm_70 minimum this "
            "PyTorch build supports (a P100 is sm_60). Re-push with "
            "--accelerator NvidiaTeslaT4; running here would fail on the first "
            "matmul or, worse, return garbage."
        )

    setup_hf_token()
    dataset_dir = resolve_dataset_dir(CONFIG["dataset_dir"])
    sys.path.insert(0, dataset_dir)
    import model_registry as registry

    prompts = load_prompts(dataset_dir, CONFIG["limit_prompts"],
                           CONFIG["jailbreak_only"])
    out_path = os.path.join(CONFIG["out_dir"], "token_ablation.csv")

    meta = {"config": CONFIG, "models": {}, "skipped": {},
            "torch": torch.__version__}
    for key in CONFIG["models"]:
        try:
            meta["models"][key] = ablate_model(key, registry, prompts, device,
                                               out_path)
        except Exception as exc:
            # One gated or OOM model must not cost the whole session.
            log(f"  SKIPPED {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            meta["skipped"][key] = f"{type(exc).__name__}: {exc}"

    meta["n_models_done"] = len(meta["models"])
    with open(os.path.join(CONFIG["out_dir"], "token_ablation_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"done: {meta['n_models_done']} models, {len(meta['skipped'])} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
