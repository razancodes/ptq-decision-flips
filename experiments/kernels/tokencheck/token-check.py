"""CPU-only probe: is HF_TOKEN reachable from a pushed kernel, and does it
actually grant access to the gated meta-llama repos? Costs no GPU quota."""
import os, sys, traceback

print("=== 1. Kaggle Secrets ===", flush=True)
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    print(f"OK: secret retrieved, length={len(token)}, prefix={token[:4]}...")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()

if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

print("\n=== 2. HF identity ===", flush=True)
try:
    from huggingface_hub import HfApi
    who = HfApi().whoami(token=token)
    print(f"OK: authenticated as {who.get('name')} ({who.get('type')})")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")

print("\n=== 3. Gated repo access (config only, no weights) ===", flush=True)
for repo in ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo, token=token)
        print(f"OK  {repo}: {cfg.model_type}, {cfg.num_hidden_layers} layers")
    except Exception as exc:
        print(f"FAILED {repo}: {type(exc).__name__}: {str(exc)[:200]}")

print("\n=== 4. Tokenizer (exercises the real download path) ===", flush=True)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=token)
    ids = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"): ids = ids["input_ids"]
    if hasattr(ids, "tolist"): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    print(f"OK: chat template -> {len(ids)} tokens")
    print("VERDICT: Llama run is safe to launch.")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {str(exc)[:300]}")
    print("VERDICT: do NOT launch the Llama run yet.")
