"""
judge_driver.py
===============
Kaggle kernel: re-label replies with the SAME LLM judge and the SAME rubrics that
produced the primary `success` labels in results.csv.

Why this runs on Kaggle rather than locally: the judge is reached through Hugging
Face Inference Providers, and the only credential available to this project lives
in Kaggle Secrets. No GPU is needed -- this kernel is CPU-only.

Fidelity to the original labelling pass (notebooks/04_classify.ipynb) is the whole
point of this file, so it is deliberately not "improved":

  * both rubric strings are injected by build_judge.py, extracted from the notebook
    via ast.literal_eval so the RUNTIME string (line continuations resolved) is
    byte-identical, and asserted here against a SHA-256 baked in at build time;
  * same model, same base_url, temperature=0, max_tokens=350, reasoning disabled,
    same 3-attempt retry with a 2 s sleep, same JSON contract;
  * a missing/empty reply is rendered the way pandas + f-string rendered it in the
    original pass (the literal 'nan'), because that is what the judge actually saw
    for the 32 empty-output rows. Faithful, not tidy -- changing it would make the
    new labels incomparable to the old ones.

Input : a Kaggle dataset containing one or more judge_queue*.csv files with columns
        uid, attack_category, attacker_instruction, model_reply  (+ any passthrough).
Output: /kaggle/working/judge_labels.csv  (uid, label, queue_file, queue_id)
        /kaggle/working/judge_meta.json
"""

import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG: build_judge.py rewrites this block ----------------------------
CONFIG = {
    "judge_model": "deepseek-ai/DeepSeek-V4-Flash",
    "base_url": "https://router.huggingface.co/v1",
    "max_workers": 8,
    "max_tokens": 350,
    "temperature": 0,
    "retries": 3,
    "retry_sleep": 2,
    "checkpoint_every": 250,
    "out_dir": "/kaggle/working",
}
# --- end CONFIG ------------------------------------------------------------

# --- RUBRICS: build_judge.py rewrites this block ---------------------------
RUBRIC_JAILBREAK = ""
RUBRIC_INJECTION = ""
RUBRIC_SHA = {"jailbreak": "", "injection": ""}
# --- end RUBRICS -----------------------------------------------------------

VALID_LABELS = {"0", "1", "error", "ambiguous"}


def log(msg):
    print(msg, flush=True)


def assert_rubric_fidelity():
    """A drifted rubric must fail loudly, not silently relabel the dataset."""
    for key, rubric in (("jailbreak", RUBRIC_JAILBREAK), ("injection", RUBRIC_INJECTION)):
        got = hashlib.sha256(rubric.encode("utf-8")).hexdigest()
        want = RUBRIC_SHA[key]
        if got != want:
            raise SystemExit(
                f"RUBRIC MISMATCH for {key}: sha256={got} expected={want}. "
                "Refusing to label with a rubric that is not the one that produced "
                "the primary labels."
            )
        log(f"rubric[{key}] ok: len={len(rubric)} sha256={got[:16]}")


def get_token():
    for env in ("JUDGE_HF_TOKEN", "HF_TOKEN"):
        if os.environ.get(env):
            log(f"token from environment: {env}")
            return os.environ[env]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception as exc:
        raise SystemExit(f"No Kaggle Secrets client available: {type(exc).__name__}: {exc}")
    for name in ("JUDGE_HF_TOKEN", "HF_TOKEN"):
        try:
            tok = client.get_secret(name)
        except Exception:
            tok = None
        if tok:
            log(f"token from Kaggle Secrets: {name}")
            return tok
    raise SystemExit(
        "No JUDGE_HF_TOKEN or HF_TOKEN secret is attached to this kernel. "
        "Attach it in the notebook editor (Add-ons -> Secrets), then Save & Run All."
    )


def find_queues():
    """Every judge_queue*.csv under /kaggle/input, found by tree walk.

    The dataset mount path differs between kernel types, so walking is the only
    reliable way (same lesson as the margin-sweep driver).
    """
    hits = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for fn in sorted(files):
            if fn.startswith("judge_queue") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit("No judge_queue*.csv found under /kaggle/input.")
    return sorted(hits)


def render_reply(value):
    """Reproduce exactly what the original f-string put in front of the judge."""
    if value is None:
        return "nan"
    text = str(value)
    if text.strip() == "" or text == "nan":
        return "nan"
    return text


def build_messages(row):
    if row["attack_category"] == "jailbreak":
        return RUBRIC_JAILBREAK, f"REPLY:\n{render_reply(row.get('model_reply'))}"
    instruction = row.get("attacker_instruction")
    instruction = "nan" if instruction is None or str(instruction).strip() == "" else str(instruction)
    return (
        RUBRIC_INJECTION,
        f"HIDDEN INSTRUCTION:\n{instruction}\n\nREPLY:\n{render_reply(row.get('model_reply'))}",
    )


def classify(client, row):
    system_prompt, user_content = build_messages(row)
    last_err = None
    for attempt in range(CONFIG["retries"]):
        try:
            completion = client.chat.completions.create(
                model=CONFIG["judge_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
                extra_body={"reasoning": {"enabled": False}},
            )
            text = completion.choices[0].message.content.strip()
            label = str(json.loads(text)["label"])
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected label: {label}")
            return row["uid"], label, ""
        except Exception as exc:  # noqa: BLE001 - identical policy to the original pass
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt == CONFIG["retries"] - 1:
                log(f"FAILED after {CONFIG['retries']} tries: {row['uid']} -- {last_err}")
                return row["uid"], None, last_err
            time.sleep(CONFIG["retry_sleep"])
    return row["uid"], None, last_err


def main():
    import pandas as pd
    from openai import OpenAI

    assert_rubric_fidelity()
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    client = OpenAI(base_url=CONFIG["base_url"], api_key=get_token())

    queues = find_queues()
    log(f"queue files: {queues}")

    frames = []
    for path in queues:
        q = pd.read_csv(path, dtype=str)
        missing = {"uid", "attack_category"} - set(q.columns)
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        q["queue_file"] = os.path.basename(path)
        q["queue_id"] = q["queue_id"] if "queue_id" in q.columns else ""
        log(f"  {os.path.basename(path)}: {len(q)} rows, "
            f"queue_id={q['queue_id'].iloc[0] if len(q) else 'n/a'}, "
            f"categories={q['attack_category'].value_counts().to_dict()}")
        frames.append(q)
    queue = pd.concat(frames, ignore_index=True)

    dupes = queue["uid"].duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate uids in the queue -- refusing to run.")
    log(f"total rows to judge: {len(queue)}")

    out_path = os.path.join(CONFIG["out_dir"], "judge_labels.csv")
    results = {}
    errors = {}
    rows = queue.to_dict("records")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
        futures = [pool.submit(classify, client, r) for r in rows]
        for fut in as_completed(futures):
            uid, label, err = fut.result()
            results[uid] = label
            if err:
                errors[uid] = err
            done += 1
            if done % CONFIG["checkpoint_every"] == 0:
                log(f"{done}/{len(rows)} judged ({time.time()-t0:.0f}s)")
                partial = queue.copy()
                partial["label"] = partial["uid"].map(results)
                partial.to_csv(out_path, index=False)

    queue["label"] = queue["uid"].map(results)
    queue["judge_error"] = queue["uid"].map(errors).fillna("")
    queue.to_csv(out_path, index=False)

    n_failed = int(queue["label"].isna().sum())
    log(f"\ndone in {time.time()-t0:.0f}s: {len(queue)-n_failed} labelled, {n_failed} failed")
    log("label distribution:")
    log(queue["label"].value_counts(dropna=False).to_string())
    if "source" in queue.columns:
        log("\nby source:")
        log(queue.groupby("source")["label"].value_counts(dropna=False).to_string())

    meta = {
        "judge_model": CONFIG["judge_model"],
        "rubric_sha": RUBRIC_SHA,
        "queue_files": [os.path.basename(p) for p in queues],
        "queue_ids": sorted(set(queue["queue_id"].dropna().astype(str))),
        "n_rows": int(len(queue)),
        "n_failed": n_failed,
        "label_counts": {str(k): int(v) for k, v in
                         queue["label"].value_counts(dropna=False).items()},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(CONFIG["out_dir"], "judge_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log("\nwrote judge_labels.csv + judge_meta.json")

    if n_failed:
        # Loud, but do not throw away the rows that did succeed.
        log(f"WARNING: {n_failed} rows have no label. Re-run to retry only those.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        log(f"ABORT: {exc}")
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
