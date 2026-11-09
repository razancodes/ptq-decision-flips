"""
build.py -- assemble the Hugging Face dataset from the committed results.

    python hf_dataset/build.py      # writes hf_dataset/data/pairs.parquet

One row per (model, quantization format, prompt) on the stated-decoding grids,
jailbreak prompts only, which is exactly the data the paper's flip numbers are
computed from. Asserts the paper's counts so a bad join cannot ship silently.
"""
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
R = os.path.join(REPO, "results", "analysis")
sys.path.insert(0, REPO)
from model_registry import MODELS  # noqa: E402

KEY = ["family", "size", "prompt_id"]


def main():
    pairs = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("regrid_pairs.csv", "phase21_pairs.csv")],
                      ignore_index=True)
    pairs = pairs[(pairs.attack_category == "jailbreak") & (pairs.method != "fp16")].copy()
    pairs["size"] = pairs["size"].astype(str)

    # fp16 replies come from the raw generations; the pair files carry only the fp16 label.
    gens = pd.concat([pd.read_csv(f) for f in [os.path.join(R, "regrid_generations.csv")]
                      + glob.glob(os.path.join(R, "p21_generations_*.csv"))], ignore_index=True)
    gens["size"] = gens["size"].astype(str)
    fp16 = gens[gens.method == "fp16"].drop_duplicates(KEY)[KEY + ["model_reply"]]
    pairs = pairs.merge(fp16.rename(columns={"model_reply": "fp16_reply"}), on=KEY, how="left", validate="m:1")

    # delta_m where it was measured: Falcon3 (all formats) and Qwen2.5 (AWQ, GPTQ).
    dmf = pd.read_csv(os.path.join(R, "delta_m_joined.csv"))[KEY + ["method", "delta_m"]]
    qd = pd.read_csv(os.path.join(R, "qwen_mechanism", "delta_m_prompts.csv"))
    qd["size"] = qd["size"].astype(str)
    q0 = qd[qd.method == "fp16"][["prompt_id", "size", "m_first"]].rename(columns={"m_first": "m_fp16"})
    qq = qd[qd.method.isin(["awq", "gptq"])].merge(q0, on=["prompt_id", "size"])
    qq["delta_m"] = qq.m_first - qq.m_fp16
    dm = pd.concat([dmf, qq[KEY + ["method", "delta_m"]]], ignore_index=True)
    dm["size"] = dm["size"].astype(str)
    pairs = pairs.merge(dm.drop_duplicates(KEY + ["method"]), on=KEY + ["method"], how="left", validate="1:1")

    # Blind human review of every judge-flagged flip.
    hr = pd.read_csv(os.path.join(R, "human_review_flip_pairs_SCORED.csv"))
    hr["size"] = hr["size"].astype(str)
    hr = hr.rename(columns={"quant_method": "method", "your_label_fp16": "human_label_fp16",
                            "your_label_quant": "human_label_quantized"})
    pairs = pairs.merge(hr[KEY + ["method", "human_label_fp16", "human_label_quantized"]],
                        on=KEY + ["method"], how="left", validate="1:1")

    commits = json.load(open(os.path.join(REPO, "checkpoints.json")))["checkpoints"]
    reg = {(v["family"], v["size"]): v for v in MODELS.values() if "family" in v}
    pairs["checkpoint"] = [reg[(f, s)][m]["repo"] for f, s, m in zip(pairs.family, pairs["size"], pairs.method)]
    pairs["checkpoint_commit"] = pairs.checkpoint.map(lambda r: commits[r]["commit"])

    out = pd.DataFrame({
        "family": pairs.family, "size": pairs["size"], "quant_format": pairs.method,
        "checkpoint": pairs.checkpoint, "checkpoint_commit": pairs.checkpoint_commit,
        "prompt_id": pairs.prompt_id, "base_request_id": pairs.prompt_id.str.rsplit("_", n=1).str[0],
        "framing": pairs.template_tier, "prompt": pairs.prompt_text,
        "fp16_reply": pairs.fp16_reply, "quantized_reply": pairs.model_reply,
        "fp16_complied": pairs.s_fp16.astype(int), "quantized_complied": pairs.y.astype(int),
        "flip": pairs.flip.astype(int), "break": pairs.broke.astype(int), "restore": pairs.restore.astype(int),
        "m0": pairs.m0, "delta_m": pairs.delta_m,
        "human_fp16_complied": pairs.human_label_fp16.astype("Int64"),
        "human_quantized_complied": pairs.human_label_quantized.astype("Int64"),
    }).sort_values(["family", "size", "quant_format", "prompt_id"]).reset_index(drop=True)

    # The paper's counts. If any of these move, the join is wrong.
    assert len(out) == 5386, len(out)
    assert out.flip.sum() == 267 and out["break"].sum() == 138, (out.flip.sum(), out["break"].sum())
    assert out.fp16_reply.notna().all() and out.m0.notna().all()
    assert out.human_quantized_complied.notna().sum() == 267 == out.loc[out.flip == 1, "human_quantized_complied"].notna().sum()
    assert out.checkpoint_commit.notna().sum() == len(out) - (out.checkpoint == "meta-llama/Llama-3.2-1B-Instruct").sum()

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out.to_parquet(os.path.join(HERE, "data", "pairs.parquet"), index=False)
    print(f"wrote {len(out)} rows; flips {out.flip.sum()}, breaks {out['break'].sum()}, "
          f"restores {out.restore.sum()}; delta_m on {out.delta_m.notna().sum()} rows")
    print(out.groupby(["family", "quant_format"]).flip.agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
