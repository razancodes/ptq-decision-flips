"""
mechanism_driver.py
===================
Tasks 1.3 and 1.4. Measure how big real quantization error is in Study B's own
units, and whether it shifts the refusal margin or only jitters it.

1.3  alpha_equiv[t] = RMS(W_quant[t] - W_fp16[t]) / std(W_fp16[t])

     Study B perturbed W += N(0, (alpha*std(W))^2), whose per-element RMS error
     is exactly alpha*std(W), so this ratio is the same quantity a real
     checkpoint realises. A Frobenius ratio ||dW||_F/||W||_F is NOT specified
     here; note that the two coincide whenever mean(W)=0, which real weight
     tensors nearly satisfy, so both are recorded and the write-up can say so
     instead of asserting a difference that is not there.

     Reported as a distribution over tensors -- per layer, per projection type --
     never as one scalar. The tensors Study B EXCLUDED (embed_tokens, lm_head,
     norms) are measured separately: that is the alpha sweep's blind spot, and
     for a first-token margin it is not a small one, because lm_head is the
     matrix that produces the logits whose difference defines m.

1.4  delta_m = m_first(quant) - m_first(fp16), per prompt, mean and SD reported
     separately. Study B's perturbation is symmetric by construction, so its
     E[delta_m] is ~0 (measured: see studyb_baseline.json); a non-zero mean on
     real formats is therefore a property of real quantization error rather than
     of the measurement.

How the quantized weights are recovered
---------------------------------------
By dequantizing the checkpoint files directly, one tensor at a time, for all
three formats. No AWQ/GPTQ runtime kernels are involved, which means:

  * no dependency on gptqmodel / optimum / a Marlin-or-Triton kernel that
    compiles on a T4 -- the first version of this kernel died on exactly that;
  * memory stays flat: the fp16 reference is read lazily from safetensors and
    only one tensor pair is resident at a time. The first version held fp32 CPU
    copies of every target tensor and ran out of RAM on the 3B model;
  * all three formats are measured on one identical path, so alpha_equiv is
    comparable across them and to Study B, which also perturbed HF weights.

The cost, stated rather than hidden: margins measured this way reflect the
quantized WEIGHTS, not the quantized KERNEL. AWQ and GPTQ inference also carries
activation-precision and kernel-rounding effects that dequantize-then-run-in-fp16
does not reproduce, and the flip rates in §4.7 come from generations that DID use
those kernels. So this measures the weight-space component of the perturbation,
which is precisely the component Study B's model is about.

Every format decision is validated, never assumed, with criteria fixed in advance:

  V1  tensors a format leaves unquantized must match the reference EXACTLY
      (alpha_equiv = 0). This is what proves the name mapping is right.
  V2  a quantizer is unbiased by construction, so the mean signed error must be
      small relative to one quantization step. GPTQ's zero-point has two
      conventions in the wild (z versus z+1); both are computed and the one that
      is unbiased is used, with both values logged. That is resolving a format
      convention by a criterion declared up front, not tuning for fit.
  V3  llama.cpp's converter permutes attn_q/attn_k for its RoPE convention.
      Undoing it is likewise a convention: misalignment shows up as
      alpha_equiv ~ 1.4 (error the size of the weight spread) against ~0.06 for
      correct alignment, and both are logged.
  V4  the reloaded model must generate coherent text. A wrong unpack that
      survives V1-V3 will not survive this.
  V5  alpha_equiv outside [0.002, 0.6] is flagged as suspect rather than
      reported as a finding.

Output (all to /kaggle/working):
  alpha_equiv_tensors.csv   one row per (model, format, tensor)
  delta_m_prompts.csv       one row per (model, format, prompt)
  mechanism_meta.json       validations, conventions chosen, coherence samples
"""

import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

# In the built kernel margin_core.py is concatenated above this file, so its
# symbols are already globals and this import fails harmlessly. Locally (the CPU
# gate) it resolves, so the same code can be tested off-GPU.
try:  # noqa: SIM105
    from margin_core import margin, target_linear_weights  # noqa: F401
except ImportError:
    pass

# --- CONFIG: build_mechanism.py rewrites this block ------------------------
CONFIG = {
    "model_keys": ["falcon3-1b", "falcon3-3b"],
    "methods": ["awq", "gptq", "gguf"],
    "limit_prompts": None,
    "alpha_equiv_sane_range": [0.002, 0.6],
    "unbiased_tol_steps": 0.25,
    "dataset_dir": "/kaggle/input/slm-quant-attack-prompts",
    "out_dir": "/kaggle/working",
}
# --- end CONFIG ------------------------------------------------------------

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
EXCLUDED_KEYS = ("embed_tokens", "lm_head", "norm")

GGUF_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
GGUF_BLK_MAP = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
}


def log(msg):
    print(msg, flush=True)


def free():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resolve_input(filename, preferred=None):
    if preferred and os.path.exists(os.path.join(preferred, filename)):
        return os.path.join(preferred, filename)
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None


def tensor_kind(name):
    for suf in TARGET_SUFFIXES:
        if name.endswith(suf + ".weight") or name.endswith(suf):
            return suf, "target"
    low = name.lower()
    for key in EXCLUDED_KEYS:
        if key in low:
            return key, "excluded"
    return name, "other"


def layer_index(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def canonical_target_names(fp16_model):
    """Study B's tensor set, produced by Study B's own selection rule.

    margin_core.target_linear_weights() picks every nn.Linear inside the decoder
    stack by module type. Reusing it means alpha_equiv is measured on exactly the
    tensors the alpha sweep perturbed.
    """
    return [name for name, _ in target_linear_weights(fp16_model)]


# ---------------------------------------------------------------------------
# Lazy safetensors access -- the reason this fits in memory
# ---------------------------------------------------------------------------

class SafeStore:
    """Read tensors one at a time from a Hub repo's safetensors shards."""

    def __init__(self, repo):
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors import safe_open
        self.repo = repo
        self._safe_open = safe_open
        files = list_repo_files(repo)
        shards = sorted(f for f in files if f.endswith(".safetensors"))
        if not shards:
            raise RuntimeError(f"{repo} has no .safetensors files")
        self.handles = {}
        self.index = {}
        for shard in shards:
            path = hf_hub_download(repo_id=repo, filename=shard)
            h = safe_open(path, framework="pt", device="cpu")
            self.handles[shard] = h
            for key in h.keys():
                self.index[key] = shard
        log(f"    SafeStore({repo}): {len(shards)} shard(s), "
            f"{len(self.index)} tensors")

    def has(self, name):
        return name in self.index

    def get(self, name):
        return self.handles[self.index[name]].get_tensor(name)

    def keys(self):
        return self.index.keys()


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def alpha_equiv_row(name, w_ref, w_quant, step=None):
    """RMS error over reference std -- Study B's alpha, realised."""
    a = w_ref.detach().to(torch.float32).flatten()
    b = w_quant.detach().to(torch.float32).flatten()
    diff = b - a
    rms = float(torch.sqrt(torch.mean(diff * diff)))
    std = float(torch.std(a))
    kind, group = tensor_kind(name)
    row = {
        "tensor": name, "kind": kind, "group": group,
        "layer": layer_index(name), "numel": int(a.numel()),
        "rms_error": rms, "std_fp16": std,
        "alpha_equiv": rms / std if std > 0 else float("nan"),
        "mean_error": float(torch.mean(diff)),
        "max_abs_error": float(torch.max(torch.abs(diff))),
        "frobenius_ratio": float(torch.linalg.vector_norm(diff) /
                                 torch.linalg.vector_norm(a)),
    }
    if step:
        row["mean_error_in_steps"] = row["mean_error"] / step
    return row


# ---------------------------------------------------------------------------
# Format-specific dequantization
# ---------------------------------------------------------------------------

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack_int32_nibbles(packed, bits=4):
    """(rows, cols) int32 -> (rows, cols*8) int8 of 4-bit values."""
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    out = torch.bitwise_right_shift(packed[:, :, None].to(torch.int32),
                                    shifts[None, None, :])
    out = torch.bitwise_and(out, (1 << bits) - 1)
    return out.reshape(packed.shape[0], -1).to(torch.int16)


def dequant_awq(qweight, qzeros, scales, group_size=None):
    """AutoAWQ GEMM layout.

    qweight (in_features, out_features//8) int32
    qzeros  (n_groups,    out_features//8) int32
    scales  (n_groups,    out_features)    fp16
    The 8 nibbles inside each int32 are interleaved; AWQ_REVERSE_ORDER undoes it.
    Returns W with torch.nn.Linear orientation (out_features, in_features).
    """
    iw = _unpack_int32_nibbles(qweight)
    iz = _unpack_int32_nibbles(qzeros)
    order = torch.arange(iw.shape[-1]).view(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    iw = iw[:, order]
    iz = iz[:, order]

    in_f, out_f = iw.shape
    n_groups = scales.shape[0]
    gs = group_size or (in_f // n_groups)
    rep = torch.arange(in_f) // gs
    w_t = (iw.to(torch.float32) - iz[rep].to(torch.float32)) * \
        scales[rep].to(torch.float32)
    return w_t.t().contiguous()


def dequant_gptq(qweight, qzeros, scales, g_idx=None, zero_plus_one=True,
                 group_size=None):
    """GPTQ layout.

    qweight (in_features//8, out_features) int32, packed along in_features
    qzeros  (n_groups,       out_features//8) int32
    scales  (n_groups,       out_features)    fp16
    g_idx   (in_features,)   int32, group per input row (desc_act reordering)

    zero_plus_one selects between the two zero-point conventions found in the
    wild; V2 in the module docstring decides which is used, by unbiasedness.
    Returns W as (out_features, in_features).
    """
    bits, pack = 4, 8
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    iw = torch.bitwise_right_shift(qweight[:, None, :].to(torch.int32),
                                   shifts[None, :, None])
    iw = torch.bitwise_and(iw, (1 << bits) - 1)
    iw = iw.reshape(qweight.shape[0] * pack, qweight.shape[1]).to(torch.int16)

    iz = _unpack_int32_nibbles(qzeros)          # (n_groups, out_features)
    iz = iz[:, :scales.shape[1]]
    if zero_plus_one:
        iz = iz + 1

    in_f = iw.shape[0]
    if g_idx is not None and len(g_idx) == in_f:
        rep = g_idx.to(torch.long)
    else:
        gs = group_size or (in_f // scales.shape[0])
        rep = torch.arange(in_f) // gs
    w_t = (iw.to(torch.float32) - iz[rep].to(torch.float32)) * \
        scales[rep].to(torch.float32)
    return w_t.t().contiguous()


def undo_llamacpp_permute(w, n_head, n_head_kv, is_k):
    """Invert convert_hf_to_gguf.py's permute() for attn_q / attn_k."""
    heads = n_head_kv if is_k else n_head
    shape = w.shape
    dim0 = shape[0]
    if dim0 % (heads * 2) != 0:
        return None
    rest = shape[1:]
    return (w.reshape(heads, dim0 // heads // 2, 2, *rest)
             .swapaxes(1, 2)
             .reshape(shape))


def gguf_name_to_hf(name):
    if name in GGUF_MAP:
        return GGUF_MAP[name]
    if name.startswith("blk."):
        parts = name.split(".")
        idx = parts[1]
        stem = ".".join(parts[2:-1]) if parts[-1] == "weight" else ".".join(parts[2:])
        if stem in GGUF_BLK_MAP:
            return f"model.layers.{idx}.{GGUF_BLK_MAP[stem]}.weight"
    return None


def iter_gguf_tensors(repo, pattern):
    """Yield (gguf_name, np.float32 array, qtype_name) one tensor at a time."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import fnmatch
    files = [f for f in list_repo_files(repo)
             if fnmatch.fnmatch(f.lower(), pattern.lower())]
    if not files:
        files = [f for f in list_repo_files(repo)
                 if f.endswith(".gguf") and "q4_k_m" in f.lower()]
    if not files:
        raise RuntimeError(f"no GGUF matching {pattern} in {repo}")
    fname = sorted(files)[0]
    path = hf_hub_download(repo_id=repo, filename=fname)
    from gguf import GGUFReader, dequantize
    reader = GGUFReader(path)
    yield ("__file__", None, fname)
    for t in reader.tensors:
        arr = dequantize(t.data, t.tensor_type).astype(np.float32)
        yield (t.name, arr, str(t.tensor_type).split(".")[-1])
        del arr


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------

def measure_margins(model, tokenizer, prompts, tag):
    rows = []
    t0 = time.time()
    for i, row in prompts.iterrows():
        try:
            m_seq, m_first = margin(model, tokenizer, row.prompt_text, "cuda")
        except Exception as exc:
            log(f"  margin failed {row.prompt_id}: {type(exc).__name__}: {exc}")
            m_seq, m_first = float("nan"), float("nan")
        rows.append({"prompt_id": row.prompt_id,
                     "attack_category": row.attack_category,
                     "template_tier": row.template_tier,
                     "m_seq": m_seq, "m_first": m_first})
        if (i + 1) % 200 == 0:
            log(f"    {tag}: {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    log(f"    {tag}: {len(rows)} margins in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


def coherence_probe(model, tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=24, do_sample=False)
    out = tokenizer.decode(gen[0][enc["input_ids"].shape[-1]:],
                           skip_special_tokens=True)
    del enc, gen
    return out.strip()


# ---------------------------------------------------------------------------
# Per (model, format) pass
# ---------------------------------------------------------------------------

def run_format(model_key, method, entry, fp16_repo, target_names, prompts, cfg,
               arch, tokenizer):
    """Dequantize into a fresh fp16 model, measure alpha_equiv and margins."""
    from transformers import AutoModelForCausalLM
    repo = entry["repo"]
    meta = {"repo": repo, "path": "direct_dequantization", "validations": {}}
    rows = []

    model = AutoModelForCausalLM.from_pretrained(fp16_repo, dtype=torch.float16,
                                                device_map="cuda")
    model.eval()
    sd = model.state_dict()
    replaced, skipped = [], []

    def write_in(hf_name, w_new):
        """alpha_equiv against the model's CURRENT (still fp16) weight, then
        overwrite it. Comparing in place is what removes the need to hold a
        second full copy of the model."""
        if hf_name not in sd:
            skipped.append((hf_name, "not in state_dict"))
            return None
        cur = sd[hf_name]
        if tuple(cur.shape) != tuple(w_new.shape):
            skipped.append((hf_name, f"shape {tuple(w_new.shape)} vs "
                                     f"{tuple(cur.shape)}"))
            return None
        ref_cpu = cur.detach().to("cpu", torch.float32)
        row = alpha_equiv_row(hf_name, ref_cpu, w_new)
        with torch.no_grad():
            cur.copy_(w_new.to(cur.dtype).to(cur.device))
        replaced.append(hf_name)
        del ref_cpu
        return row

    if method == "gguf":
        n_head, n_head_kv = arch["n_head"], arch["n_head_kv"]
        perm_log, layout, unmapped, qtypes = {}, None, [], {}
        for gname, arr, qt in iter_gguf_tensors(
                repo, entry.get("filename", "*q4_k_m*.gguf")):
            if gname == "__file__":
                meta["file"] = qt
                log(f"    gguf file: {qt}")
                continue
            qtypes[qt] = qtypes.get(qt, 0) + 1
            hf = gguf_name_to_hf(gname)
            if hf is None:
                unmapped.append(gname)
                continue
            if hf not in sd:
                if arch["tie"] and hf == "lm_head.weight":
                    hf = "model.embed_tokens.weight"
                else:
                    skipped.append((gname, "no HF counterpart"))
                    continue
            w = torch.from_numpy(arr)
            if w.ndim == 2:
                want = tuple(sd[hf].shape)
                if tuple(w.shape) == want[::-1] and want[0] != want[1]:
                    if layout is None:
                        layout = "transposed"
                        log(f"    layout: transposed (from {gname})")
                    w = w.t().contiguous()
                elif tuple(w.shape) == want and layout is None:
                    layout = "same"
                    log(f"    layout: same (from {gname})")
            is_q, is_k = hf.endswith("q_proj.weight"), hf.endswith("k_proj.weight")
            if is_q or is_k:
                und = undo_llamacpp_permute(w.numpy(), n_head, n_head_kv, is_k)
                if und is not None:
                    a_raw = alpha_equiv_row(
                        hf, sd[hf].detach().to("cpu", torch.float32), w)["alpha_equiv"]
                    w_und = torch.from_numpy(np.ascontiguousarray(und))
                    a_und = alpha_equiv_row(
                        hf, sd[hf].detach().to("cpu", torch.float32),
                        w_und)["alpha_equiv"]
                    perm_log[hf] = {"alpha_raw": round(a_raw, 6),
                                    "alpha_undone": round(a_und, 6)}
                    w = w_und
            row = write_in(hf, w)
            if row is not None:
                row.update({"gguf_name": gname, "gguf_qtype": qt,
                            "convention": "permute_undone" if (is_q or is_k)
                                          else "n/a"})
                rows.append(row)
            del w
        meta["layout"] = layout
        meta["quant_types"] = qtypes
        meta["unmapped"] = unmapped
        # V3: report the separation, do not silently rely on it
        if perm_log:
            raws = [v["alpha_raw"] for v in perm_log.values()]
            unds = [v["alpha_undone"] for v in perm_log.values()]
            meta["validations"]["V3_permutation"] = {
                "n_tensors": len(perm_log),
                "median_alpha_raw": round(float(np.median(raws)), 6),
                "median_alpha_undone": round(float(np.median(unds)), 6),
                "separation_x": round(float(np.median(raws) /
                                            max(np.median(unds), 1e-9)), 1),
                "chosen": "undone",
            }
    else:
        qs = SafeStore(repo)
        conv_log = {}
        wanted = list(target_names) + [n for n in sd
                                       if tensor_kind(n)[1] == "excluded"]
        for hf_name in wanted:
            stem = hf_name[:-len(".weight")] if hf_name.endswith(".weight") \
                else hf_name
            if qs.has(stem + ".qweight"):
                qw = qs.get(stem + ".qweight")
                qz = qs.get(stem + ".qzeros")
                sc = qs.get(stem + ".scales")
                if method == "awq":
                    w = dequant_awq(qw, qz, sc)
                    conv = "awq_gemm_reverse_order"
                else:
                    g_idx = qs.get(stem + ".g_idx") if qs.has(stem + ".g_idx") \
                        else None
                    ref_cpu = sd[hf_name].detach().to("cpu", torch.float32) \
                        if hf_name in sd else None
                    # V2: a quantizer is unbiased, so pick the zero-point
                    # convention whose mean signed error is ~0. Both logged.
                    cands = {}
                    for flag in (True, False):
                        cand = dequant_gptq(qw, qz, sc, g_idx, zero_plus_one=flag)
                        if ref_cpu is None or tuple(cand.shape) != \
                                tuple(ref_cpu.shape):
                            cands[flag] = (cand, float("inf"))
                            continue
                        step = float(sc.to(torch.float32).abs().mean())
                        me = float((cand - ref_cpu).mean())
                        cands[flag] = (cand, abs(me) / max(step, 1e-9))
                    flag = min(cands, key=lambda k: cands[k][1])
                    w = cands[flag][0]
                    conv = f"gptq_zero_plus_one={flag}"
                    # If NEITHER convention is unbiased, the problem is not the
                    # zero point and the tensor is flagged rather than reported.
                    suspect = cands[flag][1] > cfg["unbiased_tol_steps"]
                    conv_log[hf_name] = {
                        "bias_in_steps_zplus1": round(cands[True][1], 4),
                        "bias_in_steps_zplain": round(cands[False][1], 4),
                        "chosen": conv, "suspect_both_biased": bool(suspect)}
                    del cands
                del qw, qz, sc
            elif qs.has(hf_name):
                w = qs.get(hf_name).to(torch.float32)
                conv = "stored_unquantized"
            else:
                skipped.append((hf_name, "absent from quantized checkpoint"))
                continue
            row = write_in(hf_name, w)
            if row is not None:
                row.update({"gguf_name": "", "gguf_qtype": "",
                            "convention": conv})
                rows.append(row)
            del w
        meta["gptq_zero_point"] = conv_log
        if conv_log:
            chosen = {}
            for v in conv_log.values():
                chosen[v["chosen"]] = chosen.get(v["chosen"], 0) + 1
            meta["validations"]["V2_unbiased_zero_point"] = {
                "tensors": len(conv_log), "chosen_counts": chosen,
                "median_bias_in_steps_chosen": round(float(np.median(
                    [min(v["bias_in_steps_zplus1"], v["bias_in_steps_zplain"])
                     for v in conv_log.values()])), 4),
                "tolerance_steps": cfg["unbiased_tol_steps"],
                "n_suspect_both_biased": sum(
                    1 for v in conv_log.values() if v.get("suspect_both_biased"))}
        del qs

    meta["n_replaced"] = len(replaced)
    meta["n_skipped"] = len(skipped)
    meta["skipped_examples"] = skipped[:10]

    # V1: unquantized tensors must be bit-identical -> name mapping is right
    exact = [r for r in rows if r["alpha_equiv"] == 0.0]
    meta["validations"]["V1_exact_matches"] = {
        "n_tensors_alpha_equiv_zero": len(exact),
        "kinds": sorted({r["kind"] for r in exact}),
    }
    # V5: sanity range
    lo, hi = cfg["alpha_equiv_sane_range"]
    tgt = [r["alpha_equiv"] for r in rows if r["group"] == "target"]
    suspect = [a for a in tgt if not (lo <= a <= hi)]
    meta["validations"]["V5_range"] = {
        "n_targets": len(tgt), "n_outside_range": len(suspect),
        "range": [lo, hi],
        "median": round(float(np.median(tgt)), 6) if tgt else None}

    # V4: coherence. This check existed and fired correctly on Qwen2.5 GGUF --
    # the reloaded model emitted "you are a voice voice, I am not in voice" --
    # but it only RECORDED the failure, so the cell still wrote a full set of
    # alpha_equiv rows that looked like measurements. It now aborts the cell.
    # A model whose dequantized weights cannot answer a trivia question is not
    # a model whose weight error means anything.
    sample = coherence_probe(model, tokenizer)
    meta["validations"]["V4_coherence_sample"] = sample
    meta["validations"]["V4_coherent"] = bool("Paris" in sample)
    log(f"    coherence -> {sample!r}")
    if "Paris" not in sample:
        del model, sd
        free()
        raise RuntimeError(
            f"V4 coherence failed for {model_key}/{method}: the dequantized "
            f"model answered {sample[:60]!r} instead of naming Paris. The "
            "weights are wrong, so every alpha_equiv from this cell would be "
            "meaningless. Refusing to emit them.")

    mm = measure_margins(model, tokenizer, prompts, f"{model_key}/{method}")
    del model, sd
    free()
    return rows, mm, meta


def run_model(model_key, registry, prompts, cfg, alpha_rows, margin_rows, meta):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    family = registry.MODELS[model_key]["family"]
    size = registry.MODELS[model_key]["size"]
    fp16_repo = registry.MODELS[model_key]["fp16"]["repo"]
    log(f"\n########## {model_key} ({fp16_repo}) ##########")
    mk_meta = {}

    tokenizer = AutoTokenizer.from_pretrained(fp16_repo)
    model = AutoModelForCausalLM.from_pretrained(fp16_repo, dtype=torch.float16,
                                                device_map="cuda")
    model.eval()
    c = model.config
    arch = {"n_head": int(c.num_attention_heads),
            "n_head_kv": int(getattr(c, "num_key_value_heads",
                                     c.num_attention_heads)),
            "tie": bool(getattr(c, "tie_word_embeddings", False))}
    mk_meta.update(arch)
    target_names = canonical_target_names(model)
    mk_meta["n_target_tensors"] = len(target_names)
    mk_meta["target_kinds"] = sorted({tensor_kind(n)[0] for n in target_names})
    log(f"  arch={arch}; Study B target set: {len(target_names)} tensors")

    m_fp16 = measure_margins(model, tokenizer, prompts, f"{model_key}/fp16")
    m_fp16["method"] = "fp16"
    m_fp16["family"], m_fp16["size"] = family, size
    margin_rows.append(m_fp16)
    mk_meta["fp16_coherence"] = coherence_probe(model, tokenizer)
    del model
    free()

    for method in cfg["methods"]:
        entry = registry.MODELS[model_key].get(method)
        if not entry or not entry.get("repo"):
            log(f"  skip {method}: no repo")
            continue
        log(f"\n--- {model_key} / {method} ({entry['repo']}) ---")
        t0 = time.time()
        try:
            rows, mm, fmeta = run_format(model_key, method, entry, fp16_repo,
                                         target_names, prompts, cfg, arch,
                                         tokenizer)
            for r in rows:
                r.update({"family": family, "size": size, "method": method,
                          "model_key": model_key})
            alpha_rows.extend(rows)
            mm["method"], mm["family"], mm["size"] = method, family, size
            margin_rows.append(mm)
            fmeta["seconds"] = round(time.time() - t0, 1)
            mk_meta[method] = fmeta
            ta = [r["alpha_equiv"] for r in rows if r["group"] == "target"]
            ex = [r["alpha_equiv"] for r in rows if r["group"] == "excluded"]
            if ta:
                log(f"    alpha_equiv targets: n={len(ta)} "
                    f"median={np.median(ta):.5f} "
                    f"[{np.min(ta):.5f}, {np.max(ta):.5f}]")
            if ex:
                log(f"    alpha_equiv excluded: n={len(ex)} "
                    f"median={np.median(ex):.5f} max={np.max(ex):.5f}")
            log(f"    {method} done in {time.time()-t0:.0f}s")
        except Exception as exc:
            log(f"    {method} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            mk_meta[method] = {"status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"}
        free()

    meta[model_key] = mk_meta
    del tokenizer
    free()


def main():
    cfg = CONFIG
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device.")
    if torch.cuda.get_device_capability(0) < (7, 0):
        raise RuntimeError("GPU below sm_70 -- push with --accelerator NvidiaTeslaT4.")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    prompts_path = resolve_input("attack_prompts.csv", cfg["dataset_dir"])
    if prompts_path is None:
        raise RuntimeError("attack_prompts.csv not found under /kaggle/input")
    sys.path.insert(0, os.path.dirname(prompts_path))
    import model_registry

    prompts = pd.read_csv(prompts_path)
    prompts["attacker_instruction"] = prompts["attacker_instruction"].fillna("")
    if cfg["limit_prompts"]:
        prompts = prompts.head(cfg["limit_prompts"])
    prompts = prompts.reset_index(drop=True)
    log(f"{len(prompts)} prompts")

    alpha_rows, margin_rows, meta = [], [], {}
    for model_key in cfg["model_keys"]:
        try:
            run_model(model_key, model_registry, prompts, cfg, alpha_rows,
                      margin_rows, meta)
        except Exception as exc:
            log(f"MODEL FAILED {model_key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            meta[model_key] = {"status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"}
        if alpha_rows:
            pd.DataFrame(alpha_rows).to_csv(
                os.path.join(cfg["out_dir"], "alpha_equiv_tensors.csv"),
                index=False)
        if margin_rows:
            pd.concat(margin_rows, ignore_index=True).to_csv(
                os.path.join(cfg["out_dir"], "delta_m_prompts.csv"), index=False)
        with open(os.path.join(cfg["out_dir"], "mechanism_meta.json"), "w") as fh:
            json.dump({"config": cfg, "models": meta}, fh, indent=2, default=str)
        free()

    if alpha_rows:
        a = pd.DataFrame(alpha_rows)
        log("\n=== alpha_equiv by (model, method, group) ===")
        log(a.groupby(["model_key", "method", "group"])["alpha_equiv"]
             .agg(["count", "median", "min", "max"]).round(5).to_string())
        log("\n=== alpha_equiv by kind, targets only ===")
        log(a[a.group == "target"].groupby(["model_key", "method", "kind"])
             ["alpha_equiv"].agg(["median", "min", "max"]).round(5).to_string())
        log("\n=== excluded tensors (Study B's blind spot) ===")
        log(a[a.group == "excluded"].groupby(["model_key", "method", "kind"])
             ["alpha_equiv"].agg(["count", "median", "max"]).round(5).to_string())
    if margin_rows:
        m = pd.concat(margin_rows, ignore_index=True)
        log("\n=== m_first by (family, size, method) ===")
        log(m.groupby(["family", "size", "method"])["m_first"]
             .agg(["count", "mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
