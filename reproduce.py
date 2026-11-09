"""
reproduce.py -- regenerate every derived artifact behind the paper from the raw
measurements in results/, on CPU, in dependency order.

    python reproduce.py              # all CPU steps (about 2 minutes)
    python reproduce.py --network    # also re-fetch generation configs and tokenizers from the Hub
    python verify_paper.py           # then check every number the paper quotes

The raw inputs (generations, judge labels, margin sweeps, measured tensor
error) came from GPU runs and an LLM judge; experiments/ holds the code that
produced them and experiments/kernels/ holds each kernel exactly as it ran.
Everything below is deterministic: bootstraps use seed 20260827.

The committed results/ already contain these outputs, so after a run
`git status` should show no changes. Any diff is a reproducibility failure.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join("experiments", "analysis")

# (script, args, needs network). Order is the dependency order.
STEPS = [
    ("experiments/margin_analysis.py", ["--sweep", "results/margin/margin_sweep_all.csv",
                                        "--results", "results/results.csv",
                                        "--out", "results/margin/per_prompt_margins.csv"], False),
    (f"{A}/decoding_audit.py", [], True),              # reads each checkpoint's generation_config.json
    (f"{A}/continuation_token_check.py", [], True),    # reads each model's tokenizer
    (f"{A}/forensics.py", [], False),
    (f"{A}/regrid_analysis.py", [], False),             # Falcon3, stated decoding
    (f"{A}/phase21_analysis.py", [], False),            # Llama-3.2 and Qwen2.5, stated decoding
    (f"{A}/decoding_confound.py", [], False),
    (f"{A}/studyb_baseline.py", [], False),
    (f"{A}/mechanism_analysis.py", [], False),
    (f"{A}/mechanism_verdict.py", [], False),
    (f"{A}/dose_response_recheck.py", [], False),
    (f"{A}/delta_m_tstats.py", [], False),
    (f"{A}/threshold_analysis.py", [], False),
    (f"{A}/awq_anomaly.py", [], False),
    (f"{A}/null_analysis.py", [], False),
    (f"{A}/null_is_it_real.py", [], False),
    (f"{A}/empty_reproducibility.py", [], False),
    (f"{A}/human_review_analysis.py", [], False),
    # Exits 1 when its pre-registered criterion fails. It does, on Llama-3.2
    # (Spearman 0.625 < 0.90), and the paper reports that; see EXPECTED_FAIL.
    (f"{A}/token_ablation_analysis.py", [], False),
    (f"{A}/qwen_mechanism.py", [], False),
    (f"{A}/index_stats.py", [], False),
    (f"{A}/review_audit.py", [], False),
    (f"{A}/paper_tables.py", [], False),
]

# Scripts whose non-zero exit is a recorded scientific outcome, not a crash.
EXPECTED_FAIL = {f"{A}/token_ablation_analysis.py": 1}


def main():
    network = "--network" in sys.argv
    for script, args, needs_net in STEPS:
        if needs_net and not network:
            print(f"skip  {script} (needs --network)")
            continue
        t = time.time()
        r = subprocess.run([sys.executable, script, *args], cwd=HERE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == EXPECTED_FAIL.get(script, 0) != 0:
            print(f"ok    {script}  {time.time() - t:5.1f}s  (pre-registered criterion fails, as reported)")
            continue
        if r.returncode != 0:
            print(r.stdout[-3000:])
            sys.exit(f"FAILED {script} (exit {r.returncode})")
        print(f"ok    {script}  {time.time() - t:5.1f}s")
    print("\nAll steps ran. `git status` should be clean; then run verify_paper.py.")


if __name__ == "__main__":
    main()
