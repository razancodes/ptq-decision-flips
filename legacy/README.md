# Legacy notebooks (Study A)

The Google Colab notebooks behind the original grid, `results/results.csv`. Kept
as a record, because the decoding confound the paper reports was found in them.

**Do not use them to reproduce flip rates.** `03_run_experiment.ipynb` calls
`model_loaders.generate_reply()`, which passes no sampling arguments, so every
quantized checkpoint inherited its own repository's `generation_config.json`.
The stated-decoding replacements are `experiments/regrid_driver.py` and the
kernels in `experiments/kernels/`.

- `02_attack_prep.ipynb`: builds `data/attack_prompts.csv` (also `data/build_attack_prompts.py`).
  Upstream AdvBench and InjecAgent were fetched from `main`, not a pinned commit.
- `03_run_experiment.ipynb`: Study A generation, inherited decoding.
- `04_classify.ipynb`: the judge rubrics. `experiments/build_judge.py` extracts them
  from here and asserts their SHA-256. Its cells were never executed in the saved copy.
- `05_analysis.ipynb`: exploratory analysis, superseded by `experiments/analysis/`.

Secrets were read from Colab's `userdata` store; no credential is in these files.
