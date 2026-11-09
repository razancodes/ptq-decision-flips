# Analysis scripts

Run in order by `../../reproduce.py`. Each script's docstring records the
question as it was posed when the script was written, and states its
interpretation before the numbers were seen. References such as `§4.7` point
to the authors' working notes, which are not part of this release; read them as
"the analysis section under discussion at the time".

Some scripts test hypotheses the paper ended up rejecting (for example the
isotropic-noise account in `threshold_analysis.py`, or the pooled synthetic
control in `studyb_baseline.py`). They are kept because the paper's corrections
rest on them. The claims the paper makes are the ones `../../verify_paper.py`
checks, and nothing in these docstrings extends them.
