"""Reproducibility check of the cutting validation data and of the cut model.

The 14 circuits x 5 sampling seeds of results/exp_cut_repeats.csv were run again
from scratch with the released code:
    CUTREP_PREFIX=verify python experiments/exp_cut_repeats.py dj:10
    CUTREP_PREFIX=verify python experiments/exp_cut_repeats.py bv:8 bv:10 ... wstate:12
This script merges the two output files, compares every Q value with the shipped
data and refits the cut model (data/cut_cost.py fit_model) on the new runs.
Q values are seeded and must agree exactly; reconstruction times are wall-clock
times and depend on the machine and its load.
    python experiments/verify_cut_repeats.py   -> results/cut_repeats_verification.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "data"))
from cut_cost import fit_model

K = ["algo", "width", "seed"]
old = pd.read_csv(RES / "exp_cut_repeats.csv")
new = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("exp_cut_repeats_verify_*.csv"))], ignore_index=True)
new = new.drop_duplicates(K, keep="last")
new.to_csv(RES / "exp_cut_repeats_verify.csv", index=False)
m = old.merge(new, on=K, suffixes=("", "_new"))
M0 = json.loads((RES / "cut_model.json").read_text()); M1 = fit_model(new)
out = dict(rows_shipped=len(old), rows_rerun=len(new), rows_matched=len(m),
           max_abs_diff_q_whole=float((m.q_whole - m.q_whole_new).abs().max()),
           max_abs_diff_q_cut=float((m.q_cut - m.q_cut_new).abs().max()),
           shots_per_sub_equal=bool((m.shots_per_sub == m.shots_per_sub_new).all()),
           n_sub_equal=bool((m.n_sub == m.n_sub_new).all()),
           shipped={k: M0[k] for k in ("alpha", "c0", "c1", "r0", "r1", "resid_sd")},
           rerun={k: M1[k] for k in ("alpha", "c0", "c1", "r0", "r1", "resid_sd")},
           recon_s_shipped_range=[float(old.recon_s.min()), float(old.recon_s.max())],
           recon_s_rerun_range=[float(new.recon_s.min()), float(new.recon_s.max())])
M2 = dict(M0, r0=M1["r0"], r1=M1["r1"], note="cut_model.json with r0, r1 refitted on the verification rerun; eta identical")
(RES / "cut_model_verify.json").write_text(json.dumps(M2, indent=1))
(RES / "cut_repeats_verification.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
