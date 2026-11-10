"""Exp 13: error in the fitted cut model and an explicit uncertainty margin.
The simulator's true efficiency of every cut job is eta + delta (delta = -0.5,
-0.25, 0, +0.25; the residual standard deviation of the fit is about 0.2), while
the broker predicts eta - z * se_pred(n_sub), with z = 0 (point estimate, the
default) or z = 1 (one prediction standard error below the fit). Load 1, 20 seeds.
    python experiments/exp13_cut_mismatch.py      (~12 min on 2 cores)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid
if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st = [dict(cut_eff_true_delta=d, cut_eta_z=z) for z in (0.0, 1.0) for d in (-0.5, -0.25, 0.0, 0.25)]
    run_grid(["DAHS", "DAHS-NO-CUT"], st, range(1, 21), "exp13_cut_mismatch", workers=w)
