"""Exp 7: staleness estimator under limited and noisy feedback (strong drift, kappa_max=2)."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    st = [dict(kappa_max=2.0, fb_prob=p, obs_sigma=s) for s in (0.05, 0.2) for p in (1.0, 0.5, 0.2, 0.05)]
    run_grid(["DAHS", "DAHS-NO-CORRECTION"], st, SEEDS, "exp7_feedback")
