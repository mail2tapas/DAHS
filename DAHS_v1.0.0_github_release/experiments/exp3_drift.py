"""Exp 3: calibration-staleness sweep (kappa_max), monotonic drift."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    run_grid(["GREEDY-UTILITY", "TUNED-UTILITY", "DAHS", "DAHS-NO-CORRECTION", "DAHS-ORACLE-DRIFT"],
             [dict(kappa_max=k) for k in (0.0, 0.5, 1.0, 1.5, 2.0)], SEEDS, "exp3_drift")
