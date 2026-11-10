"""Exp 8: robustness to (a) error of the ESP proxy itself and (b) non-monotonic
(random-walk) intra-day drift."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    names = ["GREEDY-UTILITY", "TUNED-UTILITY", "DAHS", "DAHS-NO-CORRECTION", "DAHS-ORACLE-PATH"]
    run_grid(names, [dict(model_err_sigma=s) for s in (0.0, 0.1, 0.2, 0.4)], SEEDS, "exp8a_proxy_error")
    run_grid(names, [dict(drift_shape="rw", rw_sigma=s) for s in (0.3, 0.6, 1.0)], SEEDS, "exp8b_rw_drift")
