"""Exp 11: sensitivity of DAHS to the forgetting factor lambda (the GWO optimum
lies at its upper bound 0.999). a, b and m are kept at their tuned values;
lambda is varied from 0.90 to 1.0 under the default drift, strong drift and
non-monotonic drift.
    python experiments/exp11_lambda.py      (~20 min on 2 cores)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid
LAMS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)
if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else None
    names = [f"DAHS-LAMBDA-{l}" for l in LAMS] + ["DAHS-NO-CORRECTION"]
    st = [dict(kappa_max=1.0), dict(kappa_max=2.0), dict(drift_shape="rw", rw_sigma=0.6)]
    run_grid(names, st, range(1, 21), "exp11_lambda", workers=w)
