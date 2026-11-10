"""Exp 6b: retuned policies at other time weights (complements the fixed-policy
post-hoc analysis of Exp 6). DAHS retuned with GWO at w_t = 0.03 and 0.1."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
from qcs.schedulers import NAMES
if __name__ == "__main__":
    for w in (0.03, 0.1):
        th = tuple(json.loads((RES / f"theta_w{w}.json").read_text())["theta"])
        run_grid(["LEAST-BUSY", "MIN-TIME", "FID-THRESHOLD", "GREEDY-UTILITY", "DAHS"],
                 [dict(w_t_per_min=w, _theta=th)], SEEDS, f"exp6b_w{w}")
