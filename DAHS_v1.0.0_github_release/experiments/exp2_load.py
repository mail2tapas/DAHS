"""Exp 2: background-load sweep (other users' load on all devices)."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    run_grid(["BEST-FIDELITY", "FID-THRESHOLD", "GREEDY-UTILITY", "TUNED-UTILITY", "DAHS"],
             [dict(ext_load=x) for x in (0.6, 0.8, 1.0, 1.1, 1.2)], SEEDS, "exp2_load")
