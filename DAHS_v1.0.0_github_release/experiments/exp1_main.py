"""Exp 1: main comparison at three broker loads (20 test seeds, 48 h)."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
from qcs.schedulers import NAMES
if __name__ == "__main__":
    run_grid(NAMES + ["TUNED-UTILITY", "PG-LINEAR"], [dict(load=l) for l in (0.5, 1.0, 2.0)], SEEDS, "exp1_main")
