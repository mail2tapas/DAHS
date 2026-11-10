"""Exp 4: ablation of DAHS components at the default setting."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    run_grid(["DAHS", "DAHS-DEFAULT", "DAHS-NO-CORRECTION", "DAHS-NO-CUT", "DAHS-ORACLE-DRIFT",
              "GREEDY-UTILITY", "TUNED-UTILITY"], [dict(load=1.0)], SEEDS, "exp4_ablation")
