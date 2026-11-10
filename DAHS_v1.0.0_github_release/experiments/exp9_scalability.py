"""Exp 9: scalability with fleet size (8-64 devices; broker load scales with the fleet).
Records decision latency, throughput and turnaround. 24-hour runs, 5 seeds."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
if __name__ == "__main__":
    run_grid(["GREEDY-UTILITY", "TUNED-UTILITY", "DAHS"], [dict(fleet_factor=f, hours=24.0) for f in (1, 2, 4, 8)],
             range(1, 6), "exp9_scalability", workers=2)   # two processes on two cores: one run per core
