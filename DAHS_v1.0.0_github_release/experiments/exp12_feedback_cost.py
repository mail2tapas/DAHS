"""Exp 12: practical feedback. Strong drift (kappa_max = 2), load 1.
(a) feedback delayed by 0, 1 or 4 hours after the job ends;
(b) feedback that costs device time: each job that returns feedback occupies
its device for (1 + c) times its run time (c = 1: one extra mirror or
known-answer run of the same length), charged only to DAHS.
    python experiments/exp12_feedback_cost.py      (~15 min on 2 cores)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid
if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st = [dict(kappa_max=2.0, fb_prob=p, fb_delay_h=d, fb_probe_cost=c)
          for (p, d, c) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 4.0, 0.0),
                            (0.05, 0.0, 0.0), (0.05, 1.0, 0.0), (0.05, 4.0, 0.0),
                            (1.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.05, 0.0, 1.0)]]
    run_grid(["DAHS", "DAHS-NO-CORRECTION"], st, range(1, 21), "exp12_feedback_cost", workers=w)
