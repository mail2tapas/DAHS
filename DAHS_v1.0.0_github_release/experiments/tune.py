"""Offline tuning with the Grey Wolf Optimizer (GWO).

Targets:
  dahs            theta = (a, b, lam, m)  bounds a[0,10] b[0,100] lam[0.90,0.999] m[0,0.3]
  tuned_utility   theta = (a, b)          same bounds; no staleness correction, no cutting
Objective: mean utility over training seeds 101-106 (6-hour runs, 1-hour warm-up,
load 1), one run per seed per candidate. Test seeds (1-20) are never used.

Examples
  python experiments/tune.py --target dahs --gwo-seed 0 --out theta.json
  python experiments/tune.py --target dahs --gwo-seed 1 --out theta_gwo1.json
  python experiments/tune.py --target tuned_utility --out theta_tuned_utility.json
  python experiments/tune.py --target dahs --w 0.03 --out theta_w0.03.json
  python experiments/tune.py --target dahs --algo-split train --out theta_algo_train.json
  python experiments/tune.py --target dahs --algo-split train --cut-model cut_model_trainfam.json --out theta_algo_train_strict.json
"""
import argparse, json, sys, time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.config import Config
from qcs.sim import Sim, load_library
from qcs.schedulers import DAHS
from qcs.gwo import gwo
from qcs.runner import RES

TRAIN_SEEDS = (101, 102, 103, 104, 105, 106)
ALGO_SPLIT = {  # generalisation: tune on one half of the circuit families, test on the other
    "train": ("ghz", "dj", "wstate", "qaoa", "qftentangled", "vqe_su2", "qnn", "randomcircuit"),
    "test": ("bv", "graphstate", "qft", "vqe_real_amp", "vqe_two_local", "ae", "bmw_quark_cardinality",
             "cdkm_ripple_carry_adder"),
}


def _u(args):
    target, theta, seed, over = args
    cfg = replace(Config(), hours=6.0, **over)
    if target == "dahs":
        sch = DAHS(tuple(theta))
    else:
        sch = DAHS((theta[0], theta[1], 0.98, 0.0), correct=False, cut=False)
    return Sim(cfg, seed, sch, load_library()).run()["utility"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="dahs", choices=["dahs", "tuned_utility"])
    ap.add_argument("--gwo-seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--algo-split", default=None, choices=[None, "train"])
    ap.add_argument("--wolves", type=int, default=10)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--cut-model", default=None, help="cut-model file in results/ (default cut_model.json)")
    ap.add_argument("--out", default="theta.json")
    a = ap.parse_args()
    over = {}
    if a.w is not None: over["w_t_per_min"] = a.w
    if a.algo_split: over["algos"] = ALGO_SPLIT[a.algo_split]
    if a.cut_model: over["cut_model_file"] = a.cut_model
    if a.target == "dahs":
        lo, hi = [0.0, 0.0, 0.90, 0.0], [10.0, 100.0, 0.999, 0.30]
    else:
        lo, hi = [0.0, 0.0], [10.0, 100.0]
    t0 = time.time(); pool = Pool(); evals = [0]
    def f(x):
        evals[0] += 1
        return float(np.mean(pool.map(_u, [(a.target, tuple(x), s, over) for s in TRAIN_SEEDS])))
    best, fit, hist = gwo(f, lo, hi, n_wolves=a.wolves, iters=a.iters, seed=a.gwo_seed)
    default = f(DAHS.DEFAULT if a.target == "dahs" else (1.0, 0.0))
    RES.mkdir(exist_ok=True)
    (RES / a.out).write_text(json.dumps(dict(
        target=a.target, theta=[float(v) for v in best], train_utility=fit, default_train_utility=default,
        bounds=[lo, hi], train_seeds=TRAIN_SEEDS, hours=6.0, overrides={k: (list(v) if isinstance(v, tuple) else v) for k, v in over.items()},
        gwo_seed=a.gwo_seed, wolves=a.wolves, iters=a.iters, evaluations=evals[0], history=hist,
        seconds=time.time() - t0), indent=1))
    print(f"[{a.out}] theta={np.round(best, 3).tolist()} train utility={fit:.4f} (default {default:.4f}) "
          f"evals={evals[0]} time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
