"""Train the policy-gradient learning baseline (PG-LINEAR) with REINFORCE on
the training seeds (101-104, 6-hour episodes), then save its weights.
    python experiments/train_pg.py        (~10 min on 2 cores)"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.config import Config
from qcs.sim import Sim, load_library
from qcs.schedulers import PGLinear
from qcs.runner import RES

EPISODES = 150
SEEDS = (101, 102, 103, 104, 105, 106)


def evaluate(w):
    return float(np.mean([Sim(replace(Config(), hours=6.0), s, PGLinear(w), load_library()).run()["utility"] for s in SEEDS]))


def main():
    t0 = time.time()
    pol = PGLinear(train=True, lr=0.2, rng=np.random.default_rng(0))
    # start from a sensible prior: prefer fidelity, penalise turnaround
    pol.w = np.array([1.0, -0.01, 0.0, 0.0, 0.0, 0.0])
    hist = []; best_w, best_u = pol.w.copy(), evaluate(pol.w)
    for ep in range(EPISODES):
        seed = SEEDS[ep % len(SEEDS)] * 1000 + ep          # fresh episode workloads from training seeds
        sim = Sim(replace(Config(), hours=6.0), seed, pol, load_library()); sim.run()
        pol.update(sim)
        if (ep + 1) % 10 == 0:
            u = evaluate(pol.w); hist.append((ep + 1, u, pol.w.tolist()))
            if u > best_u: best_u, best_w = u, pol.w.copy()
            print(f"episode {ep+1}: train utility (greedy) {u:.4f} w={np.round(pol.w,3).tolist()}", flush=True)
    (RES / "pg_weights.json").write_text(json.dumps(dict(w=best_w.tolist(), train_utility=best_u, history=hist,
                                                          episodes=EPISODES, seconds=time.time() - t0), indent=1))
    print(f"saved results/pg_weights.json  best train utility {best_u:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
