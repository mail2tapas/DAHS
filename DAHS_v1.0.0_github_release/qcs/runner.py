"""Run (scheduler, setting, seed) combinations in parallel and save a CSV."""
import json, os, time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .sim import Sim, load_library
from . import schedulers as S

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
THETA = RES / "theta.json"


def load_theta():
    if THETA.exists():
        return tuple(json.loads(THETA.read_text())["theta"])
    print("warning: results/theta.json not found; DAHS uses default weights "
          "(run experiments/tune.py first)")
    return None


def load_extra():
    ex = {}
    f = RES / "theta_tuned_utility.json"
    if f.exists(): ex["tuned_utility"] = tuple(json.loads(f.read_text())["theta"]) + (0.98, 0.0)
    f = RES / "pg_weights.json"
    if f.exists(): ex["pg_w"] = json.loads(f.read_text())["w"]
    return ex


def _one(args):
    name, over, seed, theta = args
    over = dict(over)
    th = over.pop("_theta", None) or theta
    cfg = replace(Config(), **over)
    sch = S.make(name, th, load_extra())()
    r = Sim(cfg, seed, sch, load_library()).run()
    row = dict(scheduler=name, seed=seed, **{k: (v if not isinstance(v, (tuple, list)) else str(v)) for k, v in over.items()},
               n=r["n"], mean_F=r["mean_F"], mean_T_min=r["mean_T_min"], p95_T_min=r["p95_T_min"],
               utility=r["utility"], cut_share=r["cut_share"], unfinished=r["unfinished"],
               long_wait_share=r["long_wait_share"], decision_ms=r["decision_ms"],
               decision_p99_ms=r["decision_p99_ms"], throughput_per_h=r["throughput_per_h"],
               mean_queue=r["mean_queue"], cut_eligible_share=r["cut_eligible_share"],
               mean_util=float(np.mean(r["util"])), max_util=float(np.max(r["util"])))
    for i, u in enumerate(r["util"][:8]):
        row[f"util_d{i}"] = u
    for i, s in enumerate(r["dev_share"]):
        row[f"share_d{i}"] = s
    # keep per-job arrays for post-hoc analyses (break-even, by width)
    row["_F"] = r["F"].tolist(); row["_T"] = r["T"].tolist(); row["_W"] = r["width"].tolist()
    return row


def run_grid(names, settings, seeds, out_name, workers=None, theta=None):
    theta = theta if theta is not None else load_theta()
    tasks = [(n, s, sd, theta) for s in settings for n in names for sd in seeds]
    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    with Pool(workers) as p:
        rows = []
        for i, row in enumerate(p.imap_unordered(_one, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % max(1, len(tasks) // 10) == 0:
                print(f"  {i+1}/{len(tasks)} runs done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    arrays = df[["scheduler", "seed"] + [c for c in df.columns if c in settings[0] and c != "_theta"] + ["_F", "_T", "_W"]]
    arrays.to_pickle(RES / f"{out_name}_jobs.pkl")
    df.drop(columns=["_F", "_T", "_W"]).to_csv(RES / f"{out_name}.csv", index=False)
    print(f"saved results/{out_name}.csv ({len(df)} runs, {time.time()-t0:.0f}s)")
    return df
