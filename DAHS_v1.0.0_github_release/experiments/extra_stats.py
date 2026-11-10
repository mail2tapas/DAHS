"""Extra statistics: staleness-rate estimation error of DAHS,
decision time per job, device usage (default setting, test seeds 1-20)."""
import sys, json, time; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from qcs.config import Config
from qcs.sim import Sim
from qcs.schedulers import DAHS
from qcs.runner import load_theta, RES

def main():
    th = load_theta(); err = []; dt = []; ks = []
    for s in range(1, 21):
        sch = DAHS(th); sim = Sim(Config(), s, sch)
        orig = sch.decide
        def timed(sim_, job, _o=orig):
            t0 = time.perf_counter(); r = _o(sim_, job); dt.append(time.perf_counter() - t0); return r
        sch.decide = timed
        sim.run()
        kh = np.array([sch.k_hat(sim, i) for i in range(len(sim.devs))]); kt = np.array([d.kappa for d in sim.devs])
        err.extend(np.abs(kh - kt).tolist()); ks.append((kh.tolist(), kt.tolist()))
    out = dict(kappa_mae=float(np.mean(err)), kappa_corr=float(np.corrcoef(
        np.concatenate([np.array(a) for a, b in ks]), np.concatenate([np.array(b) for a, b in ks]))[0, 1]),
        decision_ms_mean=1000 * float(np.mean(dt)), decision_ms_p99=1000 * float(np.percentile(dt, 99)))
    json.dump(out, open(RES / "extra_stats.json", "w"), indent=1); print(out)

if __name__ == "__main__":
    main()
