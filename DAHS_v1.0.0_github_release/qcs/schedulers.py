"""Broker scheduling policies.

All policies see the same information at dispatch time: the published
calibration of every device, each device's queue length and its average
service time (wait estimate), and the job's circuit and shot count.
"""
import math
import numpy as np
from .sim import esp_of

NAMES = ["RANDOM", "LEAST-BUSY", "BEST-FIDELITY", "MIN-TIME", "FID-THRESHOLD",
         "GREEDY-UTILITY", "DAHS"]


def _whole(sim, job):
    """(device index, fp, published ESP now, wait estimate, run time) per device."""
    rec = sim.circs[job.circ]
    out = []
    for dev in sim.devs:
        fp = rec["dev"].get(dev.key)
        if fp is None:
            continue
        out.append((dev.idx, fp, sim.pub_esp(fp, dev), dev.wait_estimate(sim.now),
                    sim.run_time(job, fp)))
    return out


class Random:
    def decide(self, sim, job):
        c = _whole(sim, job)
        return ("whole", c[int(sim.rng_sched.integers(len(c)))][0])


class LeastBusy:                       # fewest jobs in the queue (IBM "least busy")
    def decide(self, sim, job):
        c = _whole(sim, job)
        return ("whole", min(c, key=lambda x: (sim.devs[x[0]].n_jobs(), -x[2]))[0])


class BestFidelity:                    # highest published ESP
    def decide(self, sim, job):
        return ("whole", max(_whole(sim, job), key=lambda x: x[2])[0])


class MinTime:                         # earliest predicted completion
    def decide(self, sim, job):
        return ("whole", min(_whole(sim, job), key=lambda x: x[3] + x[4])[0])


class FidThreshold:
    """Earliest completion among devices whose published ESP is within a
    relative margin of the best (fidelity-bounded queue-time minimisation)."""
    def __init__(self, margin=0.10):
        self.margin = margin

    def decide(self, sim, job):
        c = _whole(sim, job)
        best = max(x[2] for x in c)
        ok = [x for x in c if x[2] >= (1 - self.margin) * best]
        return ("whole", min(ok, key=lambda x: x[3] + x[4])[0])


class GreedyUtility:                   # maximise published ESP - w_t * predicted turnaround
    def decide(self, sim, job):
        w = sim.cfg.w_t_per_min
        return ("whole", max(_whole(sim, job), key=lambda x: x[2] - w * (x[3] + x[4]) / 60)[0])


class DAHS:
    """Drift-Aware Hybrid Scheduler.

    score = F_hat - a * w_t * T_hat/60 - b * w_t * (R/60) * rho_b      (whole or cut)
      F_hat: ESP predicted at the expected start time, with each device's
             published calibration corrected by a learned staleness rate
             k_b (true error ~ published * exp(k_b * h/24));
      T_hat: predicted wait + run time; rho_b = n_b/(n_b+1) congestion proxy.
    k_b is learned online from fidelity feedback with exponentially forgotten
    least squares (forgetting factor lam). Cut execution is considered only
    when its score beats the best whole-circuit score by margin m.
    theta = (a, b, lam, m) is tuned offline with the Grey Wolf Optimizer.
    """
    DEFAULT = (1.0, 0.0, 0.98, 0.0)
    K_MIN, K_MAX = -1.0, 3.0

    def __init__(self, theta=None, correct=True, cut=True, oracle=False):
        self.theta = tuple(theta) if theta is not None else self.DEFAULT
        self.correct, self.cut, self.oracle = correct, cut, oracle

    def reset(self, sim):
        n = len(sim.devs)
        self.sxx = np.full(n, 0.05); self.sxy = np.zeros(n)

    def k_hat(self, sim, i):
        if self.oracle:
            return sim.devs[i].kappa
        if not self.correct:
            return 0.0
        # safeguard: clip to a plausible range of staleness rates
        return float(min(max(self.sxy[i] / self.sxx[i], self.K_MIN), self.K_MAX))

    def f_hat(self, sim, fp, dev, t_start):
        d, age = dev.day_and_age(t_start)
        if self.oracle == "path":                     # knows the true multiplier path
            return esp_of(fp, dev.pub[d], dev.tab, math.exp(min(700.0, dev.log_mult(d, age))))
        return esp_of(fp, dev.pub[d], dev.tab, math.exp(min(700.0, self.k_hat(sim, dev.idx) * age / 24.0)))   # cap avoids overflow for very long predicted waits

    def decide(self, sim, job):
        a, b, lam, m = self.theta
        w = sim.cfg.w_t_per_min
        rec = sim.circs[job.circ]
        best, best_s = None, -1e9
        for dev in sim.devs:
            fp = rec["dev"].get(dev.key)
            if fp is None:
                continue
            W = dev.wait_estimate(sim.now); R = sim.run_time(job, fp)
            n = dev.n_jobs(); rho = n / (n + 1.0)
            s = self.f_hat(sim, fp, dev, sim.now + W) - a * w * (W + R) / 60 - b * w * (R / 60) * rho
            if s > best_s:
                best, best_s = ("whole", dev.idx), s
        if self.cut and sim.cfg.allow_cut and "frag" in rec:
            ov = rec["cut_overhead"]
            opts = []
            for k in (0, 1):
                lst = []
                for dev in sim.devs:
                    fp = rec["frag"][k]["dev"].get(dev.key)
                    if fp is None:
                        continue
                    W = dev.wait_estimate(sim.now); R = sim.run_time(job, fp, ov)
                    n = dev.n_jobs(); rho = n / (n + 1.0)
                    lst.append((dev.idx, math.log(max(self.f_hat(sim, fp, dev, sim.now + W), 1e-300)), W, R, rho))
                opts.append(lst)
            for (ia, fa, Wa, Ra, ra) in opts[0]:
                for (ib, fb, Wb, Rb, rb) in opts[1]:
                    T = Wa + Ra + Rb if ia == ib else max(Wa + Ra, Wb + Rb)
                    T += sim.recon_time(rec, job.shots)
                    s = sim.cut_quality(fa + fb, rec) - a * w * T / 60 - b * w * (Ra / 60 * ra + Rb / 60 * rb) - m
                    if s > best_s:
                        best, best_s = ("cut", ia, ib), s
        return best

    def feedback(self, sim, dev, age, g_pub, dec, obs):
        if not self.correct or self.oracle:
            return
        lam = self.theta[2]
        if not (0 < obs < 1 and g_pub < -1e-6):
            return
        r = (math.log(obs) - dec) / g_pub      # ~ multiplier on published error rates
        if r <= 0:
            return
        x = age / 24.0
        y = min(max(math.log(r), -1.0), 3.0)
        i = dev.idx
        self.sxx[i] = lam * self.sxx[i] + x * x
        self.sxy[i] = lam * self.sxy[i] + x * y


class PGLinear:
    """Learning baseline: softmax policy over devices with linear scores of
    per-device features, trained by REINFORCE (policy gradient) on the
    training seeds with the job utility as reward. Whole-circuit placement only.
    Features: published ESP, predicted turnaround/60, congestion, run/60,
    calibration age/24, bias."""
    NF = 6

    def __init__(self, w=None, train=False, lr=0.05, rng=None):
        self.w = np.zeros(self.NF) if w is None else np.asarray(w, float)
        self.train, self.lr = train, lr
        self.rng = rng or np.random.default_rng(0)
        self.buf = {}; self.base = 0.0

    def feats(self, sim, job):
        rec = sim.circs[job.circ]; X = []; idx = []
        for dev in sim.devs:
            fp = rec["dev"].get(dev.key)
            if fp is None:
                continue
            W = dev.wait_estimate(sim.now); R = sim.run_time(job, fp)
            n = dev.n_jobs(); _, age = dev.day_and_age(sim.now)
            X.append([sim.pub_esp(fp, dev), (W + R) / 60.0, n / (n + 1.0), R / 60.0, age / 24.0, 1.0])
            idx.append(dev.idx)
        return np.array(X), idx

    def decide(self, sim, job):
        X, idx = self.feats(sim, job)
        z = X @ self.w * 10.0
        if self.train:
            p = np.exp(z - z.max()); p /= p.sum()
            k = int(self.rng.choice(len(idx), p=p))
            self.buf[job.jid] = (X, k, p)
        else:
            k = int(np.argmax(z))
        return ("whole", idx[k])

    def update(self, sim):
        """Episode-end REINFORCE update with a running-mean baseline."""
        w = sim.cfg.w_t_per_min; g = np.zeros(self.NF); n = 0
        for j in sim.jobs:
            if j.end is None or j.jid not in self.buf:
                continue
            X, k, p = self.buf[j.jid]
            r = j.fid - w * (j.end - j.arr) / 60.0
            adv = r - self.base
            g += adv * (X[k] - p @ X); n += 1
            self.base = 0.999 * self.base + 0.001 * r
        if n:
            self.w += self.lr * g / n
        self.buf = {}


def make(name, theta=None, extra=None):
    if name.startswith("DAHS-LAMBDA-"):        # DAHS with the tuned (a, b, m) and a fixed forgetting factor
        lam = float(name[len("DAHS-LAMBDA-"):])
        return lambda: DAHS((theta[0], theta[1], lam, theta[3]))
    return {
        "RANDOM": Random, "LEAST-BUSY": LeastBusy, "BEST-FIDELITY": BestFidelity,
        "MIN-TIME": MinTime, "FID-THRESHOLD": FidThreshold,
        "GREEDY-UTILITY": GreedyUtility,
        "DAHS": lambda: DAHS(theta),
        "DAHS-DEFAULT": lambda: DAHS(None),
        "DAHS-NO-CORRECTION": lambda: DAHS(theta, correct=False),
        "DAHS-NO-CUT": lambda: DAHS(theta, cut=False),
        "DAHS-ORACLE-DRIFT": lambda: DAHS(theta, oracle=True),
        "DAHS-ORACLE-PATH": lambda: DAHS(theta, oracle="path"),
        "TUNED-UTILITY": lambda: DAHS((extra or {}).get("tuned_utility"), correct=False, cut=False),
        "PG-LINEAR": lambda: PGLinear((extra or {}).get("pg_w")),
    }[name]
