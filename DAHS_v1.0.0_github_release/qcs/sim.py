"""Discrete-event simulator of a multi-QPU quantum cloud with a job broker.

Each device (QPU) serves one job at a time from a FIFO queue that also holds
background jobs of other users. A broker dispatches its own jobs on arrival
(the scheduling decision) to a device, or, for cuttable circuits, splits them
into two fragments run on one or two devices.

Fidelity is the estimated success probability (ESP) of the transpiled circuit,
computed from gate counts on the physical qubits/couplers it uses:
    ESP = prod (1-e_gate)^count * prod (1-e_readout) * exp(-sum dur/T2)
Devices are recalibrated once a day at their own hour. The published
calibration follows a day-to-day random walk; between calibrations the true
error rates drift upward (staleness): e_true = e_pub * exp(kappa_b * h/24),
where h is hours since the last calibration and kappa_b is unknown to the
broker.
"""
import heapq, json, math, pickle, time
from collections import deque
from dataclasses import replace
from pathlib import Path
import numpy as np

from .config import Config

LIB_PATH = Path(__file__).resolve().parents[1] / "data" / "library.pkl"
_LIB = None


def load_library():
    global _LIB
    if _LIB is None:
        with open(LIB_PATH, "rb") as fh:
            _LIB = pickle.load(fh)
    return _LIB


class Device:
    def __init__(self, idx, tab, cfg, rng, n_days, replica=0):
        # key: name of the device model in the library (used for look-ups);
        # name: unique device name (replicas of a model in scalability runs)
        self.idx, self.key, self.tab = idx, tab["name"], tab
        self.name = tab["name"] if replica == 0 else f"{tab['name']}#{replica}"
        self.queue = deque()
        self.busy = False
        self.cur_end = 0.0
        self.svc_ema = cfg.ext_service_mean_s
        self.cal_offset_h = rng.uniform(0, 24)
        self.kappa = rng.uniform(0, cfg.kappa_max)
        self.phase = rng.uniform(0, 2 * math.pi)
        # published calibration for each day: AR(1) random walk on log errors
        z1 = np.zeros(len(tab["e1"])); z2 = np.zeros(len(tab["e2"])); zr = np.zeros(len(tab["ro"]))
        self.pub = []
        for d in range(n_days + 2):
            z1 = cfg.day_rho * z1 + cfg.day_sigma * rng.standard_normal(z1.size)
            z2 = cfg.day_rho * z2 + cfg.day_sigma * rng.standard_normal(z2.size)
            zr = cfg.day_rho * zr + cfg.day_sigma * rng.standard_normal(zr.size)
            b2 = np.where(rng.random(z2.size) < cfg.bad_prob,
                          rng.uniform(*cfg.bad_factor, z2.size), 1.0)
            self.pub.append(dict(
                e1=np.clip(tab["e1"] * np.exp(z1), 1e-6, 0.5),
                e2=np.clip(tab["e2"] * np.exp(z2) * b2, 1e-6, 0.5),
                ro=np.clip(tab["ro"] * np.exp(zr), 1e-6, 0.5)))
        # intra-day staleness path. "exp": log m = kappa*h/24 (monotonic).
        # "rw": log m = kappa*h/24 + rw_sigma * W(h/24), a Brownian path that
        # restarts at each calibration (non-monotonic drift).
        self.shape = cfg.drift_shape
        if self.shape == "rw":
            self.rw = rng.standard_normal((n_days + 2, 97)).cumsum(1) * cfg.rw_sigma * math.sqrt(1.0 / 96)
            self.rw[:, 0] = 0.0; self.rw = np.concatenate([np.zeros((n_days + 2, 1)), self.rw[:, :-1]], 1)

    def log_mult(self, d, age):
        """True/published error multiplier (log) at calibration day d, age h."""
        v = self.kappa * age / 24.0
        if self.shape == "rw":
            x = min(max(age / 24.0 * 96, 0.0), 96.0); i = int(x); fr = x - i
            j = min(i + 1, 96)
            v += (1 - fr) * self.rw[d, i] + fr * self.rw[d, j]
        return min(v, 700.0)   # cap: only reached by extreme backlogs during tuning
    def day_and_age(self, t):
        """Calibration index in force at time t and hours since it was taken."""
        h = t / 3600.0 - self.cal_offset_h
        if h < 0:
            return 0, h + 24.0
        d = int(h // 24.0) + 1
        if d >= len(self.pub):   # predicted start beyond the simulated calibrations: use the last one
            d = len(self.pub) - 1
        return d, h - (d - 1) * 24.0

    def n_jobs(self):
        return len(self.queue) + (1 if self.busy else 0)

    def wait_estimate(self, now):
        rem = max(0.0, self.cur_end - now) if self.busy else 0.0
        return rem + len(self.queue) * self.svc_ema


def esp_of(fp, pub, tab, mult=1.0):
    g, dec = esp_logs(fp, pub, tab, mult)
    return math.exp(g + dec)


def esp_logs(fp, pub, tab, mult=1.0):
    """(log of gate+readout success, log of decoherence factor)."""
    lg = 0.0
    if fp["i1"].size:
        lg += float((fp["c1"] * np.log1p(-np.minimum(pub["e1"][fp["i1"]] * mult, 0.5))).sum())
    if fp["i2"].size:
        lg += float((fp["c2"] * np.log1p(-np.minimum(pub["e2"][fp["i2"]] * mult, 0.5))).sum())
    if fp["im"].size:
        lg += float(np.log1p(-np.minimum(pub["ro"][fp["im"]] * mult, 0.5)).sum())
    dec = -float((fp["dur"] / np.maximum(tab["t2"][fp["used"]], 1e-6)).sum())
    return lg, dec


class Job:
    __slots__ = ("jid", "circ", "shots", "ovh", "arr", "parts", "done_parts",
                 "fid", "logf", "end", "mode", "devs", "measured")

    def __init__(self, jid, circ, shots, ovh, arr, measured):
        self.jid, self.circ, self.shots, self.ovh, self.arr = jid, circ, shots, ovh, arr
        self.parts = 0; self.done_parts = 0; self.fid = 1.0; self.logf = 0.0; self.end = None
        self.mode = None; self.devs = (); self.measured = measured


class Sim:
    def __init__(self, cfg: Config, seed: int, scheduler, lib=None):
        self.cfg, self.seed = cfg, seed
        self.lib = lib or load_library()
        # independent random streams (common random numbers across schedulers):
        # device calibration, broker arrivals, background load per device,
        # fidelity-feedback noise, and the scheduler's own randomness
        tabs = [t for t in self.lib["devices"] if t["name"] not in cfg.exclude_devices]
        fleet = [(t, r) for r in range(cfg.fleet_factor) for t in tabs]
        nd = len(fleet)
        ss = np.random.SeedSequence(seed).spawn(5 + nd)
        self.rng_dev, self.rng_arr, self.rng_fb, self.rng_sched, self.rng_model = (np.random.default_rng(x) for x in ss[:5])
        self.rng_ext = [np.random.default_rng(x) for x in ss[5:]]
        n_days = int(math.ceil(cfg.hours / 24.0)) + 1
        self.devs = [Device(i, tab, cfg, self.rng_dev, n_days, r) for i, (tab, r) in enumerate(fleet)]
        self.by_name = {d.name: d for d in self.devs}
        allc = self.lib["circuits"]
        self.circs = [c for c in allc if (cfg.algos is None or c["algo"] in cfg.algos)
                      and any(k not in cfg.exclude_devices for k in c["dev"])]
        self._merr = {}
        self.dec_times = []
        self.qlen = []
        cm = Path(__file__).resolve().parents[1] / "results" / cfg.cut_model_file
        self.cut_cal = json.loads(cm.read_text()) if cm.exists() else None
        self.cut_pred = self.cut_cal
        if cfg.cut_model_pred_file:   # broker predicts with a different fit (generalisation tests)
            self.cut_pred = json.loads((cm.parent / cfg.cut_model_pred_file).read_text())
        self.sched = scheduler
        self.now = 0.0
        self.ev = []; self._seq = 0
        self.jobs = []
        if hasattr(scheduler, "reset"):
            scheduler.reset(self)

    # ---------- helpers used by schedulers ----------
    def pub_esp(self, fp, dev, t=None, mult=1.0):
        d, _ = dev.day_and_age(self.now if t is None else t)
        return esp_of(fp, dev.pub[d], dev.tab, mult)

    def run_time(self, job, fp, overhead_factor=1):
        return job.shots * overhead_factor * (fp["dur"] + self.cfg.rep_delay_s) + job.ovh

    # ---------- event machinery ----------
    def push(self, t, kind, data):
        self._seq += 1
        heapq.heappush(self.ev, (t, self._seq, kind, data))

    def ext_rate(self, dev, t):
        u = self.cfg.ext_util.get(dev.key, 0.4) * self.cfg.ext_load
        u *= 1.0 + self.cfg.diurnal_amp * math.sin(2 * math.pi * t / 86400.0 + dev.phase)
        return max(u, 1e-6) / self.cfg.ext_service_mean_s

    def ext_service(self, dev):
        m, cv = self.cfg.ext_service_mean_s, self.cfg.ext_service_cv
        s2 = math.log(1 + cv * cv)
        return float(self.rng_ext[dev.idx].lognormal(math.log(m) - s2 / 2, math.sqrt(s2)))

    def enqueue(self, dev, item):
        dev.queue.append(item)
        if not dev.busy:
            self.start_next(dev)

    def start_next(self, dev):
        if not dev.queue:
            dev.busy = False
            return
        item = dev.queue.popleft()
        dev.busy = True
        svc = item[1]
        if item[0] == "part":            # broker job (or fragment): fidelity at start time
            _, _, job, fp = item
            d, age = dev.day_and_age(self.now)
            mult = math.exp(dev.log_mult(d, age))
            g_true, dec = esp_logs(fp, dev.pub[d], dev.tab, mult)
            if self.cfg.model_err_sigma > 0:      # error of the ESP proxy itself (fixed per circuit-device-part)
                g_true *= self.model_error(job.circ, dev.key, round(fp["dur"], 12))
            f_true = math.exp(g_true + dec)
            job.logf += g_true + dec
            # feedback to the broker: noisy observation of the achieved fidelity
            # (e.g. from known-answer checks), delivered with probability fb_prob;
            # the broker knows the published gate/readout log-success and the
            # decoherence factor it computed.
            if hasattr(self.sched, "feedback") and self.rng_fb.random() < self.cfg.fb_prob:
                g_pub, _ = esp_logs(fp, dev.pub[d], dev.tab, 1.0)
                obs = f_true * math.exp(self.cfg.obs_sigma * self.rng_fb.standard_normal())
                uses_fb = getattr(self.sched, "correct", False) and not getattr(self.sched, "oracle", False)
                if uses_fb and self.cfg.fb_probe_cost > 0:   # device time of the probe run(s) that produce feedback
                    svc = svc * (1.0 + self.cfg.fb_probe_cost)
                self.push(self.now + svc + 3600.0 * self.cfg.fb_delay_h, "fb", (dev, age, g_pub, dec, min(obs, 0.999999)))
        dev.cur_end = self.now + svc
        if self.now >= self.cfg.warmup_h * 3600.0:   # busy time inside the measured window
            dev.busy_s = getattr(dev, "busy_s", 0.0) + svc
        self.push(self.now + svc, "done", (dev, item))

    def model_error(self, c, key, tag):
        k = (c, key, tag)
        if k not in self._merr:
            self._merr[k] = math.exp(self.cfg.model_err_sigma * self.rng_model.standard_normal())
        return self._merr[k]

    def cut_quality(self, logf, rec, true=False):
        """Quality of a cut job from the summed log-ESP of its fragments.
        "product": exp(logf) (optimistic proxy).
        "calibrated": exp(logf) * eff(n_sub), eff = c0 - c1*log10(n_sub), fitted from
        full quasi-probability reconstruction runs (data/cut_cost.py), clipped to [0, 1].
        true=True gives the simulator's ground truth (shifted by cut_eff_true_delta);
        true=False the broker's prediction (lowered by cut_eta_z prediction standard errors)."""
        f = math.exp(logf)
        if self.cfg.cut_model == "product" or self.cut_cal is None:
            return f
        cal = self.cut_cal if true else self.cut_pred
        if not true and rec.get("cut_nsub", 12) > cal.get("nsub_max", float("inf")):
            return 0.0   # the broker does not cut outside the calibrated range of sub-experiment counts
        x = math.log10(rec.get("cut_nsub", 12))
        eff = cal["c0"] - cal["c1"] * x
        if true:
            eff += self.cfg.cut_eff_true_delta
        elif self.cfg.cut_eta_z and "resid_sd" in cal:   # uncertainty margin of the broker
            se = cal["resid_sd"] * math.sqrt(1 + 1 / cal["n_circuits"] + (x - cal["lx_mean"]) ** 2 / cal["lx_sxx"])
            eff -= self.cfg.cut_eta_z * se
        return min(1.0, max(0.0, f * min(1.0, eff)))

    def recon_time(self, rec, shots):
        """Classical reconstruction time: linear in the total number of shots of the cut job
        (shots x 9^q), fitted on the reconstruction runs (data/cut_cost.py)."""
        if self.cut_cal is None or self.cfg.cut_model == "product":
            return 0.0
        return max(0.0, self.cut_cal["r0"] + self.cut_cal["r1"] * shots * rec["cut_overhead"] / 1e6)

    def run(self):
        cfg = self.cfg
        H = cfg.hours * 3600.0
        for dev in self.devs:
            self.push(self.rng_ext[dev.idx].exponential(1.0 / self.ext_rate(dev, 0.0)), "ext", dev)
        rate = cfg.base_rate_per_h * cfg.load * cfg.fleet_factor * (len(self.devs) / cfg.fleet_factor / 8.0 if cfg.exclude_devices else 1.0) / 3600.0
        self.push(self.rng_arr.exponential(1.0 / rate), "arr", None)
        jid = 0
        while self.ev:
            t, _, kind, data = heapq.heappop(self.ev)
            if t > H and kind in ("arr", "ext"):
                continue
            self.now = t
            if kind == "ext":
                dev = data
                self.enqueue(dev, ("ext", self.ext_service(dev)))
                self.push(t + self.rng_ext[dev.idx].exponential(1.0 / self.ext_rate(dev, t)), "ext", dev)
            elif kind == "arr":
                c = int(self.rng_arr.integers(len(self.circs)))
                shots = int(self.rng_arr.choice(cfg.shots_choices))
                ovh = float(self.rng_arr.uniform(*cfg.overhead_s))
                job = Job(jid, c, shots, ovh, t, t >= cfg.warmup_h * 3600.0); jid += 1
                self.jobs.append(job)
                self.dispatch(job)
                self.push(t + self.rng_arr.exponential(1.0 / rate), "arr", None)
            elif kind == "done":
                dev, item = data
                dev.svc_ema = 0.98 * dev.svc_ema + 0.02 * item[1]
                if item[0] == "part":
                    job = item[2]
                    job.done_parts += 1
                    if job.done_parts == job.parts:
                        rec = self.circs[job.circ]
                        if job.mode == "whole":
                            job.end, job.fid = t, math.exp(job.logf)
                        else:   # classical reconstruction after both fragments
                            job.end, job.fid = t + self.recon_time(rec, job.shots), self.cut_quality(job.logf, rec, true=True)
                self.start_next(dev)
            elif kind == "fb":
                self.sched.feedback(self, *data)
        return self.summary()

    def dispatch(self, job):
        t0 = time.perf_counter()
        dec = self.sched.decide(self, job)
        self.dec_times.append(time.perf_counter() - t0)
        if job.measured:   # broker queue length seen by this decision (all devices)
            self.qlen.append(sum(len(d.queue) + d.busy for d in self.devs))
        rec = self.circs[job.circ]
        if dec[0] == "whole":
            dev = self.devs[dec[1]]
            fp = rec["dev"][dev.key]
            job.parts, job.mode, job.devs = 1, "whole", (dev.idx,)
            self.enqueue(dev, ("part", self.run_time(job, fp), job, fp))
        else:
            _, a, b = dec
            ov = rec["cut_overhead"]
            job.parts, job.mode, job.devs = 2, "cut", (a, b)
            for k, di in enumerate((a, b)):
                dev = self.devs[di]
                fp = rec["frag"][k]["dev"][dev.key]
                self.enqueue(dev, ("part", self.run_time(job, fp, ov), job, fp))

    def summary(self):
        J = [j for j in self.jobs if j.measured and j.end is not None]
        F = np.array([j.fid for j in J]); T = np.array([j.end - j.arr for j in J])
        dev_share = np.zeros(len(self.devs))
        busy = None
        for j in J:
            for d in j.devs: dev_share[d] += 1.0 / len(j.devs)
        return dict(
            n=len(J), F=F, T=T,
            mean_F=float(F.mean()), mean_T_min=float(T.mean() / 60), p95_T_min=float(np.percentile(T, 95) / 60),
            utility=float((F - self.cfg.w_t_per_min * T / 60).mean()),
            cut_share=float(np.mean([j.mode == "cut" for j in J])),
            dev_share=(dev_share / max(1, len(J))).tolist(),
            unfinished=sum(1 for j in self.jobs if j.measured and j.end is None),
            long_wait_share=float((T > 600).mean()),
            decision_ms=float(1000 * np.mean(self.dec_times)) if self.dec_times else 0.0,
            decision_p99_ms=float(1000 * np.percentile(self.dec_times, 99)) if self.dec_times else 0.0,
            throughput_per_h=len(J) / max(1e-9, self.cfg.hours - self.cfg.warmup_h),
            mean_queue=float(np.mean(self.qlen)) if self.qlen else 0.0,
            util=[min(1.0, getattr(d, "busy_s", 0.0) / ((self.cfg.hours - self.cfg.warmup_h) * 3600.0)) for d in self.devs],
            cut_eligible_share=float(np.mean(["frag" in self.circs[j.circ] for j in J])),
            algo=[self.circs[j.circ]["algo"] for j in J],
            width=np.array([self.circs[j.circ]["width"] for j in J]))


def run_one(cfg, seed, scheduler_factory, **over):
    cfg = replace(cfg, **over) if over else cfg
    return Sim(cfg, seed, scheduler_factory()).run()
