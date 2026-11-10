"""Build the circuit-device library used by the simulator.

For every benchmark circuit (MQT Bench, target-independent level) and every
device model (Qiskit fake IBM backends with real calibration snapshots), the
circuit is transpiled once and the following are stored:
  - gate counts per physical qubit (1q) and per coupler (2q), measured qubits
  - estimated circuit duration (s)
  - base calibration tables of each device (errors, T1, T2)
It also runs a gate-cut feasibility analysis (contiguous bipartition).

Run once:  python data/build_data.py      (about 5-10 minutes on a laptop CPU)
"""
import math, pickle, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

warnings.filterwarnings("ignore")
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider as fp
from mqt.bench import BenchmarkLevel
from qcs.bench import get_benchmark   # graph state with a fixed seed

OUT = Path(__file__).resolve().parent / "library.pkl"

DEVICES = ["FakeTorino", "FakeFez", "FakeBrisbane", "FakeSherbrooke",
           "FakeKyiv", "FakePrague", "FakeKolkataV2", "FakeGuadalupeV2"]
ALGOS = ["ghz", "bv", "dj", "graphstate", "wstate", "qaoa", "qft",
         "qftentangled", "vqe_real_amp", "vqe_su2", "vqe_two_local", "qnn",
         "ae", "randomcircuit", "bmw_quark_cardinality", "cdkm_ripple_carry_adder"]
WIDTHS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28]
ONEQ = {"sx", "x"}            # rz is virtual (zero error, zero time)
TWOQ = {"ecr", "cz", "cx"}
MIN_BEST_ESP = 0.01          # drop circuits that give no signal on any device
MAX_CUTS = 2                 # at most two gate cuts (sampling overhead 9^2 = 81)


def device_tables(b):
    t = b.target
    n = b.num_qubits
    two = [g for g in ("ecr", "cz", "cx") if g in t.operation_names][0]
    e1 = np.zeros(n); ro = np.zeros(n); t1 = np.zeros(n); t2 = np.zeros(n)
    for q in range(n):
        p = t["sx"].get((q,)); e1[q] = p.error if p and p.error is not None else 1e-3
        p = t["measure"].get((q,)); ro[q] = p.error if p and p.error is not None else 0.02
        qp = b.qubit_properties(q)
        t1[q] = qp.t1 if qp and qp.t1 else 1e-4
        t2[q] = qp.t2 if qp and qp.t2 else 1e-4
    edges, e2 = [], []
    for (a, c), p in t[two].items():
        edges.append((a, c)); e2.append(p.error if p and p.error is not None else 0.02)
    e2 = np.array(e2)
    e2 = np.clip(e2, 0, 0.5)          # a few snapshot entries are 1.0 (dead couplers)
    return dict(name=b.name, n=n, two=two, e1=e1, ro=np.clip(ro, 0, 0.5), t1=t1, t2=t2,
                edges=edges, edge_index={e: i for i, e in enumerate(edges)}, e2=e2)


def footprint(tc, dev):
    """Gate counts on physical elements of a transpiled circuit."""
    n1, n2, meas = {}, {}, []
    for inst in tc.data:
        name = inst.operation.name
        qs = [tc.find_bit(q).index for q in inst.qubits]
        if name in ONEQ:
            n1[qs[0]] = n1.get(qs[0], 0) + 1
        elif name in TWOQ:
            k = dev["edge_index"].get((qs[0], qs[1]))
            if k is None: k = dev["edge_index"].get((qs[1], qs[0]))
            n2[k] = n2.get(k, 0) + 1
        elif name == "measure":
            meas.append(qs[0])
    used = sorted(set(list(n1) + meas + [q for k in n2 for q in dev["edges"][k]]))
    return dict(i1=np.array(list(n1), int), c1=np.array(list(n1.values()), float),
                i2=np.array(list(n2), int), c2=np.array(list(n2.values()), float),
                im=np.array(meas, int), used=np.array(used, int))


def esp(fpnt, dev, mult=1.0, dur=0.0):
    e1 = np.clip(dev["e1"][fpnt["i1"]] * mult, 0, 0.5)
    e2 = np.clip(dev["e2"][fpnt["i2"]] * mult, 0, 0.5)
    ro = np.clip(dev["ro"][fpnt["im"]] * mult, 0, 0.5)
    lg = (fpnt["c1"] * np.log1p(-e1)).sum() + (fpnt["c2"] * np.log1p(-e2)).sum() + np.log1p(-ro).sum()
    t2 = dev["t2"][fpnt["used"]]
    lg -= (dur / np.maximum(t2, 1e-6)).sum()
    return math.exp(lg)


def cut_analysis(c):
    """Best contiguous bipartition: number of 2q gates crossing it."""
    n = c.num_qubits
    best = None
    for k in range(max(2, n // 2 - 2), min(n - 1, n // 2 + 2) + 1):
        g = 0
        for inst in c.data:
            qs = [c.find_bit(q).index for q in inst.qubits]
            if len(qs) >= 2 and len({q < k for q in qs}) == 2:
                g += 1
        if best is None or g < best[1]: best = (k, g)
    return best


def fragments(c, k):
    """Split a (measurement-free) circuit into qubit blocks [0,k) and [k,n);
    gates that cross the cut are removed (they are replaced by quasi-probability
    sampling, which multiplies the shot count by 9 per cut CNOT/CZ)."""
    from qiskit import QuantumCircuit
    n = c.num_qubits
    blocks = [list(range(0, k)), list(range(k, n))]
    out = []
    for blk in blocks:
        pos = {q: i for i, q in enumerate(blk)}
        f = QuantumCircuit(len(blk))
        for inst in c.data:
            qs = [c.find_bit(q).index for q in inst.qubits]
            if all(q in pos for q in qs) and not inst.clbits:
                f.append(inst.operation, [pos[q] for q in qs])
        f.measure_all()
        out.append(f)
    return out


def main():
    t0 = time.time()
    devs = []
    backs = []
    for d in DEVICES:
        b = getattr(fp, d)(); backs.append(b); devs.append(device_tables(b))
        print(f"device {b.name}: {b.num_qubits} qubits, 2q={devs[-1]['two']}, "
              f"median 2q err={np.median(devs[-1]['e2']):.4f}", flush=True)
    circuits = []
    for a in ALGOS:
        for w in WIDTHS:
            try:
                c = get_benchmark(a, BenchmarkLevel.INDEP, w)
            except Exception:
                continue
            c.remove_final_measurements(inplace=True)
            k, g = cut_analysis(c)
            c.measure_all()
            rec = dict(algo=a, width=c.num_qubits - 0, cut_k=k, cut_gates=g, dev={})
            for b, dv in zip(backs, devs):
                if c.num_qubits > b.num_qubits: continue
                tc = transpile(c, b, optimization_level=1, seed_transpiler=7)
                dur = tc.estimate_duration(b.target, unit="s")
                f = footprint(tc, dv)
                f["dur"] = float(dur)
                f["two_q"] = float(f["c2"].sum())
                f["esp0"] = esp(f, dv, 1.0, dur)
                rec["dev"][dv["name"]] = f
            if g <= MAX_CUTS and c.num_qubits >= 8:
                base = c.copy(); base.remove_final_measurements(inplace=True)
                rec["frag"] = []
                for f in fragments(base, k):
                    fd = {}
                    for b, dv in zip(backs, devs):
                        if f.num_qubits > b.num_qubits: continue
                        tf = transpile(f, b, optimization_level=1, seed_transpiler=7)
                        dur = tf.estimate_duration(b.target, unit="s")
                        ff = footprint(tf, dv); ff["dur"] = float(dur)
                        ff["esp0"] = esp(ff, dv, 1.0, dur)
                        fd[dv["name"]] = ff
                    rec["frag"].append(dict(width=f.num_qubits, dev=fd))
                rec["cut_overhead"] = 9 ** g   # provisional; data/cut_cost.py library sets 9^q (cut CNOTs after decomposition)
            best = max(v["esp0"] for v in rec["dev"].values())
            if best < MIN_BEST_ESP:
                continue
            circuits.append(rec)
            print(f"{a:24s} w={w:3d} devices={len(rec['dev'])} best ESP={best:.3f} "
                  f"cut gates={g}", flush=True)
    lib = dict(devices=devs, circuits=circuits, meta=dict(
        devices=DEVICES, algos=ALGOS, widths=WIDTHS, min_best_esp=MIN_BEST_ESP, max_cuts=MAX_CUTS,
        build_seconds=time.time() - t0))
    with open(OUT, "wb") as fh:
        pickle.dump(lib, fh)
    print(f"saved {len(circuits)} circuits x {len(devs)} devices to {OUT} "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
