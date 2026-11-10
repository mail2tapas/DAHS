"""First comparison of whole and cut execution with full reconstruction
(qiskit-addon-cutting), on noisy simulators built from the IBM device
snapshots (Qiskit Aer). It selects the validation circuits (all cuttable
families, their three smallest cuttable widths with at most 432 sub-experiments) and their devices (best
snapshot ESP for the whole circuit and for each fragment), and records the
ESP proxies. The per-circuit results and the cut model use the five seeded repetitions of
experiments/exp_cut_repeats.py; the quality columns of this script's output are
superseded by them.

Quality metric: Q = 1 - mean |<O>_est - <O>_ideal| over observables
O in {Z_i} U {Z_i Z_(i+1)} (expectation values are what cutting reconstructs).
Shot budget: S shots for whole execution and S * 9^q shots in total for cut
execution (q = cut CNOTs after decomposition, library field cut_overhead = 9^q),
spread evenly over the sub-experiments with at least 64 each.
    python experiments/exp_cut_validation.py          (selection only, seconds; writes the CSV used by exp_cut_repeats.py)
    python experiments/exp_cut_validation.py --run    (also one seeded whole/cut run per circuit, ~1 h)
"""
import sys, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, PauliList
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider as fp
from mqt.bench import BenchmarkLevel
from qcs.bench import get_benchmark   # graph state with a fixed seed
from qiskit_addon_cutting import partition_problem, generate_cutting_experiments, reconstruct_expectation_values
from qcs.sim import load_library
from qcs.runner import RES

S = 4000
FAKE = {"fake_torino": "FakeTorino", "fake_fez": "FakeFez", "fake_brisbane": "FakeBrisbane",
        "fake_sherbrooke": "FakeSherbrooke", "fake_kyiv": "FakeKyiv", "fake_prague": "FakePrague",
        "fake_kolkata": "FakeKolkataV2", "fake_guadalupe": "FakeGuadalupeV2"}
_BK = {}
def backend(name):
    if name not in _BK:
        b = getattr(fp, FAKE[name])()
        _BK[name] = (b, AerSimulator.from_backend(b))
    return _BK[name]


def observables(n):
    labs = []
    for i in range(n):
        p = ["I"] * n; p[n - 1 - i] = "Z"; labs.append("".join(p))
    for i in range(n - 1):
        p = ["I"] * n; p[n - 1 - i] = "Z"; p[n - 2 - i] = "Z"; labs.append("".join(p))
    return PauliList(labs)


def expvals_from_counts(counts, n, obs):
    """Expectation of Z-type Paulis from measured bitstrings (qubit i = bit i)."""
    tot = sum(counts.values()); out = []
    for P in obs:
        zs = [i for i in range(n) if P.z[i]]
        v = 0.0
        for bits, c in counts.items():
            b = bits.replace(" ", "")[::-1]
            par = sum(int(b[i]) for i in zs) & 1
            v += c * (1 - 2 * par)
        out.append(v / tot)
    return np.array(out)


def run_whole(circ, dev, obs, shots):
    b, sim = backend(dev)
    c = circ.copy(); c.measure_all()
    tc = transpile(c, b, optimization_level=1, seed_transpiler=7)
    res = sim.run(tc, shots=shots, seed_simulator=11).result()
    return expvals_from_counts(res.get_counts(), circ.num_qubits, obs)


def run_cut(circ, k, devs, obs, shots_total):
    n = circ.num_qubits
    labels = "".join("A" if q < k else "B" for q in range(n))
    part = partition_problem(circuit=circ, partition_labels=labels, observables=obs)
    subexp, coeffs = generate_cutting_experiments(circuits=part.subcircuits, observables=part.subobservables,
                                                  num_samples=np.inf)
    n_sub = sum(len(v) for v in subexp.values())
    per = max(64, int(shots_total / n_sub))
    results = {}
    t_q = 0.0
    for lab, circs in subexp.items():
        dev = devs[0] if lab == "A" else devs[1]
        b, sim = backend(dev)
        tcs = transpile(circs, b, optimization_level=1, seed_transpiler=7)
        sampler = SamplerV2(options=dict(backend_options=dict(noise_model=NoiseModel.from_backend(b)),
                                         run_options=dict(seed_simulator=11)))
        t0 = time.time()
        results[lab] = sampler.run(tcs, shots=per).result()
        t_q += time.time() - t0
    t0 = time.time()
    vals = reconstruct_expectation_values(results, coeffs, part.subobservables)
    t_rec = time.time() - t0
    return np.array(vals), n_sub, per, len(part.bases), t_rec


def main(run=False):
    L = load_library()
    rows = []
    # circuits with more than 432 sub-experiments (q >= 4) need >= 2.6e7 shots at the 9^q budget,
    # too slow to simulate on a laptop; the broker does not cut outside the calibrated range
    cands = [c for c in L["circuits"] if "frag" in c and c["width"] <= 20 and c.get("cut_nsub", 0) <= 432]
    # one or two widths per family to keep the run laptop-sized
    pick = {}
    for c in sorted(cands, key=lambda c: (c["algo"], c["width"])):
        pick.setdefault(c["algo"], [])
        if len(pick[c["algo"]]) < 3: pick[c["algo"]].append(c)
    t_start = time.time()
    for algo, lst in pick.items():
        for c in lst:
            n = c["width"]; k = c["cut_k"]
            base = get_benchmark(algo, BenchmarkLevel.INDEP, n)
            base.remove_final_measurements(inplace=True)
            base = transpile(base, basis_gates=["cx", "u"], optimization_level=0)
            obs = observables(n)
            sv = Statevector(base)
            ex_ideal = np.array([sv.expectation_value(SparsePauliOp(p)).real for p in obs])
            # devices: best whole device, best fragment devices (snapshot ESP, as the scheduler would)
            dW = max(c["dev"], key=lambda d: c["dev"][d]["esp0"])
            dA = max(c["frag"][0]["dev"], key=lambda d: c["frag"][0]["dev"][d]["esp0"])
            dB = max(c["frag"][1]["dev"], key=lambda d: c["frag"][1]["dev"][d]["esp0"])
            espW = c["dev"][dW]["esp0"]; espC = c["frag"][0]["dev"][dA]["esp0"] * c["frag"][1]["dev"][dB]["esp0"]
            row = dict(algo=algo, width=n, cut_gates=c["cut_gates"], cut_qpd=c["cut_qpd"], overhead=c["cut_overhead"],
                       dev_whole=dW, dev_a=dA, dev_b=dB, esp_whole=espW, esp_cut=espC, n_subexperiments=c["cut_nsub"],
                       shots_per_sub=max(64, int(S * c["cut_overhead"] / c["cut_nsub"])))
            if not run:
                rows.append(row); print(f"{algo:12s} n={n:2d} q={c['cut_qpd']} W:{dW[5:]} C:{dA[5:]}/{dB[5:]}", flush=True); continue
            try:
                ew = run_whole(base, dW, obs, S)
                ec, n_sub, per, nb, t_rec = run_cut(base, k, (dA, dB), obs, S * c["cut_overhead"])
            except Exception as ex:
                print("skip", algo, n, ex); continue
            qW = 1 - np.mean(np.abs(ew - ex_ideal)); qC = 1 - np.mean(np.abs(ec - ex_ideal))
            rows.append(dict(row, q_whole=qW, q_cut=qC, n_subexperiments=n_sub, shots_per_sub=per, recon_s=t_rec))
            print(f"{algo:12s} n={n:2d} g={c['cut_gates']} W:{dW[5:]:10s} ESP={espW:.3f} Q={qW:.3f} | "
                  f"C:{dA[5:]}/{dB[5:]} ESPprod={espC:.3f} Q={qC:.3f} subexp={n_sub} recon={t_rec:.2f}s "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    df = pd.DataFrame(rows); RES.mkdir(exist_ok=True)
    df.to_csv(RES / "exp_cut_validation.csv", index=False)
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main(run="--run" in sys.argv)
