"""Statistical uncertainty of whole and cut execution (complements
exp_cut_validation.py). For five representative cuttable circuits, whole and
cut execution are repeated with five sampling seeds; the spread of the
estimated expectation values and of the quality Q is reported.
    python experiments/exp_cut_uncertainty.py      (~5 min on 2 cores)
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from qiskit import transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.noise import NoiseModel
from mqt.bench import BenchmarkLevel
from qcs.bench import get_benchmark   # graph state with a fixed seed
from qiskit_addon_cutting import partition_problem, generate_cutting_experiments, reconstruct_expectation_values
from experiments.exp_cut_validation import backend, observables, expvals_from_counts, S
from qcs.sim import load_library
from qcs.runner import RES

CASES = [("ghz", 10), ("bv", 8), ("qnn", 10), ("wstate", 8), ("dj", 8)]
SEEDS = (11, 12, 13, 14, 15)


def whole(circ, dev, obs, seed):
    b, sim = backend(dev)
    c = circ.copy(); c.measure_all()
    tc = transpile(c, b, optimization_level=1, seed_transpiler=7)
    return expvals_from_counts(sim.run(tc, shots=S, seed_simulator=seed).result().get_counts(), circ.num_qubits, obs)


def cut(circ, k, devs, obs, shots_total, seed):
    n = circ.num_qubits
    part = partition_problem(circuit=circ, partition_labels="".join("A" if q < k else "B" for q in range(n)), observables=obs)
    sub, coeffs = generate_cutting_experiments(circuits=part.subcircuits, observables=part.subobservables, num_samples=np.inf)
    per = max(64, int(shots_total / sum(len(v) for v in sub.values())))
    res = {}
    for lab, circs in sub.items():
        b, _ = backend(devs[0] if lab == "A" else devs[1])
        sampler = SamplerV2(options=dict(backend_options=dict(noise_model=NoiseModel.from_backend(b)), run_options=dict(seed_simulator=seed)))
        res[lab] = sampler.run(transpile(circs, b, optimization_level=1, seed_transpiler=7), shots=per).result()
    return np.array(reconstruct_expectation_values(res, coeffs, part.subobservables))


def main():
    L = load_library(); rows = []
    for algo, n in CASES:
        c = next(x for x in L["circuits"] if x["algo"] == algo and x["width"] == n and "frag" in x)
        base = get_benchmark(algo, BenchmarkLevel.INDEP, n); base.remove_final_measurements(inplace=True)
        base = transpile(base, basis_gates=["cx", "u"], optimization_level=0)
        obs = observables(n); sv = Statevector(base)
        ideal = np.array([sv.expectation_value(SparsePauliOp(p)).real for p in obs])
        dW = max(c["dev"], key=lambda d: c["dev"][d]["esp0"])
        dA = max(c["frag"][0]["dev"], key=lambda d: c["frag"][0]["dev"][d]["esp0"])
        dB = max(c["frag"][1]["dev"], key=lambda d: c["frag"][1]["dev"][d]["esp0"])
        EW = np.array([whole(base, dW, obs, s) for s in SEEDS])
        EC = np.array([cut(base, c["cut_k"], (dA, dB), obs, S * c["cut_overhead"], s) for s in SEEDS])
        qW = 1 - np.abs(EW - ideal).mean(1); qC = 1 - np.abs(EC - ideal).mean(1)
        rows.append(dict(algo=algo, width=n, n_sub=int(c.get("cut_nsub", 0)),
                         sd_expval_whole=float(EW.std(0, ddof=1).mean()), sd_expval_cut=float(EC.std(0, ddof=1).mean()),
                         q_whole_mean=float(qW.mean()), q_whole_sd=float(qW.std(ddof=1)),
                         q_cut_mean=float(qC.mean()), q_cut_sd=float(qC.std(ddof=1)),
                         q_diff_mean=float((qC - qW).mean()), q_diff_sd=float((qC - qW).std(ddof=1))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "exp_cut_uncertainty.csv", index=False)


if __name__ == "__main__":
    main()
