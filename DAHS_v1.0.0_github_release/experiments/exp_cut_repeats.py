"""Repeated reconstruction runs for all 14 validation circuits (per-circuit
whole vs cut quality, and the data the cut-efficiency model is fitted on). Whole and cut
execution of every circuit of exp_cut_validation.py are run with five sampling
seeds (11-15; exactly repeatable). The device choice and the observables are
those of exp_cut_validation.py. Shot budget: S = 4000 shots for whole
execution and S * 9^q shots in total for cut execution, where q is the number
of cut CNOTs after decomposition (library field cut_qpd), spread evenly over
the n_sub = 2 * 6^q sub-experiments with at least 64 shots each.

    python experiments/exp_cut_repeats.py                 (all 14 circuits; ~1.5 h on one core)
    python experiments/exp_cut_repeats.py dj:10 ghz:8   (only these, into part files)
    python experiments/exp_cut_repeats.py dj:10@13,14           (only these seeds)
    python experiments/exp_cut_repeats.py merge                 (merge the part files into the CSV)
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
from experiments.exp_cut_validation import backend, observables, S
from experiments.exp_cut_uncertainty import whole
from qcs.sim import load_library
from qcs.runner import RES

SEEDS = (11, 12, 13, 14, 15)
import os
PREFIX = os.environ.get("CUTREP_PREFIX", "part")   # CUTREP_PREFIX=verify writes exp_cut_repeats_verify_*.csv


def cut(circ, k, devs, obs, shots_total, seed):
    """Cut execution with full reconstruction; returns (expectation values, shots per
    sub-experiment, classical reconstruction time in s)."""
    n = circ.num_qubits
    part = partition_problem(circuit=circ, partition_labels="".join("A" if q < k else "B" for q in range(n)), observables=obs)
    sub, coeffs = generate_cutting_experiments(circuits=part.subcircuits, observables=part.subobservables, num_samples=np.inf)
    per = max(64, int(shots_total / sum(len(v) for v in sub.values())))
    res = {}
    for lab, circs in sub.items():
        b, _ = backend(devs[0] if lab == "A" else devs[1])
        sampler = SamplerV2(options=dict(backend_options=dict(noise_model=NoiseModel.from_backend(b)),
                                         run_options=dict(seed_simulator=seed)))
        res[lab] = sampler.run(transpile(circs, b, optimization_level=1, seed_transpiler=7), shots=per).result()
    t0 = time.time()
    vals = reconstruct_expectation_values(res, coeffs, part.subobservables)
    return np.array(vals), per, time.time() - t0


def main(only=None):
    L = load_library()
    V = pd.read_csv(RES / "exp_cut_validation.csv")
    out = RES / "exp_cut_repeats.csv"
    old = pd.read_csv(out) if (only and out.exists()) else None
    rows = []; t0 = time.time()
    seeds = SEEDS
    if only and any("@" in x for x in only):   # e.g. dj:10@13,14 runs only these seeds
        seeds = tuple(int(z) for z in only[0].split("@")[1].split(","))
        only = [x.split("@")[0] for x in only]
    tag = "_".join(x.replace(":", "") for x in only) + ("_s" + "".join(str(z) for z in seeds) if seeds != SEEDS else "") if only else ""
    for _, v in V.iterrows():
        algo, n = v.algo, int(v.width)
        if only and f"{algo}:{n}" not in only:
            continue
        c = next(x for x in L["circuits"] if x["algo"] == algo and x["width"] == n and "frag" in x)
        base = get_benchmark(algo, BenchmarkLevel.INDEP, n); base.remove_final_measurements(inplace=True)
        base = transpile(base, basis_gates=["cx", "u"], optimization_level=0)
        obs = observables(n); sv = Statevector(base)
        ideal = np.array([sv.expectation_value(SparsePauliOp(p)).real for p in obs])
        for s in seeds:
            ew = whole(base, v.dev_whole, obs, s)
            ec, per, t_rec = cut(base, c["cut_k"], (v.dev_a, v.dev_b), obs, S * c["cut_overhead"], s)
            rows.append(dict(algo=algo, width=n, seed=s, n_sub=int(c["cut_nsub"]), cut_gates=int(c["cut_gates"]),
                             cut_qpd=int(c["cut_qpd"]), shots_per_sub=per, esp_whole=v.esp_whole, esp_cut=v.esp_cut,
                             q_whole=1 - np.abs(ew - ideal).mean(), q_cut=1 - np.abs(ec - ideal).mean(), recon_s=t_rec))
            print(f"  {algo} n={n} seed={s} ({time.time()-t0:.0f}s)", flush=True)
            if only:   # checkpoint after every seed
                pd.DataFrame(rows).to_csv(RES / f"exp_cut_repeats_{PREFIX}_{tag}.csv", index=False)
        r = pd.DataFrame(rows[-len(seeds):])
        print(f"{algo:11s} n={n:2d} n_sub={int(c['cut_nsub']):5d} shots/sub={per} Qw={r.q_whole.mean():.4f}+-{r.q_whole.std():.4f} "
              f"Qc={r.q_cut.mean():.4f}+-{r.q_cut.std():.4f} ({time.time()-t0:.0f}s)", flush=True)
    new = pd.DataFrame(rows)
    if old is not None:
        keep = ~old.apply(lambda x: f"{x.algo}:{int(x.width)}" in only, axis=1)
        new = pd.concat([old[keep], new], ignore_index=True)
    order = {(a, w): i for i, (a, w) in enumerate(zip(V.algo, V.width))}
    new = new.sort_values(by=["algo", "width", "seed"], key=lambda col: col if col.name == "seed" else col).reset_index(drop=True)
    new["_o"] = [order[(a, w)] for a, w in zip(new.algo, new.width)]
    new = new.sort_values(["_o", "seed"]).drop(columns="_o")
    if only:   # write the subset to its own file; merge with merge_parts() when all parts are done
        pd.DataFrame(rows).to_csv(RES / f"exp_cut_repeats_{PREFIX}_{tag}.csv", index=False)
    else:
        new.to_csv(out, index=False)


def merge_parts():
    """Replace the circuits of every results/exp_cut_repeats_part_*.csv in exp_cut_repeats.csv."""
    out = RES / "exp_cut_repeats.csv"; base = pd.read_csv(out)
    parts = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("exp_cut_repeats_part_*.csv"))], ignore_index=True)
    parts = parts.drop_duplicates(["algo", "width", "seed"], keep="last")
    keys = set(zip(parts.algo, parts.width))
    base = pd.concat([base[[(a, w) not in keys for a, w in zip(base.algo, base.width)]], parts], ignore_index=True)
    V = pd.read_csv(RES / "exp_cut_validation.csv")
    # circuits that were not rerun (q = g, so their budget did not change) carry the shots per
    # sub-experiment and reconstruction time of exp_cut_validation.csv, which used the same budget
    L = load_library(); lib = {(c["algo"], c["width"]): c for c in L["circuits"] if "frag" in c}
    vv = V.set_index(["algo", "width"])
    for col, src in (("shots_per_sub", "shots_per_sub"), ("recon_s", "recon_s")):
        if col not in base: base[col] = np.nan
        if src not in V or not base[col].isna().any(): continue
        m = base[col].isna()
        base.loc[m, col] = [vv.loc[(a, w), src] for a, w in zip(base.algo[m], base.width[m])]
    for col in ("cut_gates", "cut_qpd"):
        base[col] = [lib[(a, w)][col] for a, w in zip(base.algo, base.width)]
    base["n_sub"] = [lib[(a, w)]["cut_nsub"] for a, w in zip(base.algo, base.width)]
    order = {(a, w): i for i, (a, w) in enumerate(zip(V.algo, V.width))}
    base["_o"] = [order[(a, w)] for a, w in zip(base.algo, base.width)]
    base.sort_values(["_o", "seed"]).drop(columns="_o").to_csv(out, index=False)
    print("merged", len(base), "rows")


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge_parts()
    else:
        main(sys.argv[1:] or None)
