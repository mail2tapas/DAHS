"""Add the real cutting cost to the library and fit the calibrated cut-quality model.

1. For every cuttable circuit, count the sub-experiments that qiskit-addon-cutting
   generates for the same bipartition and the Z / ZZ observables, and the number
   q of cut CNOTs after decomposition (n_sub = 2 * 6^q). The shot overhead of cut
   execution is set to 9^q (cut_overhead), in the scheduler and in the validation runs.
       python data/cut_cost.py library
2. From the repeated reconstruction runs (results/exp_cut_repeats.csv, five sampling
   seeds per circuit, 9^q budget), fit on the five-run means
       eta = ESP_eq(cut) / ESP_product(fragments)  ~  c0 - c1*log10(n_sub),
   where ESP_eq converts the measured quality Q to the ESP scale through the
   whole-circuit fit 1-Q = alpha (1-ESP), and the classical reconstruction time
   t = r0 + r1 * (total shots of the cut job / 10^6), which grows with the shots to process. The fit statistics needed for prediction intervals
   (residual sd, mean and spread of log10 n_sub) are stored as well.
       python data/cut_cost.py fit
The model is written to results/cut_model.json and read by the simulator.
"""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
LIB = ROOT / "data" / "library.pkl"; RES = ROOT / "results"


def library():
    from qiskit import transpile
    from mqt.bench import BenchmarkLevel
    from qcs.bench import get_benchmark   # graph state with a fixed seed
    from qiskit_addon_cutting import partition_problem, generate_cutting_experiments
    from experiments.exp_cut_validation import observables
    L = pickle.load(open(LIB, "rb"))
    for c in L["circuits"]:
        if "frag" not in c:
            continue
        n, k = c["width"], c["cut_k"]
        base = get_benchmark(c["algo"], BenchmarkLevel.INDEP, n)
        base.remove_final_measurements(inplace=True)
        base = transpile(base, basis_gates=["cx", "u"], optimization_level=0)
        part = partition_problem(circuit=base, partition_labels="".join("A" if q < k else "B" for q in range(n)),
                                 observables=observables(n))
        q = len(part.bases)
        if q <= 3:   # enumerate explicitly; this confirms the closed form below
            sub, _ = generate_cutting_experiments(circuits=part.subcircuits, observables=part.subobservables,
                                                  num_samples=np.inf)
            c["cut_nsub"] = int(sum(len(v) for v in sub.values()))
            assert c["cut_nsub"] == len(part.subcircuits) * 6 ** q, (c["cut_nsub"], q)
        else:        # 6 QPD terms per cut CNOT, one copy per fragment
            c["cut_nsub"] = int(len(part.subcircuits) * 6 ** q)
        c["cut_qpd"] = q
        c["cut_overhead"] = 9 ** q   # sampling overhead of q cut CNOTs
        print(f"{c['algo']:12s} n={n:2d} g={c['cut_gates']} q={q} subexperiments={c['cut_nsub']} overhead={9**q}")
    pickle.dump(L, open(LIB, "wb"))


def fit_model(R, families=None):
    """Fit the cut model on per-circuit means of the repeated runs (optionally a subset of families)."""
    if families is not None:
        R = R[R.algo.isin(families)]
    G = R.groupby(["algo", "width"], as_index=False).agg(n_sub=("n_sub", "first"), esp_whole=("esp_whole", "first"),
        esp_cut=("esp_cut", "first"), q_whole=("q_whole", "mean"), q_cut=("q_cut", "mean"), recon_s=("recon_s", "mean"),
        shots_per_sub=("shots_per_sub", "first"))
    G["mshots"] = G.shots_per_sub * G.n_sub / 1e6   # total shots processed by the reconstruction, in millions
    x = 1 - G.esp_whole; y = 1 - G.q_whole
    alpha = float((x * y).sum() / (x * x).sum())
    eff = np.clip(1 - (1 - G.q_cut) / alpha, 0.0, 1.0) / G.esp_cut
    lx = np.log10(G.n_sub)
    slope, c0 = np.polyfit(lx, eff, 1)
    res = eff - (c0 + slope * lx); n = len(G)
    r1, r0 = np.polyfit(G.mshots, G.recon_s, 1)    # t_rec = r0 + r1 * (total cut shots / 1e6)
    return dict(alpha=alpha, c0=float(c0), c1=float(-slope), r0=float(r0), r1=float(r1), n_circuits=n,
                resid_sd=float(np.sqrt((res ** 2).sum() / (n - 2))), lx_mean=float(lx.mean()),
                lx_sxx=float(((lx - lx.mean()) ** 2).sum()),
                eff_fit_r2=float(1 - (res ** 2).sum() / ((eff - eff.mean()) ** 2).sum()),
                mean_q_whole=float(G.q_whole.mean()), mean_q_cut=float(G.q_cut.mean()),
                corr_esp_q_whole=float(np.corrcoef(G.esp_whole, G.q_whole)[0, 1]),
                families=sorted(G.algo.unique().tolist()), nsub_max=int(G.n_sub.max()))


def fit():
    R = pd.read_csv(RES / "exp_cut_repeats.csv")
    m = fit_model(R)
    (RES / "cut_model.json").write_text(json.dumps(m, indent=1))
    print(m)


if __name__ == "__main__":
    {"library": library, "fit": fit}[sys.argv[1] if len(sys.argv) > 1 else "fit"]()
