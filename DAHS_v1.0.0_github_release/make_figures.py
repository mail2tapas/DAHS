"""Draw the figures from saved results (no simulation)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"; FIG = RES / "figs"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
COL = {"DAHS": "#0072B2", "GREEDY-UTILITY": "#E69F00", "FID-THRESHOLD": "#009E73",
       "BEST-FIDELITY": "#D55E00", "LEAST-BUSY": "#CC79A7", "MIN-TIME": "#56B4E9", "RANDOM": "#999999",
       "DAHS-NO-CORRECTION": "#882255", "DAHS-ORACLE-DRIFT": "#000000", "DAHS-NO-CUT": "#8c564b",
       "TUNED-UTILITY": "#117733", "PG-LINEAR": "#AA4499", "DAHS-ORACLE-PATH": "#444444"}
MK = {"DAHS": "o", "GREEDY-UTILITY": "s", "FID-THRESHOLD": "^", "BEST-FIDELITY": "v", "LEAST-BUSY": "D",
      "MIN-TIME": "<", "RANDOM": "x", "DAHS-NO-CORRECTION": "P", "DAHS-ORACLE-DRIFT": "*", "DAHS-NO-CUT": "h",
      "TUNED-UTILITY": "d", "PG-LINEAR": "p", "DAHS-ORACLE-PATH": "X"}

def sweep(csv, key, xlabel, fname, metrics=("utility", "mean_F", "mean_T_min"), labels=None):
    d = pd.read_csv(RES / csv); g = d.groupby([key, "scheduler"])
    m, s = g[list(metrics)].mean(), g[list(metrics)].std()
    fig, ax = plt.subplots(1, len(metrics), figsize=(2.3 * len(metrics), 2.2))
    for i, met in enumerate(metrics):
        for sch in d.scheduler.unique():
            x = sorted(d[key].unique()); y = [m.loc[(v, sch), met] for v in x]; e = [s.loc[(v, sch), met] for v in x]
            ax[i].errorbar(x, y, yerr=e, label=sch, color=COL.get(sch), marker=MK.get(sch), ms=3, lw=1, capsize=2)
        ax[i].set_xlabel(xlabel); ax[i].set_ylabel((labels or {}).get(met, met))
        ax[i].set_xticks(sorted(d[key].unique()))
        ax[i].tick_params(axis="x", labelsize=7)
    ax[0].legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)

def scatter_main():
    d = pd.read_csv(RES / "exp1_main.csv"); d = d[d.load == 1.0]
    g = d.groupby("scheduler")[["mean_F", "mean_T_min"]].mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for sch, r in g.iterrows():
        ax.scatter(r.mean_T_min, r.mean_F, color=COL.get(sch), marker=MK.get(sch), s=30, label=sch)
    ax.set_xscale("log"); ax.set_xlabel("Mean turnaround (min, log scale)"); ax.set_ylabel("Mean achieved ESP")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig2_tradeoff.png"); plt.close(fig)

def breakeven():
    o = pd.read_csv(RES / "exp6_breakeven.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(o.w, o.adv, color=COL["DAHS"], marker="o", ms=3)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0.01, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time weight $w_t$ (utility per minute)"); ax.set_ylabel("DAHS utility minus best baseline")
    fig.tight_layout(); fig.savefig(FIG / "fig6_breakeven.png"); plt.close(fig)

def calib_drift_example():
    import sys; sys.path.insert(0, str(RES.parent))
    from qcs.config import Config; from qcs.sim import Sim, esp_of, load_library
    from qcs.schedulers import Random
    sim = Sim(Config(hours=72), 1, Random()); L = load_library()
    c = [x for x in L["circuits"] if x["algo"] == "ghz" and x["width"] == 12][0]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for dev in sim.devs[:6]:
        fp = c["dev"][dev.name]; ts = np.arange(0, 72, 0.25)
        pub, tru = [], []
        for t in ts:
            d, age = dev.day_and_age(t * 3600)
            pub.append(esp_of(fp, dev.pub[d], dev.tab)); tru.append(esp_of(fp, dev.pub[d], dev.tab, np.exp(dev.kappa * age / 24)))
        l, = ax.plot(ts, tru, lw=1, label=dev.name.replace("fake_", ""))
        ax.plot(ts, pub, lw=0.8, ls="--", color=l.get_color())
    ax.set_xlabel("Time (h)"); ax.set_ylabel("ESP of GHZ-12"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig1_drift.png"); plt.close(fig)

def gwo_convergence():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for fn, ls in (("theta.json", "-"), ("theta_gwo1.json", "--"), ("theta_gwo2.json", ":")):
        if not (RES / fn).exists(): continue
        t = json.loads((RES / fn).read_text())
        it = [h[0] + 1 for h in t["history"]]; y = [h[1] for h in t["history"]]
        ax.plot(it, y, ls, color="#0072B2", lw=1.2, label=f"GWO seed {t['gwo_seed']}")
    t = json.loads((RES / "theta_tuned_utility.json").read_text())
    ax.plot([h[0] + 1 for h in t["history"]], [h[1] for h in t["history"]], "-", color="#117733", lw=1.2, label="TUNED-UTILITY (GWO seed 0)")
    ax.set_xlabel("GWO iteration"); ax.set_ylabel("Best training utility"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig7_gwo.png"); plt.close(fig)

def scalability():
    d = pd.read_csv(RES / "exp9_scalability.csv"); d["qpus"] = 8 * d.fleet_factor
    mets = [("decision_ms", "Mean decision time (ms)"), ("throughput_per_h", "Throughput (jobs/h)"),
            ("mean_T_min", "Mean turnaround (min)"), ("mean_queue", "Mean jobs in queues")]
    fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.2))
    for i, (met, lab) in enumerate(mets):
        for sch in d.scheduler.unique():
            g = d[d.scheduler == sch].groupby("qpus")[met]
            ax[i].errorbar(g.mean().index, g.mean().values, yerr=g.std().values, label=sch, color=COL.get(sch), marker=MK.get(sch), ms=3, lw=1, capsize=2)
        ax[i].set_xscale("log", base=2); ax[i].set_xticks([8, 16, 32, 64]); ax[i].set_xticklabels(["8", "16", "32", "64"])
        ax[i].set_xlabel("Number of QPUs"); ax[i].set_ylabel(lab)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig8_scalability.png"); plt.close(fig)

if __name__ == "__main__":
    calib_drift_example()
    if (RES / "exp1_main.csv").exists():
        sweep("exp1_main.csv", "load", "Broker load factor", "fig3_load.png",
              labels={"utility": "Mean utility", "mean_F": "Mean achieved ESP", "mean_T_min": "Mean turnaround (min)"}); scatter_main()
    if (RES / "exp2_load.csv").exists():
        sweep("exp2_load.csv", "ext_load", "Background load factor", "fig4_extload.png",
              labels={"utility": "Mean utility", "mean_F": "Mean achieved ESP", "mean_T_min": "Mean turnaround (min)"})
    if (RES / "exp3_drift.csv").exists():
        sweep("exp3_drift.csv", "kappa_max", r"Max staleness rate $\kappa_{max}$", "fig5_drift.png",
              labels={"utility": "Mean utility", "mean_F": "Mean achieved ESP", "mean_T_min": "Mean turnaround (min)"})
    if (RES / "exp6_breakeven.csv").exists(): breakeven()
    if (RES / "theta_gwo2.json").exists() and (RES / "theta_tuned_utility.json").exists(): gwo_convergence()
    if (RES / "exp9_scalability.csv").exists(): scalability()
    print("figures in", FIG)
