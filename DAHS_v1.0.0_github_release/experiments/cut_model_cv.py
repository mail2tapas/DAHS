"""Predictive validation of the calibrated cut-efficiency model and the
per-circuit whole vs cut statistics. Uses results/exp_cut_repeats.csv (14 circuits
x 5 sampling seeds, 9^q shot budget) and results/cut_model.json (fit on the
five-run means, data/cut_cost.py fit).

1. 95% confidence interval of the slope and 95% prediction intervals of eta.
2. Leave-one-circuit-out and leave-one-family-out cross-validation of the
   predicted output quality Q_cut; compared with the product proxy (eta = 1),
   a constant eta, an extra fragment-ESP predictor and family intercepts.
3. Held-out sampling runs: the model fitted on the means of seeds 11-12 predicts
   every single run of seeds 13-15; coverage of its 95% prediction intervals.
4. Per circuit: paired t-test of Q_cut - Q_whole over the five seeds, with Holm's
   correction over the 14 circuits; sensitivity checks: number of seeds with
   Q_cut > Q_whole, and a Wilcoxon test over the 14 circuit means.
5. A model fitted only on the training families of Exp. 10a
   (results/cut_model_trainfam.json).
    python experiments/cut_model_cv.py   (seconds)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments"))
from cut_cost import fit_model
from qcs.stats import holm


def eff_of(q_cut, esp_cut, alpha):
    return np.clip(1 - (1 - q_cut) / alpha, 0, 1) / esp_cut


def q_of(eff, esp_cut, alpha):
    return 1 - alpha * (1 - np.clip(np.minimum(1, eff) * esp_cut, 0, 1))


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def means(R):
    return R.groupby(["algo", "width"], as_index=False, sort=False).agg(
        n_sub=("n_sub", "first"), cut_gates=("cut_gates", "first"), cut_qpd=("cut_qpd", "first"),
        shots_per_sub=("shots_per_sub", "first"), esp_whole=("esp_whole", "first"), esp_cut=("esp_cut", "first"),
        q_whole=("q_whole", "mean"), q_whole_sd=("q_whole", "std"), q_cut=("q_cut", "mean"), q_cut_sd=("q_cut", "std"),
        recon_s=("recon_s", "mean"))


def pi_halfwidth(m, x):
    n = m["n_circuits"]; tq = stats.t.ppf(0.975, n - 2)
    return tq * m["resid_sd"] * np.sqrt(1 + 1 / n + (x - m["lx_mean"]) ** 2 / m["lx_sxx"])


def main():
    R = pd.read_csv(RES / "exp_cut_repeats.csv")
    M = json.loads((RES / "cut_model.json").read_text()); alpha = M["alpha"]
    G = means(R); n = len(G)
    G["eff"] = eff_of(G.q_cut, G.esp_cut, alpha)
    lx = np.log10(G.n_sub.values); tq = stats.t.ppf(0.975, n - 2)
    out = dict(model=M)
    se_slope = M["resid_sd"] / np.sqrt(M["lx_sxx"])
    out["c1_ci"] = [M["c1"] - tq * se_slope, M["c1"] + tq * se_slope]
    out["eta_zero_at_nsub"] = float(10 ** (M["c0"] / M["c1"])) if M["c1"] > 0 else None
    out["pi"] = {}
    for m_ in (12, 72, 432, 2592):
        x0 = np.log10(m_); yh = M["c0"] - M["c1"] * x0; h = pi_halfwidth(M, x0)
        out["pi"][m_] = dict(eta=float(yh), lo=float(yh - h), hi=float(yh + h), se=float(h / tq),
                             n_circuits=int((G.n_sub == m_).sum()))

    # 2. cross-validation in units of Q
    def design(D, kind):
        x = np.log10(D.n_sub.values)
        return {"logn": np.c_[np.ones(len(D)), x], "const": np.ones((len(D), 1)),
                "logn+esp": np.c_[np.ones(len(D)), x, D.esp_cut.values]}[kind]
    def cv(groups, kind):
        pred = np.zeros(n)
        for g in pd.unique(groups):
            te = groups == g; tr = ~te
            pred[te] = design(G[te], kind) @ ols(design(G[tr], kind), G.eff.values[tr])
        return pred
    def rmse(p, mask=None):
        e = q_of(p, G.esp_cut.values, alpha) - G.q_cut.values
        return float(np.sqrt(np.mean((e if mask is None else e[mask]) ** 2)))
    loo = np.arange(n); fam = G.algo.values
    p_loo = cv(loo, "logn")
    out["cv"] = dict(product_proxy=rmse(np.ones(n)), in_sample=rmse(design(G, "logn") @ ols(design(G, "logn"), G.eff.values)),
                     loo_const=rmse(cv(loo, "const")), loo_logn=rmse(p_loo), lofo_logn=rmse(cv(fam, "logn")),
                     loo_logn_esp=rmse(cv(loo, "logn+esp")))
    keep = G.groupby("algo").algo.transform("size").values >= 2
    fams = sorted(set(fam[keep])); Xf = np.c_[np.array([[f == a for a in fams] for f in fam]).astype(float), lx]
    ef, ep = [], []
    for i in np.where(keep)[0]:
        tr = keep.copy(); tr[i] = False
        pf = Xf[i] @ ols(Xf[tr], G.eff.values[tr]); pp = design(G, "logn")[i] @ ols(design(G[tr], "logn"), G.eff.values[tr])
        ef.append(q_of(pf, G.esp_cut.values[i], alpha) - G.q_cut.values[i]); ep.append(q_of(pp, G.esp_cut.values[i], alpha) - G.q_cut.values[i])
    out["cv"]["loo_family_intercepts"] = float(np.sqrt(np.mean(np.square(ef))))
    out["cv"]["loo_logn_same_subset"] = float(np.sqrt(np.mean(np.square(ep))))
    out["cv"]["n_family_subset"] = int(keep.sum())
    out["cv_by_nsub"] = {int(m_): rmse(p_loo, G.n_sub.values == m_) for m_ in sorted(G.n_sub.unique())}

    # 3. held-out sampling runs: fit on seeds 11-12, test every run of seeds 13-15
    Mtr = fit_model(R[R.seed.isin([11, 12])]); T = R[R.seed.isin([13, 14, 15])].copy()
    xt = np.log10(T.n_sub.values); pred = Mtr["c0"] - Mtr["c1"] * xt
    T["eff"] = eff_of(T.q_cut, T.esp_cut, Mtr["alpha"])
    out["heldout_runs"] = dict(n=len(T), fit_seeds=[11, 12], test_seeds=[13, 14, 15], c0=Mtr["c0"], c1=Mtr["c1"],
        rmse_q=float(np.sqrt(np.mean((q_of(pred, T.esp_cut.values, Mtr["alpha"]) - T.q_cut.values) ** 2))),
        rmse_q_product=float(np.sqrt(np.mean((q_of(np.ones(len(T)), T.esp_cut.values, Mtr["alpha"]) - T.q_cut.values) ** 2))),
        pi_coverage=float(np.mean(np.abs(T.eff.values - pred) <= pi_halfwidth(Mtr, xt))))

    # 4. per-circuit paired t-tests with Holm's correction over the 14 circuits
    P = R.assign(d=R.q_cut - R.q_whole).groupby(["algo", "width"], sort=False).d
    G["q_diff"] = P.mean().values; G["q_diff_sd"] = P.std().values
    G["p"] = [stats.ttest_1samp(v.values, 0.0).pvalue for _, v in P]
    G["p_holm"] = holm(G.p.values)
    G["n_cut_higher"] = P.apply(lambda v: int((v > 0).sum())).values   # seeds (of 5) with Q_cut > Q_whole
    for tag, col in (("raw", "p"), ("holm", "p_holm")):
        G[f"cut_better_{tag}"] = (G.q_diff > 0) & (G[col] < 0.05)
        G[f"whole_better_{tag}"] = (G.q_diff < 0) & (G[col] < 0.05)
    G.to_csv(RES / "exp_cut_repeats_summary.csv", index=False)
    out["repeats"] = dict(n=n, cut_better_mean=int((G.q_diff > 0).sum()),
        cut_better_raw=int(G.cut_better_raw.sum()), whole_better_raw=int(G.whole_better_raw.sum()),
        cut_better_holm=int(G.cut_better_holm.sum()), whole_better_holm=int(G.whole_better_holm.sum()),
        mean_q_whole=float(G.q_whole.mean()), mean_q_cut=float(G.q_cut.mean()),
        mean_sd_q_whole=float(G.q_whole_sd.mean()), mean_sd_q_cut=float(G.q_cut_sd.mean()),
        min_detectable_sd_ratio=float(stats.t.ppf(0.975, 4) / np.sqrt(5)),
        # sensitivity checks for the small number of seeds
        exact_min_p_5=float(2 / 2 ** 5),   # smallest two-sided p of an exact sign-flip / signed-rank test with 5 pairs
        holm_sig_same_sign_all_seeds=int(sum(((r.cut_better_holm and r.n_cut_higher == 5) or (r.whole_better_holm and r.n_cut_higher == 0))
                                             for r in G.itertuples())),
        n_holm_sig=int(G.cut_better_holm.sum() + G.whole_better_holm.sum()),
        wilcoxon_circuits_p=float(stats.wilcoxon(G.q_diff).pvalue),
        ttest_circuits_p=float(stats.ttest_1samp(G.q_diff, 0.0).pvalue))

    # 5. training-family model for the strict hold-out (Exp. 10c)
    from tune import ALGO_SPLIT
    Mf = fit_model(R, families=ALGO_SPLIT["train"])
    (RES / "cut_model_trainfam.json").write_text(json.dumps(Mf, indent=1))
    out["trainfam"] = Mf
    (RES / "cut_model_cv.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("model", "trainfam")}, indent=1, default=float))


if __name__ == "__main__":
    main()
