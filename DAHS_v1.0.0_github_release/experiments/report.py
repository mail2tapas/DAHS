"""Print result tables (means +- sd, paired differences with Holm-corrected
Wilcoxon p-values) from the saved CSV files. No simulation is run."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from qcs.runner import RES
from qcs.stats import paired, holm

M = ["mean_F", "mean_T_min", "p95_T_min", "utility", "long_wait_share"]

def table(df, key=None, ref="DAHS"):
    out = []
    groups = [(None, df)] if key is None else list(df.groupby(key))
    for g, d in groups:
        print(f"\n=== {key}={g} ===" if key else "\n===")
        agg = d.groupby("scheduler")[M + ["cut_share"]].agg(["mean", "std"])
        print(agg.round(4).to_string())
        P = d.pivot_table(index="seed", columns="scheduler", values=M)
        rows = []
        for b in d.scheduler.unique():
            if b == ref: continue
            for m in M:
                mean, lo, hi, p = paired(P[m][b], P[m][ref])
                rows.append(dict(group=g, baseline=b, metric=m, diff=mean, lo=lo, hi=hi, p=p))
        R = pd.DataFrame(rows)
        for m in M:
            idx = R.metric == m
            R.loc[idx, "p_holm"] = holm(R.loc[idx, "p"].values)
        print("paired (baseline - DAHS):")
        print(R.round(4).to_string(index=False))
        out.append(R)
    return pd.concat(out)

def load(name):
    if name == "exp7_feedback":
        d = pd.read_csv(RES / "exp7_feedback.csv"); d["grp"] = d.obs_sigma.astype(str) + "_" + d.fb_prob.astype(str); return d
    if name == "exp10a_algo_holdout":
        a = pd.read_csv(RES / "exp10a_algo_holdout_all.csv"); b = pd.read_csv(RES / "exp10a_algo_holdout_train.csv")
        b["scheduler"] = "DAHS-TUNED-ON-OTHER-FAMILIES"; parts = [a, b]
        f = RES / "exp10c_algo_holdout_strict.csv"
        if f.exists():
            c = pd.read_csv(f); c["scheduler"] = "DAHS-STRICT-HOLDOUT"; parts.append(c)
        return pd.concat(parts, ignore_index=True)
    if name == "exp11_lambda":   # lambda = 0.999 is the tuned DAHS (reference of the paired tests)
        d = pd.read_csv(RES / "exp11_lambda.csv")
        d["scheduler"] = d.scheduler.replace({"DAHS-LAMBDA-0.999": "DAHS"})
        d["grp"] = np.where(d.get("drift_shape", pd.Series([None] * len(d))).eq("rw"), "rw0.6", "k" + d.kappa_max.astype(str)); return d
    if name == "exp13_cut_mismatch":
        d = pd.read_csv(RES / "exp13_cut_mismatch.csv")
        d["grp"] = "z" + d.cut_eta_z.astype(str) + "_d" + d.cut_eff_true_delta.astype(str); return d
    if name == "exp12_feedback_cost":
        d = pd.read_csv(RES / "exp12_feedback_cost.csv")
        d["grp"] = d.fb_prob.astype(str) + "_" + d.fb_delay_h.astype(str) + "_" + d.fb_probe_cost.astype(str); return d
    return pd.read_csv(RES / f"{name}.csv")

EXPS = [("exp1_main", "load"), ("exp2_load", "ext_load"), ("exp3_drift", "kappa_max"),
        ("exp4_ablation", None), ("exp5_cutting", "load"), ("exp5_cutting_product", None),
        ("exp6b_w0.03", None), ("exp6b_w0.1", None), ("exp7_feedback", "grp"),
        ("exp8a_proxy_error", "model_err_sigma"), ("exp8b_rw_drift", "rw_sigma"),
        ("exp9_scalability", "fleet_factor"), ("exp10a_algo_holdout", None), ("exp10b_device_holdout", None),
        ("exp11_lambda", "grp"), ("exp12_feedback_cost", "grp"), ("exp13_cut_mismatch", "grp")]

if __name__ == "__main__":
    for name, key in EXPS:
        try:
            d = load(name)
        except FileNotFoundError:
            continue
        print(f"\n################ {name}")
        if "long_wait_share" not in d.columns: d["long_wait_share"] = np.nan
        R = table(d, key)
        R.to_csv(RES / f"{name}_paired.csv", index=False)
