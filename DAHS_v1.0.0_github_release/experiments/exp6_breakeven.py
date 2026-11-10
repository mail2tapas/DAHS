"""Exp 6: break-even of the time weight w_t (post-hoc, no new runs).
Uses the per-job fidelity and turnaround saved by exp1 (load 1). Decisions
stay fixed; only the utility weight changes."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from qcs.runner import RES

def main():
    J = pd.read_pickle(RES / "exp1_main_jobs.pkl"); J = J[J.load == 1.0]
    ws = np.round(np.concatenate([np.arange(0, 0.02, 0.001), np.arange(0.02, 0.105, 0.005)]), 4)
    rows = []
    for w in ws:
        for _, r in J.iterrows():
            F, T = np.array(r["_F"]), np.array(r["_T"])
            rows.append(dict(w=w, scheduler=r["scheduler"], seed=r["seed"], U=float((F - w * T / 60).mean())))
    D = pd.DataFrame(rows)
    P = D.pivot_table(index=["w", "seed"], columns="scheduler", values="U")
    out = []
    for w in ws:
        m = P.loc[w].mean()
        others = m.drop("DAHS")
        out.append(dict(w=w, DAHS=m["DAHS"], best_other=others.max(), best_other_name=others.idxmax(),
                        adv=m["DAHS"] - others.max()))
    O = pd.DataFrame(out); O.to_csv(RES / "exp6_breakeven.csv", index=False)
    # per-seed break-even vs each strong baseline: smallest/largest w with positive advantage
    be = {}
    for b in ["BEST-FIDELITY", "FID-THRESHOLD", "GREEDY-UTILITY", "LEAST-BUSY"]:
        pos = [w for w in ws if (P.loc[w]["DAHS"] - P.loc[w][b]).mean() > 0]
        be[b] = dict(w_min=min(pos) if pos else None, w_max=max(pos) if pos else None)
    json.dump(be, open(RES / "exp6_breakeven.json", "w"), indent=1)
    print(O.to_string(index=False)); print(be)

if __name__ == "__main__":
    main()
