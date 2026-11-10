"""Exp 5: circuit cutting. (a) cut feasibility of the workload (library only);
(b) DAHS with and without the cut option at three loads."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from qcs.runner import run_grid, RES
from qcs.sim import load_library

def feasibility():
    L = load_library(); rows = []
    for c in L["circuits"]:
        best = max(v["esp0"] for v in c["dev"].values())
        row = dict(algo=c["algo"], width=c["width"], cut_gates=c["cut_gates"], best_whole_esp=best,
                   cuttable="frag" in c)
        if "frag" in c:
            fa, fb = c["frag"]
            row["best_cut_esp"] = max(v["esp0"] for v in fa["dev"].values()) * max(v["esp0"] for v in fb["dev"].values())
            row["overhead"] = c["cut_overhead"]
        rows.append(row)
    df = pd.DataFrame(rows); RES.mkdir(exist_ok=True); df.to_csv(RES / "exp5_cut_feasibility.csv", index=False)
    print(f"circuits={len(df)} cuttable(<=2 cuts, width>=8)={int(df.cuttable.sum())}")

if __name__ == "__main__":
    feasibility()
    # calibrated cut-quality model (default) and the optimistic product model for comparison
    run_grid(["DAHS", "DAHS-NO-CUT"], [dict(load=l) for l in (0.5, 1.0, 2.0)], range(1, 21), "exp5_cutting")
    run_grid(["DAHS", "DAHS-NO-CUT"], [dict(load=1.0, cut_model="product")], range(1, 21), "exp5_cutting_product")
