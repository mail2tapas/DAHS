"""Exp 14: sensitivity of the scheduler results to the reconstruction-time fit.
The verification rerun of the 70 reconstruction runs (experiments/verify_cut_repeats.py)
reproduced every Q value but gave a different wall-clock time fit (r1 = 5.04 instead
of 4.40 s per million shots). Here DAHS and DAHS-NO-CUT are rerun at loads 0.5, 1 and 2
with results/cut_model_verify.json (eta identical, r0 and r1 from the rerun) as both the
simulator's and the broker's model, and compared with the shipped cut_model.json.
20 seeds, same tuned weights (results/theta.json).
    python experiments/exp14_trec_sensitivity.py      (~10 min on 2 cores)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid
if __name__ == "__main__":
    st = [dict(load=l, cut_model_file=f) for f in ("cut_model.json", "cut_model_verify.json") for l in (0.5, 1.0, 2.0)]
    d = run_grid(["DAHS", "DAHS-NO-CUT"], st, range(1, 21), "exp14_trec_sensitivity")
    import json; from qcs.runner import RES
    out = {}
    for l in (0.5, 1.0, 2.0):
        x = d[d.load == l].pivot_table(index="seed", columns=["cut_model_file", "scheduler"], values="utility")
        cs = d[(d.load == l) & (d.scheduler == "DAHS")].groupby("cut_model_file").cut_share.mean()
        du = x[("cut_model_verify.json", "DAHS")] - x[("cut_model.json", "DAHS")]
        out[str(l)] = dict(gain_shipped=float((x[("cut_model.json", "DAHS")] - x[("cut_model.json", "DAHS-NO-CUT")]).mean()),
                           gain_verify=float((x[("cut_model_verify.json", "DAHS")] - x[("cut_model_verify.json", "DAHS-NO-CUT")]).mean()),
                           d_util_dahs=float(du.mean()), d_util_dahs_maxabs=float(du.abs().max()),
                           cut_share_shipped=float(cs["cut_model.json"]), cut_share_verify=float(cs["cut_model_verify.json"]))
    (RES / "exp14_trec_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
