"""Exp 10: generalisation. (a) circuit-family hold-out: DAHS tuned on half of the
families, tested on the other half; (b) device hold-out: Prague and Fez removed.
(c) strict family hold-out (python experiments/exp10_generalization.py strict): the
broker's cut model is also fitted only on the training families
(results/cut_model_trainfam.json, from experiments/cut_model_cv.py) and DAHS is
tuned with it (results/theta_algo_train_strict.json); the simulator's ground truth
still uses the model fitted on all validation circuits."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs.runner import run_grid, RES
SEEDS = range(1, 21)
sys.path.insert(0, str(Path(__file__).resolve().parent)); from experiments_common import ALGO_TEST
if __name__ == "__main__" and sys.argv[1:] == ["strict"]:
    th_s = tuple(json.loads((RES / "theta_algo_train_strict.json").read_text())["theta"])
    run_grid(["DAHS"], [dict(algos=ALGO_TEST, cut_model_pred_file="cut_model_trainfam.json", _theta=th_s)], SEEDS,
             "exp10c_algo_holdout_strict")
elif __name__ == "__main__":
    th_tr = tuple(json.loads((RES / "theta_algo_train.json").read_text())["theta"])
    st = [dict(algos=ALGO_TEST), dict(algos=ALGO_TEST, _theta=th_tr)]
    df = run_grid(["GREEDY-UTILITY", "TUNED-UTILITY", "DAHS"], [st[0]], SEEDS, "exp10a_algo_holdout_all")
    df = run_grid(["DAHS"], [st[1]], SEEDS, "exp10a_algo_holdout_train")
    run_grid(["GREEDY-UTILITY", "TUNED-UTILITY", "FID-THRESHOLD", "DAHS"],
             [dict(exclude_devices=("fake_prague", "fake_fez"))], SEEDS, "exp10b_device_holdout")
