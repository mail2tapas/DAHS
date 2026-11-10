#!/bin/bash
set -e
cd "$(dirname "$0")"
P=python
# cutting validation and cost model (needs data/library.pkl from data/build_data.py)
$P data/cut_cost.py library          # sub-experiment counts, q and the 9^q shot overhead
$P experiments/exp_cut_validation.py # circuit and device selection
$P experiments/exp_cut_repeats.py    # 14 circuits x 5 sampling seeds
$P data/cut_cost.py fit              # cut model on the five-run means
$P experiments/cut_model_cv.py       # cross-validation, held-out runs, Holm tests, training-family model
$P experiments/tune.py --target dahs --gwo-seed 0 --out theta.json
$P experiments/tune.py --target dahs --gwo-seed 1 --out theta_gwo1.json
$P experiments/tune.py --target dahs --gwo-seed 2 --out theta_gwo2.json
$P experiments/tune.py --target tuned_utility --out theta_tuned_utility.json
$P experiments/tune.py --target dahs --w 0.03 --out theta_w0.03.json
$P experiments/tune.py --target dahs --w 0.1 --out theta_w0.1.json
$P experiments/tune.py --target dahs --algo-split train --out theta_algo_train.json
$P experiments/tune.py --target dahs --algo-split train --cut-model cut_model_trainfam.json --out theta_algo_train_strict.json
$P experiments/train_pg.py
for e in exp1_main exp2_load exp3_drift exp4_ablation exp5_cutting exp6_breakeven exp6b_retuned exp7_feedback exp8_robustness exp10_generalization exp11_lambda exp12_feedback_cost exp13_cut_mismatch exp9_scalability; do
  echo "== $e $(date +%T)"; $P experiments/$e.py
done
$P experiments/exp10_generalization.py strict
$P experiments/extra_stats.py
$P experiments/exp14_trec_sensitivity.py   # needs results/cut_model_verify.json (verify_cut_repeats.py)
$P experiments/report.py
$P make_figures.py
echo ALLDONE
