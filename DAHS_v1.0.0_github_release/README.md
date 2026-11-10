# DAHS: Drift-Aware Hybrid Scheduler for Quantum Cloud Workloads

Simulator, schedulers, experiments and results of DAHS, a scheduler that places
quantum jobs on shared quantum processors using staleness-corrected estimated
success probability, predicted turnaround, congestion and optional circuit
cutting. Everything runs on a laptop CPU. No quantum hardware, GPU or cloud
account is needed.

## Reference environment
All run times were measured on Linux with 2 CPU cores and 8 GB RAM,
Python 3.13.17, qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0,
qiskit-addon-cutting 0.10.0, mqt.bench 2.3.0, numpy 2.5.3, scipy 1.18.1,
pandas 3.0.5 and matplotlib 3.11.2 (pinned in `requirements.txt`).

## Seeds
| Use | Seeds |
|---|---|
| Test runs (all simulation experiments) | 1–20 (Exp 9: 1–5) |
| Tuning (GWO) and PG-LINEAR training | 101–106 |
| GWO's own random numbers | 0, 1, 2 |
| Sampling seeds of the reconstruction runs | 11–15 |

All policies use common random numbers: for a given seed they see the same
arrivals, background jobs and calibrations.

## Data (all public, built locally)
* **Circuits:** MQT Bench (target-independent level), 16 algorithms, 4–28 qubits.
* **Devices:** 8 Qiskit fake IBM backends that carry real calibration snapshots
  (Torino, Fez, Brisbane, Sherbrooke, Kyiv, Prague, Kolkata, Guadalupe).
* Each circuit is transpiled once per device (optimisation level 1); gate counts
  on physical qubits/couplers and durations are stored in `data/library.pkl`.
* Only the day-0 calibration is real. Day-to-day evolution, intra-day drift,
  background load and fidelity feedback are synthetic.

## Install
```
pip install -r requirements.txt
```

## Run (results go to `results/`; times on 2 cores)
| Step | Command | Time | Output |
|---|---|---|---|
| Build circuit-device library | `python data/build_data.py` | ~1 min | data/library.pkl |
| Sub-experiment counts and 9^q shot overhead | `python data/cut_cost.py library` | ~2 min | data/library.pkl (cut fields) |
| Circuit and device selection for cut validation | `python experiments/exp_cut_validation.py` | seconds | exp_cut_validation.csv |
| Repeated reconstruction, 14 circuits × 5 sampling seeds | `python experiments/exp_cut_repeats.py` | ~1.5 h | exp_cut_repeats.csv |
| Calibrated cut model (fit on five-run means) | `python data/cut_cost.py fit` | seconds | cut_model.json |
| Cross-validation of the cut model; training-family model | `python experiments/cut_model_cv.py` | seconds | cut_model_cv.json, exp_cut_repeats_summary.csv, cut_model_trainfam.json |
| Sensitivity of the scheduler to the reconstruction-time fit (Exp. 14) | `python experiments/exp14_trec_sensitivity.py` | ~10 min | exp14_trec_sensitivity.csv/.json |
| Reproducibility check of the 70 reconstruction runs | see step 6 below; `python experiments/verify_cut_repeats.py` | ~1 h on two cores | cut_repeats_verification.json, cut_model_verify.json |
| Tune DAHS (GWO seeds 0, 1, 2) | `python experiments/tune.py --target dahs --gwo-seed 0 --out theta.json` (and `--gwo-seed 1/2 --out theta_gwo1/2.json`) | ~5 min each | theta.json, theta_gwo1.json, theta_gwo2.json |
| Tune TUNED-UTILITY baseline | `python experiments/tune.py --target tuned_utility --out theta_tuned_utility.json` | ~3 min | theta_tuned_utility.json |
| Retune at other time weights | `python experiments/tune.py --target dahs --w 0.03 --out theta_w0.03.json` (and `--w 0.1`) | ~5 min each | theta_w0.03.json, theta_w0.1.json |
| Tune on half the circuit families | `python experiments/tune.py --target dahs --algo-split train --out theta_algo_train.json` | ~5 min | theta_algo_train.json |
| Train PG-LINEAR learning baseline | `python experiments/train_pg.py` | ~2 min | pg_weights.json |
| Exp 1 main comparison (3 loads, 9 policies) | `python experiments/exp1_main.py` | ~13 min | exp1_main.csv |
| Exp 2 background load | `python experiments/exp2_load.py` | ~11 min | exp2_load.csv |
| Exp 3 drift strength | `python experiments/exp3_drift.py` | ~14 min | exp3_drift.csv |
| Exp 4 ablation | `python experiments/exp4_ablation.py` | ~4 min | exp4_ablation.csv |
| Exp 5 cutting in the scheduler | `python experiments/exp5_cutting.py` | ~5 min | exp5_cutting*.csv, exp5_cut_feasibility.csv |
| Exp 6 time weight, fixed policy (needs Exp 1) | `python experiments/exp6_breakeven.py` | seconds | exp6_breakeven.csv/.json |
| Exp 6b time weight, retuned | `python experiments/exp6b_retuned.py` | ~5 min | exp6b_w*.csv |
| Exp 7 sparse/noisy feedback | `python experiments/exp7_feedback.py` | ~10 min | exp7_feedback.csv |
| Exp 8 proxy error, non-monotonic drift | `python experiments/exp8_robustness.py` | ~15 min | exp8a_proxy_error.csv, exp8b_rw_drift.csv |
| Exp 9 scalability (8–64 QPUs) | `python experiments/exp9_scalability.py` | ~60 min | exp9_scalability.csv |
| Exp 10 held-out families and devices | `python experiments/exp10_generalization.py` | ~5 min | exp10a_*.csv, exp10b_device_holdout.csv |
| Tune for the strict family hold-out | `python experiments/tune.py --target dahs --algo-split train --cut-model cut_model_trainfam.json --out theta_algo_train_strict.json` | ~5 min | theta_algo_train_strict.json |
| Exp 10c strict family hold-out | `python experiments/exp10_generalization.py strict` | ~1 min | exp10c_algo_holdout_strict.csv |
| Exp 11 forgetting factor λ | `python experiments/exp11_lambda.py` | ~20 min | exp11_lambda.csv |
| Exp 12 delayed and costly feedback | `python experiments/exp12_feedback_cost.py` | ~15 min | exp12_feedback_cost.csv |
| Exp 13 error in the cut model; uncertainty margin z | `python experiments/exp13_cut_mismatch.py` | ~12 min | exp13_cut_mismatch.csv |
| Extra statistics | `python experiments/extra_stats.py` | ~2 min | extra_stats.json |
| Paired tests (seed-level bootstrap, Wilcoxon, Holm) | `python experiments/report.py` | seconds | *_paired.csv, report.txt |
| Figures | `python make_figures.py` | seconds | results/figs/*.png |

`run_all.sh` runs the tuning and experiment steps in order. Per-job result files
(`results/*_jobs.pkl`) are not included; run Exp 1 before Exp 6. Tuned weights
(`results/theta*.json`, `results/pg_weights.json`) and the calibrated cut model
(`results/cut_model.json`) are included.

## Reproducing the cutting validation
1. `python data/build_data.py` (circuit-device library, if not present).
2. `python data/cut_cost.py library`: for each cuttable circuit, the number q of
   cut CNOTs after decomposition, the n_sub = 2·6^q sub-experiments and the shot
   overhead 9^q used by the scheduler and the validation runs.
3. `python experiments/exp_cut_validation.py`: selects the 14 validation circuits
   (five cuttable families, three smallest widths, at most 432 sub-experiments)
   and their devices.
4. `python experiments/exp_cut_repeats.py`: whole versus cut execution with full
   reconstruction, five sampling seeds (11–15) per circuit (per-circuit whole vs cut quality).
5. `python data/cut_cost.py fit`, then `python experiments/cut_model_cv.py`:
   the cut model fitted on the five-run means (`results/cut_model.json`),
   prediction intervals, cross-validation, the held-out-run check (fit on seeds
   11–12, test on 13–15), Holm-corrected per-circuit tests and the
   training-family model. Reconstruction time is modelled as linear in the total
   number of shots of the cut job; the broker does not cut circuits with more
   sub-experiments than the calibrated maximum (432).

Shots: whole execution 4,000; cut execution 4,000 × 9^q in total, spread evenly
over the sub-experiments with at least 64 each. Sampler seeds are passed as
`seed_simulator`, so the runs repeat exactly.

6. Reproducibility check: all 70 runs were repeated from scratch with
   `CUTREP_PREFIX=verify python experiments/exp_cut_repeats.py dj:10` and
   `CUTREP_PREFIX=verify python experiments/exp_cut_repeats.py bv:8 bv:10 bv:12 dj:8 ghz:8 ghz:10 ghz:12 qnn:8 qnn:10 qnn:12 wstate:8 wstate:10 wstate:12`
   (in parallel), then `python experiments/verify_cut_repeats.py`. Every value of
   Q was identical to `results/exp_cut_repeats.csv`, so the fitted eta model is
   identical; only the wall-clock reconstruction times differ
   (`results/cut_repeats_verification.json`, `results/exp_cut_repeats_verify.csv`).
   The scheduler uses the shipped `results/cut_model.json`; Experiment 14 reruns
   DAHS and DAHS-NO-CUT with `results/cut_model_verify.json` (the refitted times)
   and finds no material change (`results/exp14_trec_sensitivity.json`).

Graph-state circuits: MQT Bench draws the random graph of its graph-state
benchmark without a fixed seed. `qcs/bench.py` fixes the seed to 10, so every
script uses the same circuits; with this seed no graph-state circuit is
cuttable with at most two crossing gates.

## Code layout
```
data/build_data.py         circuit-device library (transpilation, ESP features, cut fragments)
data/cut_cost.py           sub-experiment counts and calibrated cut-quality model
qcs/config.py              all simulation settings
qcs/sim.py                 discrete-event simulator (queues, background load, drift, cutting)
qcs/schedulers.py          baselines, TUNED-UTILITY, PG-LINEAR and DAHS
qcs/gwo.py                 Grey Wolf Optimizer
qcs/bench.py               MQT Bench access with a fixed graph-state seed
qcs/stats.py               paired bootstrap CI, Wilcoxon, Holm
qcs/runner.py              parallel runs -> CSV
experiments/*.py           one script per experiment
make_figures.py            figures from saved results
```

## Modelling assumptions
Fidelity is the estimated success probability (ESP) from calibration data, not a
measured output; queue waits come from a background-load model (public queue
traces are not available); intra-day drift is synthetic (monotonic by default,
optional random-walk component); the quality of cut execution uses a model
calibrated on noisy-simulation reconstructions with qiskit-addon-cutting.

License: MIT.
