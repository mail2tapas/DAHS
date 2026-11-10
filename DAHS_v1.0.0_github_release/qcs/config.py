"""Default simulation settings (all experiments start from these)."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # horizon
    hours: float = 48.0            # simulated time
    warmup_h: float = 1.0          # broker jobs arriving earlier are not measured
    # broker (our) workload
    load: float = 1.0              # multiplies the broker arrival rate
    base_rate_per_h: float = 300.0 # broker jobs per hour at load 1
    shots_choices: tuple = (1000, 2000, 4000, 8000)
    rep_delay_s: float = 250e-6    # reset/repetition delay per shot
    overhead_s: tuple = (1.0, 3.0) # per-job load/compile overhead, uniform
    # background (other users') load on each device: utilisation at ext_load=1
    ext_util: dict = field(default_factory=lambda: {
        "fake_prague": 0.80, "fake_fez": 0.75, "fake_torino": 0.70,
        "fake_brisbane": 0.60, "fake_sherbrooke": 0.55, "fake_kyiv": 0.45,
        "fake_kolkata": 0.40, "fake_guadalupe": 0.35})
    ext_load: float = 1.0
    ext_service_mean_s: float = 30.0
    ext_service_cv: float = 1.5
    diurnal_amp: float = 0.15      # +/- variation of background load over the day
    # calibration and drift
    day_sigma: float = 0.20        # day-to-day log-error random walk (published)
    day_rho: float = 0.6           # AR(1) coefficient of the day-to-day walk
    bad_prob: float = 0.01         # chance per element per day of a degraded element
    bad_factor: tuple = (3.0, 10.0)
    kappa_max: float = 1.0         # intra-day staleness: true error = published*exp(kappa*h/24)
    obs_sigma: float = 0.05        # noise of the fidelity feedback (log scale)
    # objective
    w_t_per_min: float = 0.01      # utility = fidelity - w_t * turnaround(minutes)
    allow_cut: bool = True
    # robustness / scalability / generalisation options (defaults = main setting)
    fb_prob: float = 1.0           # probability that a finished job returns fidelity feedback
    drift_shape: str = "exp"       # "exp" (monotonic) or "rw" (non-monotonic random walk)
    rw_sigma: float = 0.3          # random-walk scale for drift_shape="rw" (log scale per sqrt(day))
    model_err_sigma: float = 0.0   # log-scale error of the ESP proxy, fixed per circuit and device
    fleet_factor: int = 1          # replicate the device fleet (scalability); broker rate scales with it
    exclude_devices: tuple = ()    # device models removed from the fleet (generalisation)
    algos: tuple = None            # restrict circuit families (None = all)
    fb_delay_h: float = 0.0        # delay between the end of a job and the arrival of its feedback (hours)
    fb_probe_cost: float = 0.0     # extra device time for a job that returns feedback, as a multiple of its run time
                                   # (1.0 = one mirror/known-answer run of the same length); charged only to
                                   # policies that use feedback
    cut_eta_z: float = 0.0         # broker predicts eta - z * (prediction standard error of eta); 0 = point estimate
    cut_eff_true_delta: float = 0.0  # shift of the TRUE cut efficiency eta; the broker still predicts with the fit
    cut_model_file: str = "cut_model.json"        # fitted cut model: ground truth of the simulator
    cut_model_pred_file: str = None               # model the broker predicts with (None = the same file)
    cut_model: str = "calibrated"  # "calibrated" from reconstruction runs, or optimistic "product" of fragment ESPs

