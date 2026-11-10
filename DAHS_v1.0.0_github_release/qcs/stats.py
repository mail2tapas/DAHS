"""Paired statistics over seeds: mean difference, 95% bootstrap CI,
Wilcoxon signed-rank p-value, Holm correction."""
import numpy as np
from scipy.stats import wilcoxon


def paired(a, b, n_boot=10000, seed=0):
    """Difference a-b over paired seeds."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, d.size)).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return float(d.mean()), float(lo), float(hi), p


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps)
    order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()
