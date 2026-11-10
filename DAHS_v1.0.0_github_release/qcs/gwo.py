"""Grey Wolf Optimizer (Mirjalili et al., 2014) used offline to tune DAHS."""
import numpy as np


def gwo(f, lo, hi, n_wolves=10, iters=15, seed=0, log=print):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dim = lo.size
    X = lo + rng.random((n_wolves, dim)) * (hi - lo)
    fit = np.array([f(x) for x in X])            # maximise
    # leaders = best three positions found so far (classic GWO keeps them)
    L = sorted(zip(fit.tolist(), [x.copy() for x in X]), key=lambda z: -z[0])[:3]
    hist = []
    for it in range(iters):
        alpha, beta, delta = L[0][1], L[1][1], L[2][1]
        a = 2.0 - 2.0 * it / iters
        for i in range(n_wolves):
            new = np.zeros(dim)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(dim), rng.random(dim)
                A, C = 2 * a * r1 - a, 2 * r2
                D = np.abs(C * leader - X[i])
                new += leader - A * D
            X[i] = np.clip(new / 3.0, lo, hi)
            fit[i] = f(X[i])
        L = sorted(L + list(zip(fit.tolist(), [x.copy() for x in X])), key=lambda z: -z[0])[:3]
        hist.append((it, float(L[0][0]), L[0][1].tolist()))
        log(f"GWO iter {it+1}/{iters}: best so far={L[0][0]:.4f} theta={np.round(L[0][1], 3).tolist()}")
    return L[0][1], float(L[0][0]), hist
