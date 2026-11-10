"""Deterministic access to MQT Bench circuits.

MQT Bench's graph-state benchmark draws a random regular graph without a fixed
seed (seed=None), so it returns a different circuit in every process. All other
benchmarks used here are deterministic (fixed seeds inside MQT Bench). This
wrapper fixes the graph-state seed to 10, the seed MQT Bench uses for its other
random benchmarks, so the library, the cut analysis and the reconstruction runs
all use the same circuits."""
from mqt.bench import get_benchmark, BenchmarkLevel
from mqt.bench.benchmarks import graphstate as _gs

_gs.create_circuit.__defaults__ = (2, 10)   # (degree, seed)


def get_circuit(algo, n, level=BenchmarkLevel.INDEP):
    return get_benchmark(algo, level, n)
