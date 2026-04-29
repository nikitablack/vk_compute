import time
from collections.abc import Callable


def benchmark_with_percentiles(
    fn: Callable[[], None], runs: int = 100, percentile_count: int = 10
) -> list[float]:
    if runs <= 0:
        raise ValueError("incorrect number of runs")

    if percentile_count <= 0:
        raise ValueError("incorrect number of percentiles")

    samples: list[int] = [0] * runs

    for i in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - start

    samples.sort()

    percentiles: list[float] = []
    for i in range(1, percentile_count + 1):
        p = i / percentile_count
        index = int(p * (runs - 1))
        percentiles.append(samples[index] / 1_000.0)  # µs

    return percentiles
