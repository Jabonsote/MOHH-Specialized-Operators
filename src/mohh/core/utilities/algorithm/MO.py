
import time

import numpy as np

from mohh.core.params import Params
from mohh.core.utilities.metrics import RunMetrics
from pymoo.indicators.hv import HV

params = Params()

sign = 1
if not(params["OPTIMIZATION_KIND"]):
    sign = -1


def dominates_vectorized(p, q):
    return np.all(p >= q) and np.any(p > q)

def non_dominated_sorting_vectorized(population: set):

    population = np.array(list(population))

    # sign: 1 maximize, -1 minimize
    adjusted_population = population * sign

    N = adjusted_population.shape[0]
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):
        if is_dominated[i]:
            continue
            
        for j in range(N):
            if dominates_vectorized(adjusted_population[i], adjusted_population[j]):
                is_dominated[j] = True
            elif dominates_vectorized(adjusted_population[j], adjusted_population[i]):
                is_dominated[i] = True
                break

    front = population[~is_dominated]
    front_set = [tuple(p) for p in front]

    return [front_set]


def compute_nadir_point(front: set):

    if params["OPTIMIZATION_KIND"]:
        nadir_point = list(map(min, zip(*front)))
    else:
        nadir_point = list(map(max, zip(*front)))
    return nadir_point


def compute_hypervolume(ref_point: np.ndarray, front: np.ndarray) -> float:

    metrics = RunMetrics()
    num_points = int(front.shape[0]) if hasattr(front, "shape") else 0
    start = time.perf_counter()

    indicator = HV(ref_point = ref_point)

    hv = indicator(front)

    elapsed = time.perf_counter() - start

    metrics.timing("metrics_hv", elapsed, category="metrics")
    metrics.record("hv_calls", {
        "ref_point": [float(v) for v in np.atleast_1d(np.asarray(ref_point))],
        "num_points": num_points,
        "seconds": round(elapsed, 9),
    })

    return hv

