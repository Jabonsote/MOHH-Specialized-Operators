
import time
from math import comb

import numpy as np

from mohh.core.params import Params
from mohh.core.utilities.metrics import RunMetrics
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

params = Params()

sign = 1
if not(params["OPTIMIZATION_KIND"]):
    sign = -1


def reference_directions_for_size(k_objectives: int, target_size: int):
    """
    Uniform reference directions with a size pymoo accepts.

    get_reference_directions(..., n_points=N) only accepts sizes of the form
    C(p + k - 1, k - 1), so for some k there is no N equal to the target
    (105 points with k=4 or k=5, for instance). Pick the closest partition and
    return the directions with their actual size.
    """
    partitions = min(
        range(1, max(1, target_size) + 1),
        key=lambda p: abs(comb(p + k_objectives - 1, k_objectives - 1) - target_size)
    )

    ref_dirs = get_reference_directions("uniform", k_objectives, n_partitions=partitions)

    return ref_dirs, len(ref_dirs)


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

