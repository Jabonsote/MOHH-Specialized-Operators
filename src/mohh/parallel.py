"""
Process pool for independent runs.
"""

import os

from concurrent.futures import ProcessPoolExecutor, as_completed


def available_cores() -> int:
    """Número de núcleos lógicos disponibles (mínimo 1)."""
    return max(1, os.cpu_count() or 1)


def run_parallel(fn, tasks: list, on_complete=None) -> list:
    """
    Runs fn(task) over tasks, one worker per core, results in task order.
    """
    if len(tasks) == 0:
        return []

    n_workers = min(available_cores(), len(tasks))

    if n_workers <= 1:
        results = []
        for i, task in enumerate(tasks):
            results.append(fn(task))
            if on_complete is not None:
                on_complete(i)
        return results

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        futures = [pool.submit(fn, task) for task in tasks]
        index_by_future = {fut: i for i, fut in enumerate(futures)}

        for future in as_completed(futures):
            i = index_by_future[future]
            results[i] = future.result()
            if on_complete is not None:
                on_complete(i)

    return results