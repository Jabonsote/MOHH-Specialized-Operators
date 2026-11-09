

import os
import time
import random
import numpy as np
import pandas as pd


from tqdm import tqdm
from datetime import datetime


from itertools import product


from mohh.evaluation.params import Params, set_params
from mohh.core.utilities.instance_utils import instance_paths
from mohh.core.utilities.algorithm.MO import compute_hypervolume, non_dominated_sorting_vectorized
from mohh.core.utilities.metrics import RunMetrics


from mohh.evaluation.problem.instance import Instance


from mohh.evaluation.algorithms import MOEA_Decomposition, NSGAII, SMS_MOEA


from mohh.parallel import run_parallel


solution_type = "Natural"


initial_population_path = ""


all_mutation = ["NullMutation", "Swap_Mutation"]


def make_experiment_paths(experiment_path: list[str]):

    params = Params()

    folder_name = "evaluation_results"

    dir_path = os.path.join(experiment_path, folder_name)

    for solver in params["SOLVERS"]:

        solver_path = os.path.join(dir_path, solver)

        os.makedirs(solver_path, exist_ok=True)

    initial_solutions_path = os.path.join(dir_path, "initial_solutions")
    os.makedirs(initial_solutions_path, exist_ok=True)

    params["RESULTS_PATH"] = dir_path
    params["FILE_PATH_INITIAL_SOLUTIONS"] = initial_solutions_path


def generate_incremental_seeds(seeds_number: int, instance_name: str) -> list:

    start = datetime.now()
    seed = int(start.microsecond)  # Use microsecond as the base seed

    seeds = [seed]  # Initialize the list with the first seed

    for i in range(1, seeds_number):
        increment = random.randint(1, 1000)
        new_seed = seeds[-1] + increment
        seeds.append(new_seed)

    if initial_population_path:
        try:
            old_seeds = []

            with open(os.path.join(initial_population_path, "real", instance_name, "seeds.txt"), "r") as f:
                for line in f:
                    old_seeds.append(int(line))

            seeds = old_seeds
        except:
            pass

    return seeds


def save_seeds(seeds: list, folder_path: str, instance_name: str):

    real_path = os.path.join(folder_path, "real", instance_name, "seeds.txt")
    permutation_path = os.path.join(folder_path, "permutation", instance_name, "seeds.txt")

    with open(real_path, "w") as f:

        for seed in seeds:
            f.write(str(seed) + "\n")
        f.close()

    with open(permutation_path, "w") as f:

        for seed in seeds:
            f.write(str(seed) + "\n")
        f.close()


def get_combinations(all_crossover):

    regular_crossover = all_crossover[:-1]
    special_crossover = all_crossover[-1]

    regular_combinations = [
        (op[0], op[1], mutation, op[3], op[2])
        for op, mutation in product(regular_crossover, all_mutation)
    ]

    special_combinations = [
        (special_crossover[0], special_crossover[1], "NullMutation", special_crossover[3], special_crossover[2]),
        (special_crossover[0], special_crossover[1], "Swap_Mutation", special_crossover[3], special_crossover[2])
    ]

    all_combinations = regular_combinations + special_combinations

    sorted_combinations = sorted(all_combinations, key = lambda x: (x[2], x[0], x[4], x[3], x[1]))

    return sorted_combinations


def build_solver(solver_name, cross_type, crossover, mutation):

    params = Params()

    if solver_name == "MOEAD":
        solver = MOEA_Decomposition()
        cross_prob = params["MOEAD_CROSSOVER_PROBABILITY"]
        mutation_prob = params["MOEAD_MUTATION_PROBABILITY"]

    elif solver_name == "NSGAII":
        solver = NSGAII()
        cross_prob = params["NSGA_CROSSOVER_PROBABILITY"]
        mutation_prob = params["NSGA_MUTATION_PROBABILITY"]

    elif solver_name == "SMSEMOA":
        solver = SMS_MOEA()
        cross_prob = params["SMS_CROSSOVER_PROBABILITY"]
        mutation_prob = params["SMS_MUTATION_PROBABILITY"]

    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    if cross_type == "operator_template":
        solver.load_operator(cross_type, "HH_Operator",
                             operator=crossover,
                             solution_type=solution_type,
                             prob=cross_prob)
    else:
        solver.load_operator(cross_type, crossover, prob=cross_prob)

    solver.load_operator("mutation", mutation, prob=mutation_prob)

    return solver


def _solve_experiment_task(task: dict) -> dict:
    """
    Worker: runs one (solver, operator, experiment) task and returns the
    per-budget snapshots plus phase timings.
    """
    params_snapshot = task["params"]
    solver_name = task["solver_name"]
    cross_type = task["cross_type"]
    cross_operator = task["cross_operator"]
    mutation_operator = task["mutation_operator"]
    seed = task["seed"]
    instance = task["instance"]
    experiment = task["experiment"]
    kind = task["kind"]

    # rebuild Params in worker
    Params().update_params(params_snapshot)

    # same seed as the main process
    random.seed(seed)
    np.random.seed(seed)

    instance.set_initial_solutions(experiment=experiment, kind=kind)

    solver = build_solver(solver_name, cross_type, cross_operator, mutation_operator)

    t0 = time.time()
    solver.load_instance(instance)
    load_s = time.time() - t0

    t0 = time.time()
    solver.start_model(seed)
    setup_s = time.time() - t0

    t0 = time.time()
    results = solver.solve_instance()
    solve_s = time.time() - t0

    return {
        "snapshots": results,
        "load_s": load_s,
        "setup_s": setup_s,
        "solve_s": solve_s,
    }


def normalize_values(data: np.ndarray, vmin: float, vmax: float):

    return (data - vmin) / (vmax - vmin)


def write_front(front: np.ndarray, file_path: str):
    """
    Take a 2D numpy array and write it in disk

    Params
    """

    N, M = front.shape

    with open(file_path, "w") as file:

        file.write(f"# {N} {M}\n")

        for row in front:
            formatted_point = " ".join(f"{num:1.6e}" for num in row)
            file.write(f"{formatted_point}\n")

        file.close()


def run_experiments(experiment_path, results_paths, operators, overrides: dict = None) -> str:

    set_params(overrides)

    params = Params()
    solver_names = params["SOLVERS"]

    # generation models, same order as the operators
    model_names = params["MODELS"] or solver_names

    n_experiments = params["N_EXPERIMENTS"]
    eval_labels = list(params["EVAL_BUDGETS"].keys())

    print("\n" + "-" * 40)
    print("Starting Evaluation")
    print("-" * 40)
    print(f"Solvers:            {solver_names}")
    print(f"Models:             {model_names}")
    print(f"N experiments:      {n_experiments}")
    print(f"MO population size: {params['MO_POPULATION_SIZE']}")
    print(f"MO generations:     {params['MO_GENERATIONS']}")
    print(f"Eval budgets:       {params['EVAL_BUDGETS']}")
    print("-" * 40)

    all_crossover = []

    for i in range(0, len(operators), 3):

        model_name = model_names[i // 3]

        all_crossover.append(("operator_template", operators[i],   model_name, "Best"))
        all_crossover.append(("operator_template", operators[i+1], model_name, "Middle"))
        all_crossover.append(("operator_template", operators[i+2], model_name, "Worst"))

    all_crossover.append(("crossover", "PMX_Cross", "None", "Standard"))
    all_crossover.append(("crossover", "CX_Cross", "None", "Standard"))

    make_experiment_paths(experiment_path)

    if params["LOG_METRICS"]:
        RunMetrics.reset_instance()
        RunMetrics().init(os.path.join(params["RESULTS_PATH"], params["METRICS_DIRNAME"]),
                          phase="evaluation")
        RunMetrics().set_info("experiment_path", experiment_path)
        RunMetrics().set_info("operators", operators)
        RunMetrics().set_info("params", params.get_params())
        RunMetrics().record("config", {
            "solvers": solver_names,
            "models": model_names,
            "n_experiments": n_experiments,
            "eval_budgets": eval_labels,
            "eval_budget_values": params["EVAL_BUDGETS"],
            "mo_population_size": params["MO_POPULATION_SIZE"],
            "mo_generations": params["MO_GENERATIONS"],
        })

    instances = instance_paths(train=False)

    program_time = time.time()

    for instance_path in instances:

        instance_start = time.time()

        instance = Instance(params["MO_POPULATION_SIZE"], params["SOLUTION_TYPE"])

        instance_name = instance_path.replace("\\", "/").split("/")[-1]
        instance_name = instance_name.removesuffix(".txt")

        print("\n" + "-" * 40)
        print(f"Instance: {instance_name}")
        print("-" * 40)

        instance.load_problem(params["PROBLEM_NAME"], instance_path, n_experiments, initial_population_path)

        seeds = generate_incremental_seeds(n_experiments, instance_name)
        save_seeds(seeds, params["FILE_PATH_INITIAL_SOLUTIONS"], instance_name)

        consolidated = {label: set() for label in eval_labels}

        fronts_model_level = {label: {} for label in eval_labels}

        for combination in get_combinations(all_crossover):

            cross_type = combination[0]
            cross_operator = combination[1]
            mutation_operator = combination[2]
            combination_name = combination[3]
            solver_used = combination[4]

            kind = "permutation"

            for solver_name in solver_names:

                fronts_exp_level = {label: {} for label in eval_labels}

                with_mutation = "NM" if mutation_operator == "NullMutation" else "WM"

                # N experiments in parallel
                tasks = [
                    {
                        "params": {k: v for k, v in params.get_params().items() if k != "BNF_GRAMMAR"},
                        "solver_name": solver_name,
                        "cross_type": cross_type,
                        "cross_operator": cross_operator,
                        "mutation_operator": mutation_operator,
                        "seed": seeds[i - 1],
                        "instance": instance,
                        "experiment": i,
                        "kind": kind,
                    }
                    for i in range(1, n_experiments + 1)
                ]

                # live progress bar
                pbar = tqdm(total=n_experiments,
                            desc=f"{solver_used} {combination_name} {with_mutation} {solver_name}",
                            bar_format="{l_bar}{bar:20}{r_bar}", leave=True)

                def on_complete(_index: int):
                    pbar.update(1)

                results = run_parallel(_solve_experiment_task, tasks, on_complete=on_complete)
                pbar.close()

                for i, result in enumerate(results, start=1):

                    load_s = result["load_s"]
                    setup_s = result["setup_s"]
                    solve_s = result["solve_s"]
                    results_snapshots = result["snapshots"]

                    metrics = RunMetrics()
                    metrics.timing("moea_problem_setup", load_s, "moea")
                    metrics.timing("moea_algorithm_setup", setup_s, "moea")
                    metrics.timing("moea_solve", solve_s, "moea")

                    budget_sizes = {}
                    for label in eval_labels:

                        pareto_front = results_snapshots.get(label)

                        if pareto_front is None:
                            continue

                        budget_sizes[label] = int(pareto_front.shape[0])

                        fronts_exp_level[label][f"Experiment {i:03d}"] = pareto_front

                        consolidated[label] = consolidated[label].union([tuple(point) for point in pareto_front.tolist()])

                    metrics.record("moea_runs", {
                        "instance": instance_name,
                        "config": f"{solver_used}_{combination_name}_{with_mutation}_{solver_name}",
                        "experiment": i,
                        "load_s": round(load_s, 9),
                        "setup_s": round(setup_s, 9),
                        "solve_s": round(solve_s, 9),
                        "budget_sizes": budget_sizes,
                    })
                    metrics.increment("moea_runs", "count", 1)

                if combination_name == "Standard" and cross_operator == "SBX_Cross":
                    combination_name = "SBX"
                elif combination_name == "Standard" and cross_operator == "PMX_Cross":
                    combination_name = "PMX"
                elif combination_name == "Standard" and cross_operator == "CX_Cross":
                    combination_name = "CX"

                for label in eval_labels:
                    RunMetrics().record("consolidated_sizes", {
                        "instance": instance_name,
                        "label": label,
                        "solver": solver_name,
                        "config": f"{solver_used}_{combination_name}_{with_mutation}",
                        "size": int(len(consolidated[label])),
                    })

                for label in eval_labels:
                    if fronts_exp_level[label]:
                        fronts_model_level[label][f"{solver_name}_{combination_name}_{with_mutation}_{solver_used}"] = fronts_exp_level[label]

        # 3*M*N fronts per instance

        # consolidated front -> nadir -> HV per combination

        # avoid a nadir dominated by a single point
        def comprobate(front, consol):
            while len(front) <= 2 and len(consol) > len(front):
                for point in front:
                    consol.remove(point)
                front = non_dominated_sorting_vectorized(consol)[0]

            return front

        best_fronts = {}
        best_fronts_norm = {}
        boundaries = {}

        n_obj = instance.k_objectives
        nadir_point = np.array([1.1] * n_obj) # Fixed for normalized PFAs

        for label in eval_labels:

            if not consolidated[label]:
                continue

            metrics = RunMetrics()

            metrics.start("metrics_nds", category="metrics")
            best_front = non_dominated_sorting_vectorized(consolidated[label])[0]
            best_front = comprobate(best_front, consolidated[label])
            metrics.stop("metrics_nds")

            # vmax==vmin -> degenerate front, NaN
            min_val, max_val = np.min(best_front, axis=0), np.max(best_front, axis=0)
            boundaries[label] = [min_val, max_val]

            best_fronts[label] = best_front
            best_fronts_norm[label] = normalize_values(best_front, min_val, max_val)

            metrics.record("best_fronts", {
                "instance": instance_name,
                "label": label,
                "size": int(len(best_front)),
                "vmin": min_val.tolist() if hasattr(min_val, "tolist") else list(min_val),
                "vmax": max_val.tolist() if hasattr(max_val, "tolist") else list(max_val),
                "nadir_ref_point": [1.1] * int(n_obj),
            })

        for evals in eval_labels:

            if evals not in boundaries:
                continue

            data = {}

            front_save_path = os.path.join(params["RESULTS_PATH"], solver_name, evals, "fronts", instance_name)
            os.makedirs(front_save_path, exist_ok=True)

            metrics = RunMetrics()

            for model_operators in fronts_model_level[evals].keys():

                column_data = []

                for experiment in fronts_model_level[evals][model_operators].keys():

                    front = fronts_model_level[evals][model_operators][experiment]

                    front = normalize_values(front, boundaries[evals][0], boundaries[evals][1])

                    hv = compute_hypervolume(nadir_point, front)

                    metrics.record("hypervolumes", {
                        "instance": instance_name,
                        "label": evals,
                        "config": model_operators,
                        "experiment": experiment,
                        "hypervolume": float(hv),
                        "front_size": int(front.shape[0]),
                        "normalized": True,
                        "ref_point": [1.1] * int(n_obj),
                    })
                    metrics.increment("hypervolumes", "count")

                    column_data.append(hv)

                # median index (np.median may interpolate)
                median_indx = int(np.argsort(column_data)[len(column_data) // 2]) + 1
                median_front = fronts_model_level[evals][model_operators][f"Experiment {median_indx:03d}"]
                median_front_norm = normalize_values(median_front, boundaries[evals][0], boundaries[evals][1])

                metrics.record("median_fronts", {
                    "instance": instance_name,
                    "label": evals,
                    "config": model_operators,
                    "median_experiment": f"Experiment {median_indx:03d}",
                    "median_hypervolume": float(column_data[median_indx - 1]),
                })

                metrics.start("io_results", category="io")
                write_front(median_front, os.path.join(front_save_path, f"{model_operators}.pof"))
                write_front(median_front_norm, os.path.join(front_save_path, f"{model_operators}_norm.pof"))
                metrics.stop("io_results")

                data[model_operators] = column_data

            metrics.start("io_results", category="io")
            write_front(np.array(best_fronts[evals]), os.path.join(front_save_path, "best_front_consolidated.pof"))
            write_front(np.array(best_fronts_norm[evals]), os.path.join(front_save_path, "best_front_consolidated_norm.pof"))

            dataframe = pd.DataFrame(data)
            dataframe.insert(0, "Experiment", range(1, n_experiments + 1))

            columns = dataframe.columns

            standard_columns = [col for col in columns if col.endswith("None")]

            for solver_name in solver_names:

                new_path = os.path.join(params["RESULTS_PATH"], solver_name, evals)
                os.makedirs(new_path, exist_ok=True)
                save_path = os.path.join(new_path, instance_name + ".csv")

                subset_columns = [col for col in columns if col.endswith(solver_name)]

                subset_columns.extend(standard_columns)
                subset_columns = ["Experiment"] + subset_columns

                subset_df = dataframe[subset_columns]
                subset_df.to_csv(save_path, index=False)
            metrics.stop("io_results")

        RunMetrics().dump(f"evaluation_{instance_name}", {"instance": instance_name})

        instance_time = time.time() - instance_start
        instance_minutes = round(instance_time / 60, 2)
        instance_hours = round(instance_time / 3600, 2)

        print(f"\nInstance: {instance_name}, Minutes: {instance_minutes}, Hours: {instance_hours}\n")

    program_total_time = time.time() - program_time
    minutes = round(program_total_time /60 , 4)
    hours = round(program_total_time / 3600, 4)

    RunMetrics().record("run", {"event": "end", "total_seconds": program_total_time})
    RunMetrics().write_summary()
    RunMetrics().close()

    print(f"\nProgram Time - Minutes: {minutes}, Hours: {hours}\n")

    return params["RESULTS_PATH"]

