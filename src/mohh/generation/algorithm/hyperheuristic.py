
import copy
import time
import numpy as np
from tqdm import tqdm

from mohh.generation.params import Params
from mohh.generation.saver import PopulationSaver, MyLogger

from mohh.generation.problem.instance import Instance
from mohh.generation.algorithm.multiobjective import MOSolver
from mohh.generation.representation.population import Population
from mohh.generation.representation.population import Individual

from mohh.parallel import run_parallel

from mohh.core.utilities.print_utils import update_lines
from mohh.core.utilities.load_modules import find_module
from mohh.core.utilities.algorithm.MO import compute_nadir_point
from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.core.utilities.algorithm.MO import non_dominated_sorting_vectorized
from mohh.core.utilities.metrics import RunMetrics
from mohh.generation.utilities.algorithm.HH_auxiliars import compute_rank
from mohh.generation.utilities.algorithm.HH_auxiliars import test_individual


def _solve_individual_instance(task: dict) -> dict:
    """
    Worker: solves one instance with the MOEA for a candidate operator.
    """
    params_snapshot = task["params"]
    phenotype = task["phenotype"]
    solution_type = task["solution_type"]
    mo_cross_prob = task["mo_cross_prob"]
    seed = task["seed"]
    model_name = task["model_name"]
    instance = task["instance"]

    # rebuild Params in worker
    Params().update_params(params_snapshot)

    # deterministic seed
    np.random.seed(seed)

    solver = MOSolver(seed=seed)
    solver.load_operator("operator_template", "HH_Operator",
                         operator=phenotype,
                         solution_type=solution_type,
                         prob=mo_cross_prob)
    solver.load_operator("mutation", "NullMutation")

    start = time.time()
    solver.load_instance(instance)
    problem_setup_s = time.time() - start

    start = time.time()
    solver.start_model(model_name=model_name)
    algorithm_setup_s = time.time() - start

    start = time.time()
    results = solver.solve_instance()
    solve_s = time.time() - start

    pareto_set, pareto_front = results[:2]
    weak_non_dominated, unique_solutions, strong_non_dominated = results[2:]

    return {
        "instance_name": instance.instance_name,
        "pareto_set": pareto_set,
        "pareto_front": pareto_front,
        "weak_non_dominated": weak_non_dominated,
        "unique_solutions": unique_solutions,
        "strong_non_dominated": strong_non_dominated,
        "problem_setup_s": problem_setup_s,
        "algorithm_setup_s": algorithm_setup_s,
        "solve_s": solve_s,
    }


class HyperHeuristic():

    def __init__(self) -> None:

        params = Params()

        self.seed = params["RANDOM_SEED"]
        self.population_size = params["POPULATION_SIZE"]
        self.num_generations = params["GENERATIONS"]

        self.mo_model_name = params["MO_MODEL"]
        self.mo_cross_prob = params["MO_CROSSOVER_PROBABILITY"]
        self.mo_population_size = params["MO_POPULATION_SIZE"]

        self.selection = None
        self.crossover = None
        self.mutation = None

        self.population = None
        self.num_offsprings = params["GENERATION_SIZE"]

        self.problem_name = params["PROBLEM_NAME"]
        self.solution_type = params["SOLUTION_TYPE"]

        self.current_generation = 0

    def load_operator(self, operator_type: str, operator_name: str, **kwargs: dict):
        """
        Load a genetic operator to the algorithm.
        Choose between: selection, crossover, mutation,
        replacement and initialization.

        @param operator_type: Choose between selection, crossover,
        mutation, replacement and initialization
        @param operator_name: Operator name.
        """

        operator_type = operator_type.lower()

        module_name = f"mohh.generation.operators.{operator_type}"
        operator = find_module(module_name, operator_name, **kwargs)

        if operator_type == "initialization" or operator_type == "initialization":
            self.population = Population(operator)
            return self

        setattr(self, operator_type, operator)

        return self

    def load_instances(self, instance_paths: list) -> None:
        """
        Load instance paths, and create instance objects.

        @param instance_paths: List with instances file paths.
        """
        self.instances = []

        if type(self.mo_population_size) == int:
            self.mo_population_size = [self.mo_population_size]*len(instance_paths)

        for indx, instance_path in enumerate(instance_paths):
            instance = Instance(self.mo_population_size[indx], self.solution_type)

            instance.load_problem(self.problem_name, instance_path)

            self.instances.append(instance)

    def evaluation_step(self):
        """
        Evaluates the individuals by solving the instances (in parallel).
        """

        logger = MyLogger().get_logger()

        params = Params()

        # same order as the sequential version
        individuals = list(self.population.non_evaluated())

        # one task per candidate x instance
        tasks = [
            {
                "params": {k: v for k, v in params.get_params().items() if k != "BNF_GRAMMAR"},
                "phenotype": individual.phenotype,
                "solution_type": self.solution_type,
                "mo_cross_prob": self.mo_cross_prob,
                "seed": self.seed,
                "model_name": self.mo_model_name,
                "instance": instance,
            }
            for individual in individuals
            for instance in self.instances
        ]

        # live progress bar
        pbar = tqdm(total=len(individuals), desc="Evaluating...",
                    bar_format="{l_bar}{bar:20}{r_bar}", leave=True)

        n_instances = len(self.instances)

        # advance when a candidate finishes all its instances
        def on_complete(task_index: int):
            if (task_index + 1) % n_instances == 0:
                pbar.update(1)

        results = run_parallel(_solve_individual_instance, tasks, on_complete=on_complete)
        results_iter = iter(results)

        for individual in individuals:

            for instance in self.instances:

                result = next(results_iter)

                metrics = RunMetrics()
                metrics.timing("moea_problem_setup", result["problem_setup_s"], "moea")
                metrics.timing("moea_algorithm_setup", result["algorithm_setup_s"], "moea")
                metrics.timing("moea_solve", result["solve_s"], "moea")

                pareto_set = result["pareto_set"]
                pareto_front = result["pareto_front"]
                weak_non_dominated = result["weak_non_dominated"]
                unique_solutions = result["unique_solutions"]
                strong_non_dominated = result["strong_non_dominated"]

                metrics.record("moea_results", {
                    "generation": self.current_generation,
                    "individual": individual.name,
                    "instance": instance.instance_name,
                    "front_size": int(pareto_front.shape[0]) if hasattr(pareto_front, "shape") else len(pareto_front),
                    "pareto_set_size": int(len(pareto_set)),
                    "weak_non_dominated": int(weak_non_dominated),
                    "unique_solutions": int(unique_solutions),
                    "strong_non_dominated": int(strong_non_dominated),
                })
                metrics.increment("moea_runs", "count", 1)

                instance_name = instance.instance_name
                individual.pareto_sets[instance_name] = pareto_set
                individual.pareto_fronts[instance_name] = pareto_front
                individual.unique_solutions[instance_name] = unique_solutions
                individual.weak_non_dominated[instance_name] = weak_non_dominated
                individual.strong_non_dominated[instance_name] = strong_non_dominated

                instance.non_dominated_front = \
                    instance.non_dominated_front.union([tuple(element) for element in pareto_front.tolist()])

        pbar.close()

        total_time = pbar.format_dict["elapsed"]

        logger.info(f"Evaluation took: {total_time} seconds")

    def compute_metrics(self):
        """
        Compute an indicator metric and rank the individuals.
        """

        logger = MyLogger().get_logger()

        fitness_values = [[0]*len(self.instances) for _ in range(len(self.population.individuals))]

        instance_sorting_time = [0.0]*len(self.instances)
        instance_hv_time = [0.0]*len(self.instances)

        sorting_text = ", ".join([f"{i:05.1f} min" for i in instance_sorting_time])
        hv_text = ", ".join([f"{i:05.1f} min" for i in instance_hv_time])
        print("\nSorting: ", sorting_text)
        print("Hypervolumes: ", hv_text)

        pbar = tqdm(self.instances, desc="Computing metrics...",
                     bar_format="{l_bar}{bar:20}{r_bar}", leave=True)

        for i, instance in enumerate(pbar):

            instance_name = instance.instance_name

            metrics = RunMetrics()
            generation = self.current_generation

            sorting_time_init = time.time()
            metrics.start("metrics_nds", category="metrics")

            instance.fronts = non_dominated_sorting_vectorized(instance.non_dominated_front)

            metrics.stop("metrics_nds")

            sorting_time = (time.time() - sorting_time_init) / 60

            instance.non_dominated_front = set(instance.fronts[0])

            instance.nadir_point = compute_nadir_point(instance.non_dominated_front) # TODO: Multiply by 1.1

            nadir_changed = instance.previous_nadir_point != instance.nadir_point
            metrics.record("nadir", {
                "generation": generation,
                "instance": instance_name,
                "nadir_point": list(instance.nadir_point),
                "changed": bool(nadir_changed),
                "consolidated_size": int(len(instance.non_dominated_front)),
            })

            hv_time_init = time.time()

            if nadir_changed:

                metrics.start("metrics_hv_recompute", category="metrics")
                for individual in self.population.evaluated():
                    front = individual.pareto_fronts[instance_name]
                    individual.hypervolumes[instance_name] = compute_hypervolume(instance.nadir_point, front)
                    metrics.record("hypervolumes", {
                        "generation": generation,
                        "instance": instance_name,
                        "individual": individual.name,
                        "hypervolume": float(individual.hypervolumes[instance_name]),
                        "front_size": int(front.shape[0]),
                        "recomputed": True,
                    })
                metrics.stop("metrics_hv_recompute")

            for individual in self.population.non_evaluated():
                front = individual.pareto_fronts[instance_name]

                individual.hypervolumes[instance_name] = compute_hypervolume(instance.nadir_point, front)
                metrics.record("hypervolumes", {
                    "generation": generation,
                    "instance": instance_name,
                    "individual": individual.name,
                    "hypervolume": float(individual.hypervolumes[instance_name]),
                    "front_size": int(front.shape[0]),
                    "recomputed": False,
                })

            hv_time = (time.time() - hv_time_init) / 60

            instance.previous_nadir_point = instance.nadir_point

            ranks = compute_rank(self.population.get_hypervolumes(instance_name))

            for j, individual in enumerate(self.population.individuals):
                individual.fitness_values[instance_name] = ranks[j]
                fitness_values[j][i] = ranks[j]

                metrics.record("ranks", {
                    "generation": generation,
                    "instance": instance_name,
                    "individual": individual.name,
                    "rank": ranks[j],
                    "hypervolume": float(individual.hypervolumes[instance_name]),
                })

            instance_sorting_time[i] = sorting_time
            instance_hv_time[i] = hv_time

            sorting_text = ", ".join([f"{i:05.1f} min" for i in instance_sorting_time])
            hv_text = ", ".join([f"{i:05.1f} min" for i in instance_hv_time])

            update_lines(["Sorting: " + sorting_text,
                        "Hypervolumes: " + hv_text])

        for individual in self.population.non_evaluated():
            individual.evaluated = True

        scores = [sum(fitness_values[i]) / len(self.instances) for i in range(len(self.population.individuals))]

        for individual, score in zip(self.population.individuals, scores):
            RunMetrics().record("fitness", {
                "generation": self.current_generation,
                "individual": individual.name,
                "fitness": float(score),
            })

        pbar.close()

        total_time = pbar.format_dict["elapsed"]

        logger.info(f"Metrics took: {total_time} seconds")
        logger.info(f"Sorting: {sorting_text}")
        logger.info(f"Hypervolumes: {hv_text}")

        return scores

    def evolutionary_step(self, scores: list):
        """
        Compute an evolutionary step.
        Selection, Crossover, Mutation.
        """

        genomes = copy.deepcopy(self.population.get_genomes())

        phenotypes = copy.deepcopy(self.population.get_phenotypes())

        genome_usage = copy.deepcopy(self.population.get_genome_usage())

        offspring = []

        metrics = RunMetrics()
        metrics.increment("ge", "attempted_parent_pairs", 0)
        metrics.start("ge_operators_total", category=None)

        while len(offspring) < self.num_offsprings:

            metrics.start("ge_selection", category="ge")
            parents, indices = self.selection(genomes, scores)
            metrics.stop("ge_selection")
            metrics.increment("ge", "attempted_parent_pairs", 1)
            relevant_genomes = [genome_usage[i] for i in indices]

            metrics.start("ge_crossover", category="ge")
            children = self.crossover(parents, relevant_genomes)
            metrics.stop("ge_crossover")

            metrics.start("ge_mutation", category="ge")
            children = self.mutation(children)
            metrics.stop("ge_mutation")

            for child in children:
                metrics.start("ge_mapping", category="ge")
                ind = Individual(child)
                metrics.stop("ge_mapping")

                if ind.invalid or ind.phenotype in phenotypes:
                    metrics.increment("ge", "rejected_invalid" if ind.invalid else "rejected_repeated", 1)
                    continue

                metrics.start("ge_validation", category="ge")
                tested = test_individual(ind.phenotype, self.solution_type)
                metrics.stop("ge_validation")

                if tested == False:
                    metrics.increment("ge", "rejected_invalid_output", 1)
                    continue

                offspring.append(ind)
                metrics.increment("ge", "offspring_created", 1)

                if len(offspring) == self.num_offsprings:
                    break

        metrics.stop("ge_operators_total")
        metrics.record("ge", {
            "generation": self.current_generation,
            "num_offsprings": self.num_offsprings,
            "offspring_created": len(offspring),
        })

        for child in offspring:

            child.is_offspring = True

            child.name = f"Individual_{self.population.num_individuals_created+1:06d}"

            self.population.num_individuals_created += 1

        self.population.individuals.extend(offspring)

    def replace_step(self, scores: list) -> list:
        """
        Apply replacement with elitism.
        """

        metrics = RunMetrics()
        metrics.start("ge_replacement", category="ge")

        population = self.population.get_population()

        offspring = self.population.get_offspring()

        elitism_size = len(population) - len(offspring)

        paired_population = list(zip(population, scores[:len(population)]))

        paired_offspring = list(zip(offspring, scores[len(population):]))

        sorted_population = sorted(paired_population, key=lambda x: x[1])

        preserved_population = sorted_population[:elitism_size]

        preserved_population.extend(paired_offspring)

        final_pairs = sorted(preserved_population, key=lambda x: x[1])

        new_population = [ind for ind, _ in final_pairs] 

        scores = [score for _, score in final_pairs]

        self.population.individuals = new_population

        for individual in self.population.individuals:
            individual.is_offspring = False

        metrics.stop("ge_replacement")

        return scores

    def _metrics_save_consolidated(self, generation: int):
        """
        Persist the consolidated (non-dominated union) fronts + nadir points
        of every instance into run_metrics/. Timed as I/O, so it never
        inflates the "pure computation" cost.
        """
        metrics = RunMetrics()
        for instance in self.instances:
            metrics.save_json(
                f"fronts/consolidated/generation_{generation:04d}/instance_{instance.instance_name}.json",
                {
                    "generation": generation,
                    "instance": instance.instance_name,
                    "nadir_point": list(instance.nadir_point),
                    "size": int(len(instance.non_dominated_front)),
                    "front": sorted(instance.non_dominated_front),
                },
            )

    def run(self):
        """
        Execute the generation hyperheuristic.
        """

        population_saver = PopulationSaver()

        logger_object = MyLogger()
        logger = logger_object.get_logger()

        logger.info(f"MO-Solver: {self.mo_model_name}")

        # no-op unless LOG_METRICS
        metrics = RunMetrics()
        metrics.set_info("phase", "generation")
        metrics.set_info("mo_model", self.mo_model_name)
        metrics.set_info("instances", [i.instance_name for i in self.instances])
        metrics.record("run", {"event": "start", "model": self.mo_model_name,
                               "num_instances": len(self.instances)})

        params = Params()
        mo_pop_sizes = self.mo_population_size if isinstance(self.mo_population_size, list) else [self.mo_population_size]
        mo_pop_display = mo_pop_sizes[0] if len(set(mo_pop_sizes)) == 1 else mo_pop_sizes

        metrics.set_info("params", params.get_params())

        print("\n" + "-" * 40)
        print("Starting Generation")
        print("-" * 40)
        print(f"MOEA model:         {self.mo_model_name}")
        print(f"Population size:    {self.population_size}")
        print(f"Generations:        {self.num_generations}")
        print(f"Elite size:         {params['ELITE_SIZE']}")
        print(f"MO population size: {mo_pop_display}")
        print(f"MO generations:     {params['MO_GENERATIONS']}")
        print("-" * 40)

        start_time = time.time()

        print("\n" + "-" * 40)
        print("Generation 0.")
        print("-" * 40)
        print("Start initial population.")
        logger.info(f"Generation 0/{self.num_generations}")

        self.current_generation = 0
        metrics.start("ge_initialization", category="ge")
        self.population.initialize_population(self.population_size)
        metrics.stop("ge_initialization")
        metrics.record("ge_initialization", {
            "generation": 0,
            "population_size": self.population_size,
            "individuals_created": self.population.num_individuals_created,
        })

        print("Evaluating initial population...")

        metrics.start("evaluation_step", category=None)
        self.evaluation_step()
        metrics.stop("evaluation_step")

        metrics.start("compute_metrics_step", category=None)
        scores = self.compute_metrics()
        metrics.stop("compute_metrics_step")

        metrics.start("io_saver", category="io")
        population_saver.save_population(self.population, generation=0)
        population_saver.save_consolidated_fronts(self.instances, generation=0)
        metrics.stop("io_saver")

        self._metrics_save_consolidated(0)
        metrics.dump("generation_0000", {"generation": 0})

        for gen in range(1, self.num_generations + 1):

            print("\n" + "-" * 40)
            print(f"Generation {gen}/{self.num_generations}")
            print("-" * 40)
            logger.info(f"Generation {gen}/{self.num_generations}")

            self.current_generation = gen
            metrics.start("generation_compute", category=None)

            self.evolutionary_step(scores)

            # pop is now N+n; only the n offspring are evaluated

            # consolidated may stop being non-dominated once offspring are added
            metrics.start("evaluation_step", category=None)
            self.evaluation_step()
            metrics.stop("evaluation_step")

            metrics.start("compute_metrics_step", category=None)
            scores = self.compute_metrics()
            metrics.stop("compute_metrics_step")

            metrics.stop("generation_compute")

            metrics.start("io_saver", category="io")
            population_saver.save_population(self.population, generation=gen-1)

            population_saver.save_consolidated_fronts(self.instances, generation=gen)
            metrics.stop("io_saver")

            scores = self.replace_step(scores)

            # replace drops the worst n

            metrics.start("io_saver", category="io")
            population_saver.save_population(self.population, generation=gen)
            metrics.stop("io_saver")

            self._metrics_save_consolidated(gen)
            metrics.set_info("best_individual", self.population.individuals[0].phenotype)
            metrics.dump(f"generation_{gen:04d}", {"generation": gen})

            print(f"\nBest Individual: {self.population.individuals[0].phenotype}")
            logger.info(f"Best Individual: {self.population.individuals[0].phenotype}")

        end = time.time()
        seconds = round(end - start_time, 2)
        minutes = round(seconds / 60, 2)
        hours = round(minutes / 60, 2)
        print(f"\n\nTotal time: {seconds} seconds.")
        print(f"Minutes: {minutes}")
        print(f"Hours: {hours}")
        logger.info(f"Total time in seconds: {seconds}")
        logger.info(f"Total time in minutes: {minutes}")
        logger.info(f"Total time in hours: {hours}")

        # write summary.json
        metrics.record("run", {"event": "end", "total_seconds": seconds})
        metrics.write_summary()
        metrics.close()

        population_saver.reset_instance()
        logger_object.reset_instance()
