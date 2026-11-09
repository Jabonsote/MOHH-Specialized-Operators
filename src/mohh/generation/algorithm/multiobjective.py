import copy
from math import comb
import numpy as np

from mohh.generation.params import Params
from mohh.generation.problem.instance import Instance
from mohh.core.utilities.load_modules import find_module
from mohh.generation.operators.operator_template import NullMutation

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions


class MOSolver:

    def __init__(self, seed) -> None:
        """
        Initialize solver for an individual.
        """

        params = Params()

        self.generations = params["MO_GENERATIONS"]
        self.seed = seed

        self.crossover = None
        self.mutation = None

        self.pop = None
        self.problem = None

    def load_operator(self, operator_type: str, operator_name: str, **kwargs: dict):
        """
        Loads a genetic operator to the algorithm.
        Choose between: crossover and mutation

        @param operator_type: Choose between: crossover and mutation.
        @param operator_name: Name of the operator.
        """

        if operator_type == "mutation" and operator_name == "NullMutation":
            self.mutation = NullMutation()

        else:
            module_name = f"mohh.generation.operators.{operator_type}"
            operator = find_module(module_name, operator_name, **kwargs)

            operator_type = "crossover" if operator_type == "operator_template" else operator_type
            setattr(self, operator_type, operator)

        return self

    def load_instance(self, instance: Instance):
        """
        Load an instance object and copy its initial solution.

        @param instance: Instance object.
        """

        self.problem = copy.deepcopy(instance.problem)

        self.pop = Population.new("X", instance.initial_solutions.copy())

        self.pop_size = instance.population_size
        self.k_objectives = instance.k_objectives

        Evaluator().eval(self.problem, self.pop)

    def start_model(self, model_name: str):
        """
        Load the MOEA.

        @param model_name: MOEA solver name.
        """

        params = Params()

        if model_name == "MOEAD":

            n_partitions = min(
                range(1, self.pop_size + 1),
                key=lambda p: abs(
                    comb(p + self.k_objectives - 1, self.k_objectives - 1)
                    - self.pop_size
                )
            )

            ref_dirs = get_reference_directions(
                "uniform",
                self.k_objectives,
                n_partitions=n_partitions
            )

            self.pop_size = len(ref_dirs)

            # resize pop to match ref_dirs
            if len(self.pop) > self.pop_size:
                self.pop = self.pop[:self.pop_size]
            elif len(self.pop) < self.pop_size:
                diff = self.pop_size - len(self.pop)
                extra = Population.new("X", self.pop[:diff].get("X"))
                Evaluator().eval(self.problem, extra)
                self.pop = Population.merge(self.pop, extra)

            self.algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=params["NUM_NEGIHBORS"],
                prob_neighbor_mating=params["PROB_NEIGHBORS"],
                sampling=self.pop,
                crossover=self.crossover,
                mutation=self.mutation
            )

        elif model_name == "NSGAII":

            self.algorithm = NSGA2(
                pop_size=self.pop_size,
                sampling=self.pop,
                crossover=self.crossover,
                mutation=self.mutation
            )

        elif model_name == "SMSEMOA":

            self.algorithm = SMSEMOA(
                pop_size=self.pop_size,
                sampling=self.pop,
                crossover=self.crossover,
                mutation=self.mutation
            )

        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        self.algorithm.setup(
            self.problem,
            termination=("n_gen", self.generations),
            verbose=False,
            seed=self.seed
        )

    def solve_instance(self):
        """
        MOEA process.
        """

        while self.algorithm.has_next():

            pop = self.algorithm.ask()

            try:
                self.algorithm.evaluator.eval(self.problem, pop)

            except TypeError as e:

                if "NoneType" in str(e):
                    # SMS-EMOA: empty population
                    break

                else:
                    print(e)
                    raise

            except Exception as e:
                print(e)
                raise

            self.algorithm.tell(infills=pop)

        results = self.algorithm.result()
        pareto_set = results.X
        pareto_front = results.F

        number_weak_non_dominated = len(pareto_set)

        number_unique_results = len(np.unique(pareto_set, axis=0))

        pareto_front, mask = np.unique(pareto_front, return_index=True, axis=0)
        pareto_set = pareto_set[mask]

        number_unique_pareto_front = len(pareto_front)

        return (
            pareto_set,
            pareto_front,
            number_weak_non_dominated,
            number_unique_results,
            number_unique_pareto_front
        )
