

import copy
import numpy as np


from mohh.evaluation.params import Params


from abc import ABC, abstractmethod


from mohh.evaluation.problem.instance import Instance
from mohh.core.utilities.load_modules import find_module
from mohh.evaluation.operators.operator_template import NullMutation


from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions


from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA


from pymoo.core.callback import Callback


class MyCallback(Callback):

    def __init__(self, population_size: int, eval_budgets: dict):
        super().__init__()
        self.populations = {}

        # budget -> generation checkpoint
        self.checkpoints = {label: max(1, budget // population_size)
                            for label, budget in eval_budgets.items()}

    def notify(self, algorithm):

        for label, checkpoint_gen in self.checkpoints.items():

            if algorithm.n_gen == checkpoint_gen:
                results = algorithm.result()
                pareto_front = results.F

                pareto_front, mask = np.unique(pareto_front, return_index=True, axis=0)

                self.populations[label] = pareto_front


class MOSolver(ABC):

    def __init__(self) -> None:
        """
        Initialize solver for an operator
        """

        params = Params()

        self.generations = params["MO_GENERATIONS"]

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
            module_name = f"mohh.evaluation.operators.{operator_type}"
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

    @abstractmethod
    def start_model(self):
        """
        Start the MOEA.
        """
        return

    @abstractmethod
    def solve_instance(self):
        """
        Execute MOEA
        """
        return


class MOEA_Decomposition(MOSolver):

    def __init__(self) -> None:
        super().__init__()

    def start_model(self, seed):
        """
        Start the MOEA.
        """

        params = Params()

        ref_dirs = get_reference_directions("uniform", self.k_objectives, n_points=self.pop_size)

        self.algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=params["NUM_NEGIHBORS"],
            prob_neighbor_mating=params["PROB_NEIGHBORS"],
            sampling=self.pop,
            crossover=self.crossover,
            mutation=self.mutation
        )

        self.algorithm.setup(self.problem, termination=("n_gen", self.generations),
                             verbose=False, seed=seed,
                             callback=MyCallback(self.pop_size, params["EVAL_BUDGETS"]))

    def solve_instance(self):
        """
        MOEA execution
        """

        while self.algorithm.has_next():

            pop = self.algorithm.ask()

            self.algorithm.evaluator.eval(self.problem, pop)

            self.algorithm.tell(infills=pop)

        snapshots_data = self.algorithm.callback.populations

        return snapshots_data


class NSGAII(MOSolver):
    def __init__(self) -> None:
        super().__init__()

    def start_model(self, seed):
        """
        Start the MOEA.

        @param seed: Random seed.
        """

        params = Params()

        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=self.pop,
            crossover=self.crossover,
            mutation=self.mutation
        )
        # tournament + rank&crowding survival

        self.algorithm.setup(self.problem, termination=("n_gen", self.generations),
                             verbose=False, seed=seed,
                             callback=MyCallback(self.pop_size, params["EVAL_BUDGETS"]))

    def solve_instance(self):
        """
        MOEA execution
        """

        while self.algorithm.has_next():

            pop = self.algorithm.ask()

            self.algorithm.evaluator.eval(self.problem, pop)

            self.algorithm.tell(infills=pop)

        snapshots_data = self.algorithm.callback.populations

        return snapshots_data


class SMS_MOEA(MOSolver):
    def __init__(self) -> None:
        super().__init__()

    def start_model(self, seed):
        """
        Start the MOEA.

        @param seed: Random seed.
        """

        params = Params()

        self.algorithm = SMSEMOA(
            pop_size=self.pop_size,
            sampling=self.pop,
            crossover=self.crossover,
            mutation=self.mutation
        )
        # tournament + HV-contribution survival

        self.algorithm.setup(self.problem, termination=("n_gen", self.generations),
                             verbose=False, seed=seed,
                             callback=MyCallback(self.pop_size, params["EVAL_BUDGETS"]))

    def solve_instance(self):
        """
        MOEA execution
        """

        while self.algorithm.has_next():

            pop = self.algorithm.ask()

            self.algorithm.evaluator.eval(self.problem, pop)

            self.algorithm.tell(infills=pop)

        snapshots_data = self.algorithm.callback.populations

        return snapshots_data

