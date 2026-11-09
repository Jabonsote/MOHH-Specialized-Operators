

import os


from mohh.evaluation.params import Params
from mohh.core.problem.instance import Instance as _Instance
from mohh.core.problem.instance import QuadraticAssignment, MOSampling


class Instance(_Instance):

    def __init__(self, population_size: int, solution_type: str) -> None:
        super().__init__(population_size, solution_type)

        self.initial_solutions_experiments: dict = {}

    def save_initial_solutions(self):

        params = Params()

        file_path = params["FILE_PATH_INITIAL_SOLUTIONS"]

        instance_level = os.path.join(file_path, "real", self.file_name)
        instance_level_permutations = os.path.join(file_path, "permutation", self.file_name)
        os.makedirs(instance_level, exist_ok=True)
        os.makedirs(instance_level_permutations, exist_ok=True)

        for experiment in self.initial_solutions_experiments.keys():

            initial_solutions = self.initial_solutions_experiments[experiment]
            experiment = experiment.lower()

            with open(os.path.join(instance_level, experiment + ".txt"), "w") as f:

                for i in range(self.population_size):
                    sol = initial_solutions[i].tolist()

                    sol = ", ".join(["{:.04f}".format(w) for w in sol])

                    sol = "[" + sol + "]\n"

                    f.write(sol)

                f.close()

        for experiment in self.initial_solutions_permutations_experiments.keys():

            initial_solutions = self.initial_solutions_permutations_experiments[experiment]
            experiment = experiment.lower()

            with open(os.path.join(instance_level_permutations, experiment + ".txt"), "w") as f:

                for i in range(self.population_size):
                    sol = initial_solutions[i]

                    sol = ", ".join(["{:03d}".format(w) for w in sol])

                    sol = "[" + sol + "]\n"

                    f.write(sol)

                f.close()

    def create_initial_solution(self) -> None:
        """
        Generates an initial solution.
        It must be executed only one time per instance.
        """

        generator = MOSampling(kind="real")
        initial_solutions = generator.generate(self.n_variables,
                                               self.population_size)

        initial_solutions_permutation = [self._decode_random_keys(sol) for sol in initial_solutions]

        return initial_solutions, initial_solutions_permutation

    def create_solutions(self, n_experiments: int) -> None:
        """
        Generates N initial solutions.
        """

        initial_solutions = {}
        initial_solutions_permutations = {}

        for experiment in range(1, n_experiments+1):

            solutions, permutations = self.create_initial_solution()

            initial_solutions[f"Experiment_{experiment:03d}"] = solutions
            initial_solutions_permutations[f"Experiment_{experiment:03d}"] = permutations

        self.initial_solutions_experiments = initial_solutions
        self.initial_solutions_permutations_experiments = initial_solutions_permutations

    def load_solutions(self, file_path: str, n_experiments: int) -> None:
        """
        Load initial solutions
        """

        initial_solutions = {}
        initial_solutions_permutations = {}

        for experiment in range(1, n_experiments+1):

            real_sol_path = os.path.join(file_path, "real", self.file_name, f"experiment_{experiment:03d}.txt")
            perm_sol_path = os.path.join(file_path, "permutation", self.file_name, f"experiment_{experiment:03d}.txt")

            try:
                solutions = self.load_initial_solution(real_sol_path)
                initial_solutions[f"Experiment_{experiment:03d}"] = solutions

            except:
                pass

            try:
                permutations = self.load_initial_solution(perm_sol_path, permutation=True)

            except:
                permutations = [self._decode_random_keys(sol) for sol in solutions]

            initial_solutions_permutations[f"Experiment_{experiment:03d}"] = permutations

        self.initial_solutions_experiments = initial_solutions
        self.initial_solutions_permutations_experiments = initial_solutions_permutations

    def load_problem(self, problem_type: str, file_name: str,
                     n_experiments:int, init_solutions_path: str = "") -> None:
        """
        Loads an instance file.

        @param problem_type: QAP or TSP.
        @param file_name: Instance file path.
        @param n_experiments: Number of initial populations to create/load, one per experiment.
        @param init_solutions_path: Path to a saved initial population, if any.
        """

        self.file_path = file_name.replace("\\", "/")
        self.file_name = self.file_path.split("/")[-1].removesuffix(".txt")
        self.instance_name = self.file_name

        if problem_type.lower() == "qap":
            self._extract_qap_parameters()

            if init_solutions_path:
                self.load_solutions(init_solutions_path, n_experiments)

            else:
                self.create_solutions(n_experiments)

            self.save_initial_solutions()

            self.previous_nadir_point = [0] * self.k_objectives

            self.problem = QuadraticAssignment(self.n_variables, self.k_objectives,
                                                self.file_path, self.solution_type)

        elif problem_type.lower() == "tsp":
            pass

    def set_initial_solutions(self, experiment: int, kind: str="real"):
        """
        Set initial solutions for its respective experiment
        """

        if kind == "real":
            solutions = self.initial_solutions_experiments[f"Experiment_{experiment:03d}"]

        else:
            solutions = self.initial_solutions_permutations_experiments[f"Experiment_{experiment:03d}"]

        self.problem.kind_solutions = kind

        self.initial_solutions = solutions

