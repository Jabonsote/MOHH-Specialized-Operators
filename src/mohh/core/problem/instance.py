

import os
import numpy as np


from mohh.core.params import Params
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling


class Instance():

    num_instances_created = 0

    def __init__(self, population_size: int, solution_type: str) -> None:
        """
        Creates an empty instance object.

        @param population_size: Number of solutions per instance.
        @param solution_type: Natural or Real.
        """
        self.n_variables: int = None
        self.k_objectives: int = None
        self.identification: int = None
        self.kind: str = None

        self.file_name: str = None
        self.file_path: str = None
        self.instance_name: str = None

        self.solution_type: str = solution_type.lower()

        self.population_size :int = population_size
        self.initial_solutions: np.ndarray = None

        self.fronts: list = []
        self.non_dominated_front: set = set()
        self.nadir_point: list = []
        self.previous_nadir_point: list = []

        Instance.num_instances_created += 1
        self.instance_id :str = f"Instance_{Instance.num_instances_created:03d}"

    def save_initial_solutions(self):

        """
        Save in disk the initial solutions
        """

        params = Params()

        file_path = params["FILE_PATH_INITIAL_SOLUTIONS"]

        if self.solution_type != "real":
            instance_level = os.path.join(file_path, "permutation")
        else:
            instance_level = os.path.join(file_path, "real")

        os.makedirs(instance_level, exist_ok=True)

        save_path = os.path.join(instance_level, self.file_name + ".txt")

        with open(save_path, "w") as f:

            for i in range(self.population_size):
                sol = self.initial_solutions[i].tolist()

                if self.solution_type == "real":
                    sol = ", ".join(["{:.04f}".format(w) for w in sol])
                else:
                    sol = ", ".join(["{:03d}".format(w) for w in sol])

                sol = "[" + sol + "]\n"

                f.write(sol)
            f.close()

    def create_initial_solution(self) -> None:
        """
        Generates an initial solution.
        It must be executed only one time per instance.
        """

        generator = MOSampling(kind=self.solution_type)
        initial_solutions = generator.generate(self.n_variables,
                                                self.population_size)

        self.initial_solutions = initial_solutions

    def load_initial_solution(self, file_path: str, permutation: bool = False):

        data = []
        with open(file_path, "r") as file:

            if not(permutation):

                for line in file:
                    data.append(eval(line))

            else:
                for line in file:
                    line = line.strip()[1:-1]
                    p = [int(value, 10) for value in line.split(",")]
                    data.append(p)

        return np.array(data)

    def load_solutions(self, file_path: str) -> None:
        """
        Load initial solutions
        """

        if self.solution_type == "real":
            sol_path = os.path.join(file_path, "real", self.file_name + ".txt")
            permutation = False
        else:
            sol_path = os.path.join(file_path, "permutation", self.file_name + ".txt")
            permutation = True

        try:
            solutions = self.load_initial_solution(sol_path, permutation)
            self.initial_solutions = solutions
        except Exception as e:
            print(e)
            raise(ValueError)

    def load_problem(self, problem_type: str, file_name: str,
                     init_solutions_path: str = "") -> None:
        """
        Loads an instance file.

        @param problem_type: QAP or TSP.
        @param file_name: Instance file path.
        @param init_solutions_path: Path to a saved initial population, if any.
        """

        self.file_path = file_name.replace("\\", "/")
        self.file_name = self.file_path.split("/")[-1].removesuffix(".txt")
        self.instance_name = self.file_name

        if problem_type.lower() == "qap":
            self._extract_qap_parameters()

            if init_solutions_path:
                self.load_solutions(init_solutions_path)

            else:
                self.create_initial_solution()

            self.save_initial_solutions() 

            self.previous_nadir_point = [0] * self.k_objectives

            self.problem = QuadraticAssignment(self.n_variables, self.k_objectives,
                                                self.file_path, self.solution_type)

        elif problem_type.lower() == "tsp":
            pass

    def _extract_qap_parameters(self) -> None:
        """
        Extract parameters from file names.
        """
        instance_name = self.file_name

        parameters: list = instance_name.split("-")

        self.n_variables: int = int(parameters[0][2:])

        self.k_objectives: int = int(parameters[1][:-2])

        self.identification: int = int(parameters[2][:-2])

        self.kind: str = parameters[2][1:]

    def _decode_random_keys(self, solution: np.ndarray) -> list:
        """
        Auxiliar function to decode random_keys into natural numbers.

        @param solution: Numpy array with float numbers.

        :return: List with int numbers.
        """
        augmented_keys = list(zip([i for i in range(len(solution))], solution))

        augmented_keys.sort(key = lambda x: x[1])

        return list(zip(*augmented_keys))[0]


class QuadraticAssignment(Problem):

    def __init__(self, n_var: int, k_obj: int, file_name: str,
                kind_solutions: str, **kwargs):
        """
        QAP instance class for Pymoo.

        @param n_var: Number of variables.
        @param k_obj: Number of objective matrices.
        @param file_name: Instance file path.
        @param kind_solutions: Natural or Real.

        :return: QAP Instance.
        """
        super().__init__(n_var=n_var, n_obj=k_obj, **kwargs)

        self.k_obj = k_obj
        self.n_var = n_var

        self.weights = None
        self.positions = None

        self.kind_solutions = kind_solutions

        self.init_with_instance_file(file_name)

    def init_with_instance_file(self, instance_file: str) -> None:
        """
        Loads the positions of the factories and their weights in numpy arrays.

        @param instance_file: Instance path to load.
        """
        if isinstance(instance_file, str):
            self.instance_file = instance_file
        else:
            raise TypeError("Instance_file must be a string")

        file = open(self.instance_file, 'r')
        file_content = list(map(str.strip, file.readlines()))
        file.close()

        parameters = file_content[0].replace("=", " ").split()

        n_variables = int(parameters[1])
        k_objectives = int(parameters[3])

        file_content = list(map(lambda x: " ".join(x.split()) , file_content[1:]))

        positions = np.array( [list(map(int, line.split(" "))) for line in file_content[:n_variables]] )

        # flux: (k, n, n)
        flux = np.zeros((k_objectives, n_variables, n_variables))

        for i in range(k_objectives):
            start = (i+1)*n_variables + 1 + i
            end = (i+2)*n_variables + 1 + i

            flux[i] = np.array( [list(map(int, line.split(" "))) for line in file_content[start:end]] )

        self.weights = flux
        self.positions = positions

    def decode_random_keys(self, solution:np.ndarray) -> list:
        """
        Auxiliar function to decode random_keys into natural numbers.

        @param solution: Numpy array with float numbers.

        :return: List with int numbers.
        """
        augmented_keys = list(zip([i for i in range(len(solution))], solution))

        augmented_keys.sort(key = lambda x: x[1])

        return list(zip(*augmented_keys))[0]

    def _evaluate(self, x:np.ndarray, out:dict, *args, **kwargs):
        """
        Evaluates the population and fills out["F"].

        Complexity: O(p x k x n^2).

        @param x: Population.
        @param out: Dictionary to update.
        """
        population_size, n_var = x.shape

        values = np.zeros((population_size, self.k_obj))

        if self.kind_solutions.lower() == "real":
            solutions = np.array([self.decode_random_keys(x[i,:].copy()) for i in range(population_size)])
        else:
            solutions = x

        for i in range(population_size):
            perm = solutions[i]
            reordered_dist = self.positions[np.ix_(perm, perm)]
            values[i] = np.einsum('kij, ij -> k', self.weights, reordered_dist)

        out["F"] = values


class MOSampling(Sampling):

    def __init__(self, kind: str="real") -> None:
        """
        @param kind: natural or real
        """
        super().__init__()

        self.kind = kind.lower()

    def generate(self, n_var: int, n_samples: int) -> np.ndarray:
        """
        Generate initial solutions.

        @param n_var: Number of variables.
        @param n_samples: Number of solutions.

        :return: Initial solutions in a numpy array.
        """
        if self.kind == "real":
            X = np.random.random((n_samples, n_var))

        else:
            X = np.zeros((n_samples, n_var), dtype=int)
            for i in range(n_samples):
                X[i] = np.random.permutation(n_var)

        return X

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        """
        Function for pymoo functionality.
        """

        n_variables = problem.n_var

        X = self.generate(n_variables, n_samples)

        return X

