

import json
import hashlib
import logging
import numpy as np


from mohh.generation.params import Params
from os import getcwd, makedirs, path


class PopulationSaver():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PopulationSaver, cls).__new__(cls)
        return cls._instance

    def __init__(self, experiment_path: str = None,
                current_model: str = None, file_format: str="json") -> None:
        if not hasattr(self, "initialized"):

            params = Params()
            self.base_save_dir = params[""]
            self.file_format = file_format
            self.individuals_dir = None
            self.generations_dir = None
            self.initial_solutions_dir = None
            self.existing_individuals = set()

            self._initialize_directories(experiment_path, current_model)
            self.initialized = True

    def _initialize_directories(self, experiment_path: str, current_model: str):
        """
        Generates neccesary folders and files for saving statistics and parameters.
        """

        params = Params()

        if params['EXPERIMENT_NAME']:
            path_1 = path.join(getcwd(), "results")

            if not path.isdir(path_1):
                makedirs(path_1, exist_ok=True)

            params['FILE_PATH'] = path.join(path_1, params['EXPERIMENT_NAME'])

        else:
            params['FILE_PATH'] = path.join(getcwd(), experiment_path)

        if not path.isdir(params['FILE_PATH']):
            makedirs(params['FILE_PATH'], exist_ok=True)

        if not path.isdir(path.join(params['FILE_PATH'],
                                    current_model + "_" + str(params['TIME_STAMP']))):
            makedirs(path.join(params['FILE_PATH'],
                               current_model + "_" + str(params['TIME_STAMP'])), exist_ok=True)

        params['FILE_PATH'] = path.join(params['FILE_PATH'],
                                        current_model + "_" + str(params['TIME_STAMP']), "generation_results")

        for name in ["individuals", "initial_solutions", "generations"]:

            if not path.isdir(path.join(params["FILE_PATH"], name)):
                makedirs(path.join(params["FILE_PATH"], name), exist_ok=True)

            params["FILE_PATH_"+name.upper()] = path.join(params["FILE_PATH"], name)

            setattr(self, name + "_dir", params["FILE_PATH_" + name.upper()])

        if params["SAVE"] and params["LOG_METRICS"]:
            from mohh.core.utilities.metrics import RunMetrics

            metrics_root = path.join(path.dirname(params["FILE_PATH"]),
                                     params["METRICS_DIRNAME"])
            params["METRICS_PATH"] = metrics_root

            RunMetrics().init(metrics_root, phase="generation")
            RunMetrics().set_info("model", current_model)

        self._save_params_to_file()

    def _save_params_to_file(self):
        """
        Save evolutionary parameters in a parameters.txt file.

        :return: Nothing.
        """

        params = Params()

        filename = path.join(params['FILE_PATH'], "parameters.txt")
        savefile = open(filename, 'w')

        col_width = max(len(param) for param in params.keys())

        for param in sorted(params.keys()):
            spaces = [" " for _ in range(col_width - len(param))]
            savefile.write(str(param) + ": " + "".join(spaces) +
                           str(params[param]) + "\n")

        savefile.close()

    def _generate_individual_id(self, individual):
        # hash of the genome
        unique_string = json.dumps(individual.genome)
        return hashlib.md5(unique_string.encode()).hexdigest()

    def save_consolidated_fronts(self, instances, generation: int):

        generation_dir = path.join(self.generations_dir, f"generation_{generation:04d}")
        makedirs(generation_dir, exist_ok=True)

        for instance in instances:
            instance_name = instance.instance_name

            fronts = instance.fronts

            consolidated = instance.non_dominated_front

            data = {f"Front_{i:03d}": front for i, front in enumerate(fronts)}
            data["consolidated"] = list(consolidated)
            data["nadir_point"] = instance.nadir_point
            data["num_variables"] = instance.n_variables
            data["num_objectives"] = instance.k_objectives

            save_path = path.join(generation_dir, f"Instance_Fronts_{instance_name}.{self.file_format}")

            with open(save_path, "w") as file:
                json.dump(data, file, indent=4)

    def save_population(self, population, generation: int):

        offspring_data = []
        population_data = []
        generation_dir = path.join(self.generations_dir, f"generation_{generation:04d}")
        makedirs(generation_dir, exist_ok=True)

        for individual in population.individuals:
            individual_id = individual.name
            individual_dir = path.join(self.individuals_dir, individual_id)

            if individual_id not in self.existing_individuals:
                makedirs(individual_dir, exist_ok=True)

                self._save_individual_general_info(individual, individual_dir, generation)

                self.existing_individuals.add(individual_id)

                for instance_id in individual.pareto_fronts.keys():

                    self._save_individual_instance_data(individual, instance_id, individual_dir)

            if individual.is_offspring:
                offspring_data.append(individual_id)
            else:
                population_data.append(individual_id)

        if offspring_data:
            self._save_generation_data(offspring_data, generation_dir, offspring=True)

        else:
            self._save_generation_data(population_data, generation_dir)

    def _save_individual_general_info(self, individual, individual_dir, generation):
        individual_info = {
            "name":individual.name,
            "phenotype":individual.phenotype,
            "number_nodes":individual.nodes,
            "codons_usage":individual.used_codons,
            "depth":individual.depth,
            "born_in_generation": generation,
            "genome":individual.genome,
        }

        save_path = path.join(individual_dir, f"general_info.{self.file_format}")
        with open(save_path, "w") as file:
            json.dump(individual_info, file, indent=4)

    def _save_individual_instance_data(self, individual, instance_name, individual_dir):

        front = individual.pareto_fronts[instance_name].copy()
        sol_set = individual.pareto_sets[instance_name].copy()

        if sol_set.dtype == float:
            sol_set = np.round(sol_set, 4)

        front_list = front.tolist()
        sol_set_list = sol_set.tolist()

        data = {
            "number_unique_solutions": individual.unique_solutions[instance_name],
            "weak_non_dominated_solutions":individual.weak_non_dominated[instance_name],
            "strong_non_dominated_solutions":individual.strong_non_dominated[instance_name],
            "front":front_list,
            "solution_set":sol_set_list,
        }

        save_path = path.join(individual_dir, f"instance_{instance_name}.{self.file_format}")
        with open(save_path, "w") as file:
            json.dump(data, file, indent=4)

    def _save_generation_data(self, generation_data, generation_dir, offspring: bool = False):

        name_dir = "offspring" if offspring else "population"

        population_file_path = path.join(generation_dir, f"{name_dir}.{self.file_format}")

        with open(population_file_path, "w") as file:
            json.dump(generation_data, file, indent=4)

    @classmethod
    def reset_instance(cls):
        cls._instance = None


class MyLogger():

    _instance = None

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super(MyLogger, cls).__new__(cls)

        return cls._instance

    def __init__(self):

        if hasattr(self, "logger"):
            return

        params = Params()
        experiment_path = params["FILE_PATH"]
        log_path = path.join(experiment_path, "CLI_output.log")

        self.logger = logging.getLogger("MyLogger")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s :: %(message)s")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def close_logger(self) -> None:
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    @classmethod
    def reset_instance(cls):

        if cls._instance is not None:
            cls._instance.close_logger()
            cls._instance = None

