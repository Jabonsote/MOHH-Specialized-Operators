
import os
import random

from datetime import datetime
from socket import gethostname

from mohh.core.params import Params
from mohh.core.utilities.paths import project_root

hostname = gethostname().split(".")
machine_name = hostname[0]

hh_params = {
    'POPULATION_SIZE': 50,
    'GENERATIONS': 30,
    'ELITE_SIZE': 15,

    'MAX_TREE_DEPTH': 90,
    'MAX_INIT_TREE_DEPTH': 10,
    'MIN_INIT_TREE_DEPTH': None,

    'MAX_TREE_NODES': None,
    
    'CODON_SIZE': 255,
    'INIT_GENOME_LENGTH': 100,
    'MAX_GENOME_LENGTH': None,
    'MAX_WRAPS': 0,

    'INITIALIZATION': "Rvd",
    
    'SELECTION': "Tournament",
    'TOURNAMENT_SIZE': 3,

    'CROSSOVER': 'KPointCrossover',
    'CROSSOVER_PROBABILITY': 1.0,
    'CROSSOVER_K_POINTS': 2,
    'CROSSOVER_PARENTS': 2,

    'MUTATION': 'BitFlipMutation',
    'MUTATION_PROBABILITY': 0.01,

    'REPLACEMENT': "HHReplacement"
}

mo_params = {
    'MO_MODEL': "MOEAD",

    'NUM_NEGIHBORS': 10,
    'PROB_NEIGHBORS': 0.9,

    'MO_POPULATION_SIZE': 105,
    'MO_GENERATIONS': 300,

    'MO_CROSSOVER': None,
    'MO_CROSSOVER_PROBABILITY': 1.0,

    'MO_MUTATION': "SwapMutation",
    'MO_MUTATION_PROBABILITY': 0.01,
    'MO_MUTATION_BOOL': False,
}

problem_params = {
    'PROBLEM_NAME': "QAP",
    'SOLUTION_TYPE': "Natural",
    # maximize if True
    'OPTIMIZATION_KIND': False,

    'DATASET': "mqap",
    'DATASET_TRAIN': "train",
    'DATASET_TEST': None,
}

grammar_params = {
    'GRAMMAR_FILE': "naturals.bnf",

    # depths used by the grammar analyser
    'PERMUTATION_RAMPS': 5,
}

misc_params = {
    'MACHINE': machine_name,

    "SAVE": True,

    'EXPERIMENT_NAME': None,
    'RANDOM_SEED': None,

    'LOG_METRICS': True,
    'METRICS_DIRNAME': "run_metrics",
    'METRICS_PATH': None,
}

params_dict = dict(hh_params, **mo_params, **problem_params, **grammar_params, **misc_params)

def set_params(experiment_path: str, current_model: str, overrides: dict = None):

    params = Params()
    params.update_params(params_dict)

    if overrides:
        params.update_params(overrides)

    from mohh.core.representation import grammar
    from mohh.generation.saver import PopulationSaver, MyLogger

    start = datetime.now()

    if params["RANDOM_SEED"] is None:
        params["RANDOM_SEED"] = int(start.microsecond)

    random.seed(params["RANDOM_SEED"])

    hm = "%02d%02d" % (start.hour, start.minute)
    params["TIME_STAMP"] = "_".join([gethostname(),
                                     str(start.year),
                                     str(start.month),
                                     str(start.day), hm,
                                     str(params["RANDOM_SEED"])])
    
    print("\nStart:\t", start, "\n")

    if params["SAVE"]:
        PopulationSaver(experiment_path, current_model)
        logger = MyLogger().get_logger()
        logger.info("Start")

    params["GENERATION_SIZE"] = params["POPULATION_SIZE"] - \
                                params["ELITE_SIZE"]
    
    params["BNF_GRAMMAR"] = grammar.Grammar(
        os.path.join(project_root(), "grammars", params["GRAMMAR_FILE"])
    )
