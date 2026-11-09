

import os
import random


from datetime import datetime
from socket import gethostname


from mohh.core.params import Params
from mohh.core.utilities.paths import project_root


hostname = gethostname().split(".")
machine_name = hostname[0]


mo_params = {

    'SOLVERS': ["MOEAD", "NSGAII", "SMSEMOA"],

    'N_EXPERIMENTS': 31,

    # 105 (pop) * 100/300/500 generations
    'EVAL_BUDGETS': {"10k": 10500, "30k": 31500, "50k": 52500},

    'MO_POPULATION_SIZE': 105,
    'MO_GENERATIONS': 500,

    'NUM_NEGIHBORS': 10,
    'PROB_NEIGHBORS': 0.9,

    'MOEAD_CROSSOVER': None, # SBX
    'MOEAD_CROSSOVER_PROBABILITY': 1.0,

    'MOEAD_MUTATION': "PM_Mutation", # PM
    'MOEAD_MUTATION_PROBABILITY': 0.05,
    'MOEAD_MUTATION_BOOL': False,

    'NSGA_CROSSOVER': None, # SBX
    'NSGA_CROSSOVER_PROBABILITY': 1.0,

    'NSGA_MUTATION': "PM_Mutation", # PM
    'NSGA_MUTATION_PROBABILITY': 0.05,
    'NSGA_MUTATION_BOOL': False,

    'SMS_CROSSOVER': None,
    'SMS_CROSSOVER_PROBABILITY': 1.0,

    'SMS_MUTATION': "PM_Mutation",
    'SMS_MUTATION_PROBABILITY': 0.05,
    'SMS_MUTATION_BOOL': False


}


problem_params = {
    'PROBLEM_NAME': "QAP",
    'SOLUTION_TYPE': "Real",
    # maximize if True
    'OPTIMIZATION_KIND': False,

    'DATASET': "mqap",
    'DATASET_TRAIN': None,
    'DATASET_TEST': "test",
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


params_dict = dict(mo_params, **problem_params, **grammar_params, **misc_params)


def set_params(overrides: dict = None):

    params = Params()
    params.update_params(params_dict)

    if overrides:
        params.update_params(overrides)

    from mohh.core.representation import grammar

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

    params["BNF_GRAMMAR"] = grammar.Grammar(
        os.path.join(project_root(), "grammars", params["GRAMMAR_FILE"])
    )

