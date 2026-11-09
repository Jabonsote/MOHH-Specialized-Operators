
import numpy as np
from mohh.core.utilities.algorithm.HH_functions import *


def compute_rank(hypervolumes: list) -> list:

    sorted_values = sorted(hypervolumes, reverse=True)

    ranks = dict()
    for i, value in enumerate(sorted_values):
        temp = ranks.get(value, [])
        temp.append(i+1)
        ranks[value] = temp

    average_ranks = {value: sum(rank) / len(rank) for value, rank in ranks.items()}

    ranked_values = [average_ranks[value] for value in hypervolumes]

    return ranked_values


def test_individual(expression: str, sol_type: str):
    """
    Verify if an individual phenotype returns invalid solutions.

    @param expression: Individual phenotype.
    @param sol_type: Natural or Real, type of solution to test.
    """

    if sol_type.lower() == "real":
        x = np.random.random(4)
        y = np.random.random(4)
    else:
        x = np.random.permutation(4)
        y = np.random.permutation(4)

    try:
        sol = eval(expression)

        if sol_type == "natural" and not(is_permutation(sol)):
            return False
        
        return True
    
    except:
        return False


def is_permutation(array: np.ndarray):
    """
    Check if the array has the correct length
    and contains unique elements.
    """
    return set(array) == set(range(len(array)))