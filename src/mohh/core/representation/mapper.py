
import numpy as np

from mohh.core.params import Params
from collections import deque
  

def mapper(genome: list):
    """
    Maps a genome into a phenotype, via the grammar.

    @param genome: Individual's genome.
    """

    genome = list(genome)

    phenotype, genome, tree, nodes, invalid, depth, \
    used_codons = map_ind_from_genome(genome)

    if invalid:
        phenotype, nodes, depth, used_codons = None, np.nan, np.nan, np.nan

    return phenotype, genome, tree, nodes, invalid, depth, used_codons

def map_ind_from_genome(genome: list) -> tuple:
    """
    Code adapted from PonyGE2 (github.com/PonyGE/PonyGE2).

    Making use of the genome, create an individual.
    """

    params = Params()

    max_tree_depth, max_wraps = params["MAX_TREE_DEPTH"], params["MAX_WRAPS"]
    bnf_grammar = params["BNF_GRAMMAR"]

    n_input = len(genome)

    used_input, current_depth, max_depth, nodes, wraps = 0, 1, 1, 1, -1

    output = deque()

    unexpanded_symbols = deque([(bnf_grammar.start_rule, 1)])

    while (wraps < max_wraps) and unexpanded_symbols:

        if max_tree_depth and (max_depth > max_tree_depth):
            # max tree depth exceeded
            break

        if used_input % n_input == 0 and \
            used_input > 0 and \
            any([i[0]["type"] == "NT" for i in unexpanded_symbols]):
            # genome exhausted: wrap around
            wraps += 1

        current_item = unexpanded_symbols.popleft()
        current_symbol, current_depth = current_item[0], current_item[1]

        if max_depth < current_depth:
            max_depth = current_depth

        if current_symbol["type"] != "NT":
            output.append(current_symbol["symbol"])

        else:
            production_choices = bnf_grammar.rules[current_symbol[
                "symbol"]]["choices"]
            no_choices = bnf_grammar.rules[current_symbol["symbol"]][
                "no_choices"]

            current_production = genome[used_input % n_input] % no_choices

            used_input += 1

            children = deque()
            nt_count = 0

            for prod in production_choices[current_production]['choice']:

                child = [prod, current_depth + 1]

                # extendleft reverses order
                children.appendleft(child)
                if child[0]["type"] == "NT":
                    nt_count += 1

            unexpanded_symbols.extendleft(children)

            if nt_count > 0:
                nodes += nt_count
            else:
                nodes += 1

    output = "".join(output)

    if len(unexpanded_symbols) > 0:
        # not fully expanded: invalid
        return None, genome, None, nodes, True, max_depth, used_input

    return output, genome, None, nodes, False, max_depth, used_input

