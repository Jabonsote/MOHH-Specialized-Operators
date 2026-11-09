
from math import floor
from sys import maxsize
from re import DOTALL, MULTILINE, finditer, match

from mohh.core.params import Params

params = Params()
  

class Grammar(object):
    """
    Parser for Backus-Naur Form.

    Adapted from PonyGE2 (github.com/PonyGE/PonyGE2).
    """

    def __init__(self, file_name):
        """
        Initialises an instance of the grammar class. This instance is used
        to parse a given file_name grammar.

        :param file_name: A specified BNF grammar file.
        """

        if file_name.endswith("pybnf"):
            self.python_mode = True

        else:
            self.python_mode = False

        self.rules, self.permutations = {}, {}

        self.non_terminals, self.terminals = {}, {}
        self.start_rule, self.codon_size = None, params['CODON_SIZE']
        self.min_path, self.max_arity, self.min_ramp = None, None, None

        self.ruleregex = r'(?P<rulename><\S+>)\s*::=\s*(?P<production>(?:(?=\#)\#[^\r\n]*|(?!<\S+>\s*::=).+?)+)'
        self.productionregex = r'(?=\#)(?:\#.*$)|(?!\#)\s*(?P<production>(?:[^\'\"\|\#]+|\'.*?\'|".*?")+)'
        self.productionpartsregex = r'\ *([\r\n]+)\ *|([^\'"<\r\n]+)|\'(.*?)\'|"(.*?)"|(?P<subrule><[^>|\s]+>)|([<]+)'

        self.recursion_cache = {}

        self.read_bnf_file(file_name)

        self.check_depths()

        self.check_recursion(self.start_rule["symbol"], [])

        self.set_arity()

        self.set_grammar_properties()

        self.check_permutations()

        if params['MIN_INIT_TREE_DEPTH']:
            self.min_ramp = params['MIN_INIT_TREE_DEPTH']

        elif hasattr(params['INITIALISATION'], "ramping"):
            self.get_min_ramp_depth()

        if params['REVERSE_MAPPING_TARGET'] or params['TARGET_SEED_FOLDER']:
            self.concat_NTs, self.climb_NTs = {}, {}

            self.find_concatenation_NTs()

    def read_bnf_file(self, file_name):
        """
        Reads a BNF grammar file and stores its production rules.

        :param file_name: A specified BNF grammar file.
        :return: Nothing.
        """

        with open(file_name, 'r') as bnf:
            content = bnf.read()

            for rule in finditer(self.ruleregex, content, DOTALL):

                if self.start_rule is None:
                    self.start_rule = {"symbol": rule.group('rulename'),
                                       "type": "NT"}

                self.non_terminals[rule.group('rulename')] = {
                    'id': rule.group('rulename'),
                    'min_steps': maxsize,
                    'expanded': False,
                    'recursive': True,
                    'b_factor': 0}

                tmp_productions = []

                for p in finditer(self.productionregex,
                                  rule.group('production'), MULTILINE):

                    if p.group('production') is None or p.group(
                            'production').isspace():
                        continue

                    tmp_production, terminalparts = [], None

                    # GE_RANGE:dataset_n_vars -> 0|1|...|n_vars-1 (same for dataset_n_is/os)
                    GE_RANGE_regex = r'GE_RANGE:(?P<range>\w*)'
                    m = match(GE_RANGE_regex, p.group('production'))
                    if m:
                        try:
                            if m.group('range') == "dataset_n_vars":
                                # n columns from dataset
                                n = params['FITNESS_FUNCTION'].n_vars
                            elif m.group('range') == "dataset_n_is":
                                # n input symbols
                                n = params['FITNESS_FUNCTION'].n_is
                            elif m.group('range') == "dataset_n_os":
                                # n output symbols
                                n = params['FITNESS_FUNCTION'].n_os
                            else:
                                n = int(m.group('range'))
                        except (ValueError, AttributeError):
                            raise ValueError("Bad use of GE_RANGE: "
                                             + m.group())

                        for i in range(n):
                            tmp_production, terminalparts = [], None
                            symbol = {
                                "symbol": str(i),
                                "type": "T",
                                "min_steps": 0,
                                "recursive": False}
                            tmp_production.append(symbol)
                            if str(i) not in self.terminals:
                                self.terminals[str(i)] = \
                                    [rule.group('rulename')]
                            elif rule.group('rulename') not in \
                                    self.terminals[str(i)]:
                                self.terminals[str(i)].append(
                                    rule.group('rulename'))
                            tmp_productions.append({"choice": tmp_production,
                                                    "recursive": False,
                                                    "NT_kids": False})
                        # stop here (later productions still processed)
                        continue

                    for sub_p in finditer(self.productionpartsregex,
                                          p.group('production').strip()):

                        if sub_p.group('subrule'):
                            if terminalparts is not None:
                                symbol = {"symbol": terminalparts,
                                          "type": "T",
                                          "min_steps": 0,
                                          "recursive": False}
                                tmp_production.append(symbol)
                                if terminalparts not in self.terminals:
                                    self.terminals[terminalparts] = \
                                        [rule.group('rulename')]
                                elif rule.group('rulename') not in \
                                        self.terminals[terminalparts]:
                                    self.terminals[terminalparts].append(
                                        rule.group('rulename'))
                                terminalparts = None

                            tmp_production.append(
                                {"symbol": sub_p.group('subrule'),
                                 "type": "NT"})

                        else:
                            # unescape special chars
                            if terminalparts is None:
                                terminalparts = ''
                            terminalparts += ''.join(
                                [part.encode().decode('unicode-escape') for
                                 part in sub_p.groups() if part])

                    if terminalparts is not None:
                        symbol = {"symbol": terminalparts,
                                  "type": "T",
                                  "min_steps": 0,
                                  "recursive": False}
                        tmp_production.append(symbol)
                        if terminalparts not in self.terminals:
                            self.terminals[terminalparts] = \
                                [rule.group('rulename')]
                        elif rule.group('rulename') not in \
                                self.terminals[terminalparts]:
                            self.terminals[terminalparts].append(
                                rule.group('rulename'))
                    tmp_productions.append({"choice": tmp_production,
                                            "recursive": False,
                                            "NT_kids": False})

                if not rule.group('rulename') in self.rules:
                    self.rules[rule.group('rulename')] = {
                        "choices": tmp_productions,
                        "no_choices": len(tmp_productions)}

                    if len(tmp_productions) == 1:
                        print("Warning: Grammar contains unit production "
                              "for production rule", rule.group('rulename'))
                        print("         Unit productions consume GE codons.")
                else:
                    raise ValueError("lhs should be unique",
                                     rule.group('rulename'))

    def check_depths(self):
        """
        Minimum distance from each NT to the nearest T.

        :return: Nothing.
        """

        counter, graph = 1, []

        for rule in sorted(self.rules.keys()):
            choices = self.rules[rule]['choices']

            self.non_terminals[rule]['b_factor'] = self.rules[rule][
                'no_choices']

            for choice in choices:
                graph.append([rule, choice['choice']])

        while graph:
            removeset = set()
            for edge in graph:
                if all([sy["type"] == "T" or
                        self.non_terminals[sy["symbol"]]['expanded']
                        for sy in edge[1]]):
                    removeset.add(edge[0])

            for s in removeset:
                self.non_terminals[s]['expanded'] = True
                self.non_terminals[s]['min_steps'] = counter

            graph = [e for e in graph if e[0] not in removeset]
            counter += 1

    def check_recursion(self, cur_symbol, seen):
        """
        Traverses the grammar recursively and sets the properties of each rule.

        :param cur_symbol: symbol to check.
        :param seen: Contains already checked symbols in the current traversal.
        :return: Boolean stating whether or not cur_symbol is recursive.
        """

        if cur_symbol not in self.non_terminals.keys():
            return False

        if cur_symbol in seen:
            return True

        seen.append(cur_symbol)

        choices = self.rules[cur_symbol]['choices']

        recursive = False
        for choice in choices:
            for sym in choice['choice']:
                if sym["type"] == "NT":
                    if sym["symbol"] in self.recursion_cache:
                        recursion_result = self.recursion_cache[sym["symbol"]]
                    else:
                        recursion_result = self.check_recursion(sym["symbol"], seen)
                        self.recursion_cache[sym["symbol"]] = recursion_result

                    recursive = recursive or recursion_result

        self.non_terminals[cur_symbol]['recursive'] = recursive
        seen.remove(cur_symbol)

        return recursive

    def set_arity(self):
        """
        Sets the minimum path and the maximum arity of the grammar.

        :return: Nothing
        """

        self.min_path = self.non_terminals[self.start_rule["symbol"]][
            'min_steps']

        self.max_arity = max(self.non_terminals[NT]['min_steps']
                             for NT in self.non_terminals)

        for rule in self.rules:
            for choice in self.rules[rule]['choices']:
                NT_kids = [i for i in choice['choice'] if i["type"] == "NT"]
                if NT_kids:
                    choice['NT_kids'] = True
                    for sym in NT_kids:
                        sym['min_steps'] = self.non_terminals[sym["symbol"]][
                            'min_steps']

        for rule in self.rules:
            for prod in self.rules[rule]['choices']:
                for sym in [i for i in prod['choice'] if i["type"] == "NT"]:
                    sym['recursive'] = self.non_terminals[sym["symbol"]][
                        'recursive']
                    if sym['recursive']:
                        prod['recursive'] = True

    def set_grammar_properties(self):
        """
        Goes through all non-terminals and finds the production choices with
        the minimum steps to terminals and with recursive steps.

        :return: Nothing
        """

        for nt in self.non_terminals:
            choices = self.rules[nt]['choices']

            for choice in choices:
                choice['max_path'] = max([item["min_steps"] for item in
                                          choice['choice']])

            min_path = min([choice['max_path'] for choice in choices])

            self.non_terminals[nt]['min_path'] = [choice for choice in
                                                  choices if choice[
                                                      'max_path'] == min_path]

            self.non_terminals[nt]['recursive'] = [choice for choice in
                                                   choices if choice[
                                                       'recursive']]

    def check_permutations(self):
        """
        Counts derivation tree combinations at a given depth.

        :return: Nothing.
        """

        ramps = params['PERMUTATION_RAMPS']

        perms_list = []
        if self.max_arity > self.min_path:
            for i in range(max((self.max_arity + 1 - self.min_path), ramps)):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])
        else:
            for i in range(ramps):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])

    def check_all_permutations(self, depth):
        """
        Counts derivation tree combinations up to a given depth.

        :param depth: Depth to count permutations for.
        :return: The permutations possible at the given depth.
        """

        if depth < self.min_path:
            # bug: asks for a tree smaller than any we can create
            s = "representation.grammar.Grammar.check_all_permutations\n" \
                "Error: cannot check permutations for tree smaller than the " \
                "minimum size."
            raise Exception(s)

        if depth in self.permutations.keys():
            return self.permutations[depth]

        else:
            pos, depth_per_symbol_trees, productions = 0, {}, []

            for NT in self.non_terminals:
                a = self.non_terminals[NT]

                for rule in self.rules[a['id']]['choices']:
                    if rule['NT_kids']:
                        productions.append(rule)

            start_symbols = self.rules[self.start_rule["symbol"]]['choices']

            for choice in productions:
                key = str([sym['symbol'] for sym in choice['choice']])

                depth_per_symbol_trees[key] = {}

            for i in range(2, depth + 1):

                for choice in productions:
                    sym_pos = 1

                    for j in choice['choice']:
                        symbol_arity_pos = 0

                        if j["type"] == "NT":
                            for child in self.rules[j["symbol"]]['choices']:

                                if len(child['choice']) == 1 and \
                                        child['choice'][0]["type"] == "T":
                                    symbol_arity_pos += 1

                                else:
                                    key = [sym['symbol'] for sym in
                                           child['choice']]
                                    if (i - 1) in depth_per_symbol_trees[
                                        str(key)].keys():
                                        symbol_arity_pos += \
                                            depth_per_symbol_trees[str(key)][
                                                i - 1]

                            sym_pos *= symbol_arity_pos

                    key = [sym['symbol'] for sym in choice['choice']]
                    depth_per_symbol_trees[str(key)][i] = sym_pos

            for sy in start_symbols:
                key = [sym['symbol'] for sym in sy['choice']]
                if str(key) in depth_per_symbol_trees:
                    pos += depth_per_symbol_trees[str(key)][depth] if depth in \
                                                                      depth_per_symbol_trees[
                                                                          str(
                                                                              key)] else 0
                else:
                    pos += 1

            self.permutations[depth] = pos

            return pos

    def get_min_ramp_depth(self):
        """
        Minimum depth at which ramping can start with unique solutions.

        :param self: An instance of the representation.grammar.grammar class.
        :return: The minimum depth at which unique solutions can be generated
        """

        max_tree_depth = params['MAX_INIT_TREE_DEPTH']
        size = params['POPULATION_SIZE']

        depths = range(self.min_path, max_tree_depth + 1)

        if size % 2:
            # population size is odd
            size += 1

        if size / 2 < len(depths):
            # pop too small to cover all ramping depths
            depths = depths[:int(size / 2)]

        unique_start = int(floor(size / len(depths)))
        ramp = None

        for i in sorted(self.permutations.keys()):
            if self.permutations[i] > unique_start:
                ramp = i
                break
        self.min_ramp = ramp

    def find_concatenation_NTs(self):
        """
        Finds the non-terminals that can combine derivation trees, i.e. those
        with a production choice holding several non-terminals.

        :return: Nothing.
        """

        for rule in sorted(self.rules.keys()):

            concat = [choice for choice in self.rules[rule]['choices'] if
                      choice['NT_kids']]

            if concat:
                for choice in concat:

                    symbols = [[sym['symbol'], sym['type']] for sym in
                               choice['choice']]

                    NTs = [sym['symbol'] for sym in choice['choice'] if
                           sym['type'] == "NT"]

                    for NT in NTs:
                        # concat_NTs: root -> nodes it can be reduced with
                        conc = [choice['choice'], rule, symbols]

                        if NT not in self.concat_NTs:
                            self.concat_NTs[NT] = [conc]
                        else:
                            if conc not in self.concat_NTs[NT]:
                                self.concat_NTs[NT].append(conc)

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)
