
import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SBX


class SBX_Cross(Crossover):

    def __init__(self, prob, **kwargs):
        self.crossover = SBX(prob)
        super().__init__(2, 2, prob, **kwargs)

    def _do(self, problem, X, **kwargs):

        # X: (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        problem.xu = np.array([1.0 for _ in range(problem.n_var)])
        problem.xl = np.array([0.0 for _ in range(problem.n_var)])

        parents = [[Individual(X=X[0,i]), Individual(X=X[1,i])] for i in range(X.shape[1])]

        off = self.crossover.do(problem, parents)
        Xp = off.get("X")
        Xp = Xp.reshape(2, n_matings, n_var)

        # out: (n_parents, n_matings, n_var)
        return Xp


class PMX_Cross(Crossover):

    def __init__(self, prob, **kwargs):
        super().__init__(2, 2, prob, **kwargs)

    def is_permutation(self, arr):

        n = len(arr)

        arr_set = set(arr)

        return len(arr_set) == n and min(arr) == 0 and max(arr) == n-1

    def _do(self, problem, X, **kwargs):

        # X: (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        offspring = np.full((2, n_matings, n_var), -1)

        for i in range(n_matings):

            parent1, parent2 = X[0, i], X[1, i]

            cx_point1, cx_point2 = sorted(np.random.choice(n_var, 2, replace=False))

            offspring[0, i, cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
            offspring[1, i, cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

            def map_genes(o, p, cx1, cx2):
                for j in range(n_var):
                    if j < cx1 or j >= cx2:
                        gene = p[j]
                        while gene in o[cx1:cx2]:
                            gene = p[np.where(o==gene)[0][0]]
                        o[j] = gene

            map_genes(offspring[0, i], parent2, cx_point1, cx_point2)
            map_genes(offspring[1, i], parent1, cx_point1, cx_point2)

            if self.is_permutation(offspring[0, i]) and self.is_permutation(offspring[1, i]):
                continue

            else:
                raise(ValueError)

        return offspring


class CX_Cross(Crossover):

    def __init__(self, prob, **kwargs):
        super().__init__(2, 2, prob, **kwargs)

    def is_permutation(self, arr):
        n = len(arr)
        arr_set = set(arr)
        return len(arr_set) == n and min(arr) == 0 and max(arr) == n - 1

    def _do(self, problem, X, **kwargs):
        # X: (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        offspring = np.full((2, n_matings, n_var), -1)

        for i in range(n_matings):
            parent1, parent2 = X[0, i], X[1, i]

            size = len(parent1)

            indx = np.random.randint(size)
            start_indx = indx

            while True:
                offspring[0, i, indx] = parent1[indx]
                offspring[1, i, indx] = parent2[indx]

                val = parent2[indx]
                indx = np.where(parent1 == val)[0][0]

                if indx == start_indx:
                    break

            for j in range(size):
                if offspring[0, i, j] == -1:
                    offspring[0, i, j] = parent2[j]
                if offspring[1, i, j] == -1:
                    offspring[1, i, j] = parent1[j]

        if not (self.is_permutation(offspring[0, i]) and self.is_permutation(offspring[1, i])):
            raise ValueError("Offspring are not valid permutations!")
        
        return offspring
