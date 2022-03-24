import numpy as np
from pymoo.core.repair import Repair


class NormalizeWeights(Repair):

    def _do(self, problem, pop, **kwargs):

        # the packing plan for the whole population (each row one individual)
        Z = pop.get("X")

        # now repair each indvidiual i
        for i in range(len(Z)):

            # the packing plan for i
            z = Z[i]
            #print(z)
            z = np.ceil(z)
            sum = np.sum(z)
            z = z * 100 / sum
            #print(z)
            Z[i] = z

        # set the design variables for the population
        pop.set("X", Z)
        return pop
