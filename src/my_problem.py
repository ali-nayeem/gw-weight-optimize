from pymoo.core.problem import Problem, ElementwiseProblem
from global_info import Input
import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from my_repair import NormalizeWeights
from scipy.stats import spearmanr
from global_info import EvaluatePop
from pymoo.factory import get_visualization
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population


class GW(ElementwiseProblem):
    def __init__(self, sample_id: int, sample_path: str, feature_names: list, init_weight):
        self.sample_id = sample_id
        self.sample_path = sample_path
        self.init_weight = init_weight
        df = pd.read_csv(sample_path + 'sample' + str(sample_id) + '.csv', header=None, skiprows=1,
                         usecols=list(range(8, 17)))  # 0.3144654088
        df = df.rename(columns=Input.input_features_columns)
        self.sub_weight = df[Input.features_order].to_numpy()
        self.sub_weight = self.sub_weight / 100
        self.validation_class = df['validation'].to_numpy()

        super().__init__(n_var=8, n_obj=3, n_constr=7, xl=2, xu=50)

    def _evaluate_sample(self, id: int):
        dfq = pd.DataFrame(index=list(range(self.sub_weight[id].shape[0])))
        dfq['sum'] = np.inner(x, self.sub_weight[id])
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[5, 4, 3, 2, 1])
        q_class = dfq['class'].to_numpy()
        mis_classed = q_class != self.validation_class[id]
        diff = q_class - self.validation_class[id]
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)

    def initialize(self, size: int, ahp_weight, rand_percentage: float):
        rand_init = Initialization(FloatRandomSampling(),
                                     repair=NormalizeWeights())
        X1 = rand_init.do(self, int(size * rand_percentage))
        X2 = np.full((size-int(size * rand_percentage), 8), ahp_weight)
        X2 = Population.new(X=X2)
        mutation = get_mutation("int_pm", eta=30, prob=1 / 8)
        X2_ = mutation.do(self, X2)
        X2_ = NormalizeWeights().do(self, X2_)
        X = Population.merge(X1, X2_)
        #print(X)
        return X

    def _evaluate(self, x, out, *args, **kwargs):
        dfq = pd.DataFrame(index=list(range(self.sub_weight.shape[0])))
        dfq['sum'] = np.inner(x, self.sub_weight)
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[5, 4, 3, 2, 1])
        q_class = dfq['class'].to_numpy()
        mis_classed = q_class != self.validation_class
        diff = q_class - self.validation_class
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)
        coef, p = spearmanr(self.validation_class, q_class)
        #print(np.sum(diff))

        out["F"] = np.column_stack([sum_diff, sum_misclassed, -coef])
        out["G"] = np.column_stack([x[7] - x[6] + 1, x[6] - x[5] + 1, x[5] - x[4] + 1, x[4] - x[3] + 1,
                                    x[3] - x[2] + 1, x[2] - x[1] + 1, x[1] - x[0] + 1])


if __name__ == '__main__':
    eval = EvaluatePop()
    prob = GW(0, Input.sample_path, Input.input_features_label, Input.ahp_weight)
    x = []
    for f in Input.features_order:
        x.append(Input.ahp_weight[f])
    prob.evaluate(np.array(x), None)
    pop = prob.initialize(100, np.array(x), 0.5)
    X =  pop.get("X")
    X2 = np.full((100, 8), np.array(x))


    algorithm = NSGA2(pop_size=100,
                      # n_offsprings=100,
                      sampling=pop,
                      #crossover=get_crossover("real_sbx", prob=0.8, eta=40),
                      crossover=get_crossover("int_sbx", prob=0.8, eta=30),
                      #mutation=get_mutation("real_pm", eta=45, prob=1 / 8),
                      mutation=get_mutation("int_pm", eta=30, prob=1 / 8),
                      repair=NormalizeWeights(),
                      eliminate_duplicates=True)
    res = minimize(prob,
                   algorithm,
                   termination=('n_gen', 200),
                   seed=1,
                   verbose=True,
                   save_history=False)

    pop = res.pop

    print(pop.get("X"))
    df_score = eval.evaluate_pop(pop.get("X"))
    #df_score.to_csv("../score2.csv")
    #np.savetxt("../weights2.csv", pop.get("X"), delimiter=",", fmt='%i', header=",".join(Input.features_order))
    get_visualization("pcp").add(df_score.to_numpy()).show()
    print('Single:', res.exec_time)


# plot = Scatter()
    # plot.add(res.F, color="red")
    #plot.show()
