from pymoo.core.problem import Problem
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
import multiprocessing
import matplotlib.pyplot as plt
from pymoo.visualization.pcp import PCP
from pymoo.util.function_loader import load_function



class GW_Parallel(Problem):
    def __init__(self, sample_id: int, sample_path: str, feature_names: list, init_weight, cpu_cores=1):
        self.sample_id = sample_id
        self.sample_path = sample_path
        self.init_weight = init_weight
        df = pd.read_csv(sample_path + 'sample' + str(sample_id) + '.csv', header=None, skiprows=1,
                         usecols=[0].extend(list(range(8, 17))))  # 0.3144654088
        df = df.rename(columns=Input.input_features_columns)
        self.FID = df['FID']
        self.sub_weight = df[Input.features_order].to_numpy()
        #self.sub_weight = self.sub_weight# / 100
        self.validation_class = df['validation'].to_numpy()
        self.cpu_cores = cpu_cores

        super().__init__(n_var=8, n_obj=3, n_constr=7, xl=2, xu=50)


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


    def compute_short(self, x):
        if np.isnan(x).any():
            return np.inf, np.inf, np.inf, 2, 2, 2, 2, 2, 2, 2
        dfq = pd.DataFrame(index=list(range(self.sub_weight.shape[0])))
        dfq['sum'] = np.inner(x, self.sub_weight)
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        q_class = dfq['class'].to_numpy()
        mis_classed = q_class != self.validation_class
        diff = q_class - self.validation_class
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)
        coef, p = spearmanr(self.validation_class, q_class)


        return sum_diff, sum_misclassed, -coef, \
               x[7] - x[6] + 1, x[6] - x[5] + 1, x[5] - x[4] + 1, \
               x[4] - x[3] + 1, x[3] - x[2] + 1, x[2] - x[1] + 1, x[1] - x[0] + 1

    def _evaluate(self, x, f, *args, **kwargs):
        with multiprocessing.Pool(self.cpu_cores) as p:
            dcs = p.map(self.compute_short, x)
        # list = [x for (x, y) in dcs]
        # print(str(np.min(list))) #+ " , " + str(np.min(array[:, 1])))
        f_g = np.array(dcs)
        f['F'] = f_g[:, 0:3]
        f['G'] = f_g[:, 3:10]

        #print(len(dcs))


if __name__ == '__main__':
    prob = GW_Parallel(0, Input.sample_path, Input.input_features_label, Input.ahp_weight, cpu_cores=8)
    x = []
    for f in Input.features_order:
        x.append(Input.ahp_weight[f])
    pop = prob.initialize(20, np.array(x), rand_percentage=0.8)
    X = pop.get("X")
    X2 = np.full((100, 8), np.array(x))


    algorithm = NSGA2(pop_size=100,
                      # n_offsprings=100,
                      #sampling=pop,
                      crossover=get_crossover("real_sbx", prob=0.8, eta=40),
                      #crossover=get_crossover("int_sbx", prob=0.8, eta=30),
                      mutation=get_mutation("real_pm", eta=45, prob=1 / 8),
                      #mutation=get_mutation("int_pm", eta=30, prob=1 / 8),
                      repair=NormalizeWeights(),
                      eliminate_duplicates=True)
    res = minimize(prob,
                   algorithm,
                   termination=('n_gen', 300),
                   seed=0,
                   verbose=True,
                   save_history=False)

    pop = res.pop
    eval_validation = EvaluatePop(prob.FID)
    nd_sort = load_function('fast_non_dominated_sort')
    #print(pop.get("X"))
    df_score = eval_validation.evaluate_pop(pop.get("X"))
    #df_score.to_csv("../score2.csv")
    #np.savetxt("../weights2.csv", pop.get("X"), delimiter=",", fmt='%i', header=",".join(Input.features_order))
    d = df_score#.to_numpy()
    min_obj = d[:,[0,1,2]]
    min_obj[:,2] = -min_obj[:,2]
    nd_indices = nd_sort(min_obj)[0]
    d = d[nd_indices]
    get_visualization("pcp").add(d).show()
    out_df = pd.DataFrame()
    out_df['FID'] = eval_validation.FID
    out_df['validation'] = eval_validation.validation_class
    var_df = pd.DataFrame()
    for nd_i in nd_indices:
        out_df['outer-'+ str(nd_i)] = eval_validation.get_prediction_score(pop.get("X")[nd_i])
        outer, inner = eval_validation.format_weights_as_table(pop.get("X")[nd_i])
        var_df['outer-' + str(nd_i) + '-weight'] = outer
        var_df['outer-' + str(nd_i) + '-rank'] = inner
    out_df.to_csv('../ga_only_outer_score.csv', index=False)
    var_df.to_csv('../ga_only_outer_var.csv', index=False)
    print('manual thread:', res.exec_time)


# plot = Scatter()
    # plot.add(res.F, color="red")
    #plot.show()
