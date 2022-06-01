from pymoo.core.problem import Problem
from global_info import Input, EvaluateJointPop
import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.factory import get_crossover, get_mutation, get_selection
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
from pymoo.util.function_loader import load_function




class WeightVector(Problem):
    def __init__(self, size, init_weight, **kwargs):
        self.size = size
        self.init_weight = init_weight
        super().__init__(n_var=size, n_obj=3, n_constr=size-1, xl=2, xu=50, **kwargs)
        self.nd_best = []
        self.F = None
        self.G = None

    def initialize(self, size: int, rand_percentage: float):
        # ahp_weight = np.array(self.init_weight)
        rand_init = Initialization(FloatRandomSampling(),  repair=NormalizeWeights())
        X1 = rand_init.do(self, int(size * 1))
        return X1.get("X")
        # X2 = np.full((size-int(size * rand_percentage), 8), ahp_weight)
        # X2 = Population.new(X=X2)
        # mutation = get_mutation("int_pm", eta=30, prob=1 / 8)
        # X2_ = mutation.do(self, X2)
        # X2_ = NormalizeWeights().do(self, X2_)
        # X = Population.merge(X1, X2_)
        # #print(X)
        # return X

    def evaluate_pop(self, pop_mat, tests_per_indiv, pop_i): #todo: use mode from histogram to estimate objectives
        objectives = np.zeros((pop_mat.shape[0], self.n_obj))
        constraints = np.zeros((pop_mat.shape[0], self.n_var - 1))
        for indiv_i in range(pop_mat.shape[0]):
            objectives[indiv_i] = np.median(tests_per_indiv[indiv_i], axis=0) #todo: try median
            if pop_i != 0:
                for i in range(1, self.n_var):
                    constraints[indiv_i, i-1] = pop_mat[indiv_i, i] - pop_mat[indiv_i, i-1] + 1

        pop = Population.new(X=pop_mat)
        self.F = objectives
        self.G = constraints
        # pop.set('F', objectives)
        # pop.set('G', constraints)
        # set_cv(pop)
        #pop.set('evaluated', True)
        return pop



    def _evaluate(self, x, f, *args, **kwargs):
        if (x.shape[0] != self.F.shape[0]):
            print("---------ERROR-----------")
        f['F'] = self.F #f_g[:, 0:3]
        f['G'] = self.G #f_g[:, 3:10]
        #pass

        #print(len(dcs))


class CooperativeCoevolution:
    def __init__(self, pop_size: int, tournament_size: int, max_iter: int, seed: int):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.max_iter = max_iter
        #self.outer_problem = WeightVector(8, Input.ahp_outer_weight)
        self.problems = []
        self.populations = []
        self.nd_sort = load_function('fast_non_dominated_sort')
        self.nd_samples = []
        self.seed = seed
        np.random.seed(seed)


    def evaluate_sample(self, X: list):
        for x in X:
            if np.isnan(x).any():
                return np.inf, np.inf, np.inf

        x = X[0] #outer weights
        inner_weight_map_dic = {}
        df_table = Input.sub_weight.copy()

        for i in range(1, len(X)):
            inner_weight_map_dic[i] = dict(zip(Input.ahp_weights[i], X[i])) #{ k:v for  }
            df_table[Input.features_order[i-1]].replace(inner_weight_map_dic[i], inplace=True)

        dfq = pd.DataFrame(index=list(range(len(df_table))))
        dfq['sum'] = np.inner(x, df_table.to_numpy())
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        q_class = dfq['class'].to_numpy()
        mis_classed = q_class != Input.validation_class
        diff = q_class - Input.validation_class
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)
        coef, p = spearmanr(Input.validation_class, q_class)
        return sum_diff, sum_misclassed, -coef

    def evaluate_joint(self):
        indiv_in_sample = {}
        for pop_i in range(len(self.populations)):
            indiv_in_sample[pop_i] = []
            for indiv_i in range(self.populations[pop_i].shape[0]):
                indiv_in_sample[pop_i].append([])#indiv_in_sample[(pop_i, indiv_i)] = []
        sample_count = np.zeros((len(self.problems), self.pop_size), dtype='int')
        joint_sample = []
        for pop_i in range(len(self.populations)):
            for indiv_i in range(self.populations[pop_i].shape[0]):
                m = sample_count[pop_i, indiv_i]
                for w in range(m, self.tournament_size):
                    X = []
                    for l in range(len(self.populations)):
                        selected_i = indiv_i
                        if l != pop_i:
                            selected_i = np.random.randint(self.pop_size)

                        sample_count[l][selected_i] += 1
                        X.append(self.populations[l][selected_i])
                        indiv_in_sample[l][selected_i].append(len(joint_sample))#indiv_in_sample[(l, selected_i)].append(len(joint_sample))
                    joint_sample.append(X)

        return joint_sample, indiv_in_sample

    def generate_offspring(self, problem: WeightVector, pop_mat, tests_per_indiv, pop_i):
        pop = problem.evaluate_pop(pop_mat, tests_per_indiv, pop_i)
        # pop = Evaluator().eval(problem, pop)
        # mating = Mating(crossover=get_crossover("real_sbx", prob=0.8, eta=40),
        #                 mutation=get_mutation("real_pm", eta=45, prob=1 / problem.n_var),
        #                 selection=get_selection('random'),
        #                 repair=NormalizeWeights())
        # offspring = mating.do(problem, pop, self.pop_size)
        # # offspring = NormalizeWeights().do(offspring)
        # return offspring.get("X")
        algorithm = NSGA2(pop_size=self.pop_size,
                          # n_offsprings=100,
                          sampling=pop,
                          crossover=get_crossover("real_sbx", prob=0.8, eta=40),
                          #crossover=get_crossover("int_sbx", prob=0.8, eta=30),
                          mutation=get_mutation("real_pm", eta=45, prob=1 / problem.n_var),
                          #mutation=get_mutation("int_pm", eta=30, prob=1 / 8),
                          repair=NormalizeWeights(),
                          eliminate_duplicates=False)
        res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 2),
                   seed=self.seed,
                   verbose=True,
                   save_history=True)

        pop = res.history[1].off
        #print("Offspring size: " + str(pop.get("X").shape[0]))
        return pop.get("X")

    def run(self):
        for i in range(Input.num_feature+1):
            self.problems.append(WeightVector(len(Input.ahp_weights[i]), Input.ahp_weights[i]))
            self.populations.append(self.problems[i].initialize(self.pop_size, rand_percentage=0.8))
        for iter_i in range(self.max_iter):
            joint_sample, indiv_in_sample = self.evaluate_joint()
            # objectives = []
            # for X in joint_sample:
            #     objectives.append(list(self.evaluate_sample(X)))
            # print(np.array(objectives))

            with multiprocessing.Pool(8) as p:
                dcs = p.map(self.evaluate_sample, joint_sample)

            joint_sample_objectives = np.array(dcs)
            nd_indices = self.nd_sort(joint_sample_objectives)[0]
            new_nd_solutions = [joint_sample[i] for i in nd_indices]
            self.nd_samples.extend(new_nd_solutions)

            for pop_i in range(len(self.populations)):
                test_per_indiv = []
                for indiv_i in range(self.populations[pop_i].shape[0]):
                    test_per_indiv.append(joint_sample_objectives[indiv_in_sample[pop_i][indiv_i]])

                print( "Gen: " + str(iter_i) +",Pop: " + str(pop_i))
                self.populations[pop_i] = self.generate_offspring(self.problems[pop_i], self.populations[pop_i], test_per_indiv, pop_i)

        return joint_sample_objectives, self.nd_samples
            #print(self.nd_samples)




if __name__ == '__main__':

    cov = CooperativeCoevolution(pop_size=20, tournament_size=20,
                                 max_iter=20, seed=0)
    result, jointPop = cov.run()
    evalBase = EvaluateJointPop()
    d = evalBase.evaluate_pop(jointPop)
    min_obj = d[:,[0,1,2]]
    min_obj[:,2] = -min_obj[:,2]
    nd_indices = cov.nd_sort(min_obj)[0]
    d = d[nd_indices]
    out_df = pd.DataFrame()
    out_df['FID'] = evalBase.FID
    out_df['validation'] = (evalBase.validation_class >= 3) * 1
    #writer = pd.ExcelWriter('../var.xls', engine = 'xlwt')
    var_df = pd.DataFrame()
    for nd_i in nd_indices:
        out_df['all-relaxed-'+ str(nd_i) + '-score'] = evalBase.get_prediction_score(jointPop[nd_i])
        outer, inner = evalBase.format_weights_as_table(jointPop[nd_i])
        var_df['all-relaxed-' + str(nd_i) + '-weight'] = outer
        var_df['all-relaxed-' + str(nd_i) + '-rank'] = inner
        #writer.save()
    out_df = out_df.sort_values(by='FID')
    out_df.to_csv('../all-relaxed-score.csv', index=False)
    var_df.to_csv('../all-relaxed-var.csv', index=False)

    get_visualization("pcp").add(d).show()













