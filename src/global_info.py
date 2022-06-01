import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import multiprocessing


class Input:
    input_features_label = ['soil',	'slope', 'rainfall', 'lulc', 'lithology', 'lineament', 'geomorphology',	'drainage']
    input_features_columns = {0: 'FID', 8: 'soil', 9:'slope', 10:'rainfall', 11:'lulc', 12:'lithology',
                              13:'lineament', 14:'geomorphology',	15:'drainage', 16:'validation'}

    features_order = ['lithology', 'slope', 'drainage', 'geomorphology', 'lulc', 'lineament', 'rainfall', 'soil']
    ahp_weight = {'soil': 2,	'slope': 23, 'rainfall': 3, 'lulc': 7, 'lithology': 35, 'lineament': 4, 'geomorphology': 10,	'drainage': 15}
    ahp_outer_weight = [35, 23, 15, 10, 7, 5, 3, 2]
    ahp_weights = [[35, 23, 15, 10, 7, 5, 3, 2],
                        [34, 28, 18, 8, 6, 4, 3],
                        [41, 27, 15, 9, 5, 3],
                        [46, 26, 15, 9, 5],
                        [44, 26, 16, 10, 4],
                        [53, 27, 14, 6],
                        [42, 26, 16, 10, 6],
                        [42, 25, 14, 8, 5, 3, 2], #rainfall: special treatment
                        [53, 30, 11, 6]]
    num_feature = 8
    sample_id = 0
    sample_path = '../data/proportionate-sampling/'
    output_path = '../outputFiles/'
    ref_data_excel = '../data/excel/GW_potential_8Indicator_318pointsData.xls'
    sub_weight = None
    validation_class = None

df = pd.read_csv(Input.sample_path + 'sample' + str(Input.sample_id) + '.csv', header=None, skiprows=1,
                 usecols=[0].extend(list(range(8, 17))))  # 0.3144654088
df = df.rename(columns=Input.input_features_columns)
Input.FID = df['FID'].to_list()
Input.sub_weight = df[Input.features_order]
#sub_weight = sub_weight# / 100
Input.validation_class = df['validation'].to_numpy()


class EvaluatePop:
    def __init__(self, FID):
        df = pd.read_excel(Input.ref_data_excel, sheet_name='AHP_weight',
                         header=None, skiprows=1, usecols=[0].extend(list(range(8, 17))))  # 0.3144654088
        df = df.rename(columns=Input.input_features_columns)
        df = df[~df.FID.isin(FID)]
        self.FID = df['FID']
        self.sub_weight = df[Input.features_order].to_numpy()
        #self.sub_weight = self.sub_weight / 100
        self.validation_class = df['validation'].to_numpy()

    def get_prediction_score(self, x):
        dfq = pd.DataFrame(index=list(range(self.sub_weight.shape[0])))
        dfq['sum'] = np.inner(x, self.sub_weight)
        #dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        return dfq['sum'].to_numpy()

    def get_predicted_labels(self, x):
        dfq = pd.DataFrame(index=list(range(self.sub_weight.shape[0])))
        dfq['sum'] = np.inner(x, self.sub_weight)
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        return dfq['class'].to_numpy()

    def format_weights_as_table(self, X):
        x = X
        outer = []
        inner = []
        for f_i in range(1, len(Input.ahp_weights)):
            outer.extend([x[f_i-1]] * len(Input.ahp_weights[f_i]))
            inner.extend(Input.ahp_weights[f_i])
        #df = pd.DataFrame()
        #df['Weight'] = outer
        #df['Rank'] = inner
        #df = df.set_index(["Weight", "Rank"])
        return outer, inner#df

    def _evaluate_one(self, x):
        q_class = self.get_predicted_labels(x)
        mis_classed = q_class != self.validation_class
        diff = q_class - self.validation_class
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)
        coef, p = spearmanr(self.validation_class, q_class)
        #print(np.sum(diff))
        return sum_diff, sum_misclassed, coef, p

    def evaluate_pop(self, X):
        # df_score = pd.DataFrame(columns=['sum-diff', 'sum-misclassed', 'spearman-r', 'p-value'],
        #                         index=list(range(X.shape[0])))
        # for i in range(X.shape[0]):
        #     df_score.loc[i] = self._evaluate_one(X[i])
        # return df_score
        with multiprocessing.Pool(8) as p:
            dcs = p.map(self._evaluate_one, X)
        return np.array(dcs)


class EvaluateJointPop:
    def __init__(self):
        df = pd.read_excel(Input.ref_data_excel, sheet_name='AHP_weight',
                           header=None, skiprows=1, usecols=[0].extend(list(range(8, 17))))  # 0.3144654088
        df = df.rename(columns=Input.input_features_columns)
        df = df[~df.FID.isin(Input.FID)]
        self.FID = df['FID']
        self.sub_weight = df[Input.features_order]#.to_numpy()
       # self.sub_weight = self.sub_weight #/ 100
        self.validation_class = df['validation'].to_numpy()

    def get_prediction_score(self, X):
        x = X[0] #outer weights
        inner_weight_map_dic = {}
        df_table = self.sub_weight.copy()

        for i in range(1, len(X)):
            inner_weight_map_dic[i] = dict(zip(Input.ahp_weights[i], X[i])) #{ k:v for  }
            df_table[Input.features_order[i-1]].replace(inner_weight_map_dic[i], inplace=True)

        dfq = pd.DataFrame(index=list(range(len(df_table))))
        dfq['sum'] = np.inner(x, df_table.to_numpy())
        #dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        return dfq['sum'].to_numpy()

    def get_predicted_labels(self, X):
        x = X[0] #outer weights
        inner_weight_map_dic = {}
        df_table = self.sub_weight.copy()

        for i in range(1, len(X)):
            inner_weight_map_dic[i] = dict(zip(Input.ahp_weights[i], X[i])) #{ k:v for  }
            df_table[Input.features_order[i-1]].replace(inner_weight_map_dic[i], inplace=True)

        dfq = pd.DataFrame(index=list(range(len(df_table))))
        dfq['sum'] = np.inner(x, df_table.to_numpy())
        dfq['class'] = pd.qcut(dfq['sum'], 5, labels=[1, 2, 3, 4, 5])
        return dfq['class'].to_numpy()

    def evaluate_sample(self, X: list):
        q_class = self.get_predicted_labels(X) #dfq['class'].to_numpy()
        mis_classed = q_class != self.validation_class
        diff = q_class - self.validation_class
        sum_diff = np.sum(np.abs(diff))
        sum_misclassed = np.sum(mis_classed)
        coef, p = spearmanr(self.validation_class, q_class)
        return sum_diff, sum_misclassed, coef, p

    def evaluate_pop(self, joint_sample):

        with multiprocessing.Pool(8) as p:
            dcs = p.map(self.evaluate_sample, joint_sample)
        return np.array(dcs)

    def format_weights_as_table(self, X):
        x = X[0]
        outer = []
        inner = []
        for f_i in range(1, 9):
            outer.extend([x[f_i-1]] * len(X[f_i]))
            inner.extend(list(X[f_i]))
        #df = pd.DataFrame()
        #df['Weight'] = outer
        #df['Rank'] = inner
        #df = df.set_index(["Weight", "Rank"])
        return outer, inner#df




if __name__ == "__main__":
    df = pd.read_excel('../../GW_PotentialAllFinalWeights318PointData.xls')
    df = df[~df.FID.isin(Input.FID)]
    df.to_csv('GW_OnlyValidation.cs  v', index=False)