import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class Input:
    input_features_label = ['soil',	'slope', 'rainfall', 'lulc', 'lithology', 'lineament', 'geomorphology',	'drainage']
    input_features_columns = {8: 'soil', 9:'slope', 10:'rainfall', 11:'lulc', 12:'lithology',
                              13:'lineament', 14:'geomorphology',	15:'drainage', 16:'validation'}

    features_order = ['lithology', 'slope', 'drainage', 'geomorphology', 'lulc', 'lineament', 'rainfall', 'soil']
    ahp_weight = {'soil': 2,	'slope': 23, 'rainfall': 3, 'lulc': 7, 'lithology': 35, 'lineament': 4, 'geomorphology': 10,	'drainage': 15}
    sample_path = '../data/proportionate-sampling/'
    output_path = '../outputFiles/'
    ref_data_excel = '../data/excel/GW_potential_8Indicator_318pointsData.xls'


class EvaluatePop:
    def __init__(self):
        df = pd.read_excel(Input.ref_data_excel, sheet_name='AHP_weight',
                         header=None, skiprows=1, usecols=list(range(8, 17)))  # 0.3144654088
        df = df.rename(columns=Input.input_features_columns)
        self.sub_weight = df[Input.features_order].to_numpy()
        self.sub_weight = self.sub_weight / 100
        self.validation_class = df['validation'].to_numpy()

    def _evaluate_one(self, x):
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
        return [sum_diff, sum_misclassed, coef, p]

    def evaluate_pop(self, X):
        df_score = pd.DataFrame(columns=['sum-diff', 'sum-misclassed', 'spearman-r', 'p-value'],
                                index=list(range(X.shape[0])))
        for i in range(X.shape[0]):
            df_score.loc[i] = self._evaluate_one(X[i])
        return df_score

