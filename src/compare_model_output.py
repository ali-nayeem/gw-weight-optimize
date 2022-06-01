import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pymoo.factory import get_visualization
from pymoo.visualization.pcp import PCP

col1 = ['all-relaxed-55']
col2 = ['all-domain-21', 'all-domian-38', 'all-domain-41']
col3 = ['outer-nd-12', 'outer-nd-17', 'outer-nd-44', 'outer-nd-59']
col4 = ['ahp', 'MifAhp', 'fuzzy_ahp', 'AhpAnp', 'MifAnp', 'fuzzyAnp']
col4_color = {'ahp': 'blue', 'MifAhp': 'green', 'fuzzy_ahp': 'cyan', 'AhpAnp': 'grey', 'MifAnp': 'magenta', 'fuzzyAnp':'black'}



def evaluate_model_output(q_class, validation_class):
    #q_class = self.get_predicted_labels(x)
    mis_classed = q_class != validation_class
    diff = q_class - validation_class
    sum_diff = np.sum(np.abs(diff))
    sum_misclassed = np.sum(mis_classed)
    coef, p = spearmanr(validation_class, q_class)
    #print(np.sum(diff))
    return np.array([sum_diff, sum_misclassed, coef])

plot = PCP(title=("Total row = 219", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["abs diff", "misclassified", "spearman-corr"] #, "p-value"]
           )
plot.set_axis_style(color="grey", alpha=1)

df_all = pd.read_csv('../ga_all_relax.csv')
# col = df_all.columns.to_list()
# col.pop(0)
# col.pop(0)

x = evaluate_model_output(df_all['all-relaxed-55'].to_numpy(), df_all['validation'].to_numpy())
plot.add(x, linewidth=2, color="red", linestyle='dashed', label="all-relaxed-55")

# for col_name in col2:
#     x = evaluate_model_output(df_all[col_name].to_numpy(), df_all['validation'].to_numpy())
#     plot.add(x, linewidth=2, color="blue", linestyle='dashdot', label=col_name)

for col_name in col4:
    x = evaluate_model_output(df_all[col_name].to_numpy(), df_all['validation'].to_numpy())
    plot.add(x, linewidth=2, color=col4_color[col_name], linestyle='solid', label=col_name)

xx = np.zeros((len(col3), 3))
for i in range(len(col3)):
    xx[i] = evaluate_model_output(df_all[col3[i]].to_numpy(), df_all['validation'].to_numpy())

plot.add(xx, linewidth=2, color="green", linestyle='dashed')


xx = np.zeros((len(col2), 3))
for i in range(len(col2)):
    xx[i] = evaluate_model_output(df_all[col2[i]].to_numpy(), df_all['validation'].to_numpy())

plot.add(xx, linewidth=2, color="blue", linestyle='dashdot')

plot.show()



