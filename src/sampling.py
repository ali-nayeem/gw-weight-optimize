import pandas as pd
import random
random.seed(0)

no_of_instance = 100
output_dir = "../../Data/proportionate-sampling/"


for iteration in range(no_of_instance):
    seed_i = random.randint(0, 100000)
    df = pd.read_excel('../../Data/excel/GW_potential_8Indicator_318pointsData.xls', sheet_name='AHP_weight')#0.3144654088

    sample = df.groupby('validation point categorical data', group_keys=False).apply(lambda x: x.sample(frac=0.3145, random_state=seed_i, replace=False))
    sample.to_csv(output_dir + 'sample'+str(iteration)+".csv", index=False)

    #print(df2)
    #validation point categorical data
