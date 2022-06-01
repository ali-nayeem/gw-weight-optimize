import pandas as pd

df = pd.read_excel('../../Data/excel/pa_counties_pop_change_1990-2000.xls', names=['county', 'pop'])
df['class'] = pd.qcut(df['pop'], 5, labels=[5, 4, 3, 2, 1])
df['class'] = pd.qcut(df['pop'], 5, labels=[5, 4, 3, 2, 1])

df2=df.sort_values(by='pop', ascending=False)
df2.to_csv('../quantile.csv')
