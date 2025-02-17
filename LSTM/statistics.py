import pandas as pd

df = pd.read_csv('experiment_1.csv')

stats = df.groupby('loaded model')['Mean phase error'].agg(['max', 'min', 'mean'])

result = stats.stack().reset_index()
result.columns = ['loaded_model', 'statistics', 'value']

result.to_csv('experiment_1_stats.csv', index=False)

print(result)
