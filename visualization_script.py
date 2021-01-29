import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DF = pd.read_csv(
    'results\\tables\\results_performance_experiment.csv'
)

plt.figure(figsize=(10, 10))
sns.catplot(
    x='model',
    y='metric_value',
    col='metric_name',
    kind='box',
    data=RESULTS_DF,
    sharey=False
)

plt.savefig('results\\figures\\model_comparison.pdf')
