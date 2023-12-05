import glob as glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_prefix = "config_40k_visium"
dataset_name = dataset_prefix.rsplit('config_')[1]
compare_CoDi_metrics = True

if compare_CoDi_metrics:
    files = glob.glob('../test/' + f'{dataset_prefix}_benchmark_CoDi_*.csv')
    output_path = f'comparison_CoDi_{dataset_name}'
else:
    files = glob.glob('../test/' + f'{dataset_prefix}_benchmark_*.csv')
    output_path = f'comparison_{dataset_name}'
dfs = {}
for f in files:
    if not compare_CoDi_metrics:
        if ('CoDi' in f) and (not 'KLD' in f):
            continue
    name = f.split('_')[-1].replace('.csv', '')
    dfs[name] = pd.read_csv(f)
    sns.scatterplot(data=dfs[name], x='Subsample', y='Accuracy')
plt.legend(labels=dfs.keys())
plt.xticks(dfs[name]['Subsample'].values, size=7)
plt.title(f'Cell type detection using different distances on {dataset_name}')
plt.savefig(output_path + '.png', dpi=150)