import os
import glob as glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_prefix = "config_40k_visium"
dataset_name = dataset_prefix.rsplit('config_')[1]
compare_CoDi_metrics = False
output_suffix = ''

cur_path = os.path.dirname(os.path.realpath(__file__))

if compare_CoDi_metrics:
    files = glob.glob(os.path.join(cur_path, '../test/') + f'{dataset_prefix}_benchmark_CoDi_*.csv')
    output_path = f'comparison_CoDi_{dataset_name}{output_suffix}'
else:
    files = glob.glob(os.path.join(cur_path, '../test/') + f'{dataset_prefix}_benchmark_*.csv')
    output_path = f'comparison_{dataset_name}{output_suffix}'
dfs = {}
for f in files:
    if not compare_CoDi_metrics:
        if ('CoDi' in f) and (not 'KLD' in f):
            continue
    if compare_CoDi_metrics:
        name = f.split('benchmark_CoDi_')[-1].replace('.csv', '')
    else:
        name = f.split('benchmark_')[-1].replace('.csv', '')
        if name == "CoDi_KLD":
            name = "CoDi"
    dfs[name] = pd.read_csv(f)
    sns.scatterplot(data=dfs[name], x='Subsample', y='Accuracy')
plt.legend(labels=dfs.keys())
plt.xticks(dfs[name]['Subsample'].values, size=7)
plt.title(f'Cell type detection accuracy for {dataset_name}')
plt.box(False)
plt.savefig(output_path + '.png', dpi=150)