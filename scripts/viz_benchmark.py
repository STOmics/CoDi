import os
import glob as glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_prefix = "config_40k_visium"
dataset_prefix = "config_breast"
# dataset_prefix = "config_breast_distances"
# dataset_prefix = "config_40k_visium_distances"
title_dataset_name = 'Visium Mouse brain'
title_dataset_name = 'snRNA breast'
dataset_name = dataset_prefix.rsplit('config_')[1]
compare_CoDi_metrics = False
output_suffix = ''

cur_path = os.path.dirname(os.path.realpath(__file__))
if compare_CoDi_metrics:
    files = glob.glob(os.path.join('../test/') + f'{dataset_prefix}_benchmark_CoDi_*.csv')
    output_path = f'comparison_CoDi_{dataset_name}{output_suffix}'
else:
    files = glob.glob(os.path.join(cur_path, '../test/') + f'{dataset_prefix}_benchmark_*.csv')
    output_path = f'comparison_{dataset_name}{output_suffix}'
dfs = {}
for f in sorted(files):
    print(f)
    if not compare_CoDi_metrics:
        if ('CoDi_' in f) and (not 'KLD' in f):
            continue
    if compare_CoDi_metrics:
        name = f.split('benchmark_CoDi_')[-1].replace('.csv', '')
    else:
        name = f.split('benchmark_')[-1].replace('.csv', '')
        if name == "CoDi_KLD":
            name = "CoDi"
    dfs[name] = pd.read_csv(f)
    print(len(dfs[name]), name)
    sns.scatterplot(data=dfs[name], x='Subsample', y='Accuracy')
plt.legend(labels=dfs.keys(), fontsize="7", loc ="best")  # center right, upper right
print('********', list(dfs.keys()))
plt.xticks(dfs[list(dfs.keys())[0]]['Subsample'].values, size=7)
plt.title(f'Cell type detection accuracy for subsampled {title_dataset_name}')
plt.box(False)
plt.xlabel('Subsample factor')
plt.ylim([0.6, 1.05])
plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight')