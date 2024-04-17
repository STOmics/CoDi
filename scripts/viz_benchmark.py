import os
import glob as glob
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for dataset_prefix, compare_CoDi_metrics, title_dataset_name in zip(['config_40k_visium', 'config_breast', 'config_40k_visium_distances', 'config_breast_distances'], [False, False, True, True], ['Visium Mouse brain', 'snRNA breast', 'Visium Mouse brain', 'snRNA breast']):
    dataset_name = dataset_prefix.rsplit('config_')[1]
    output_suffix = ''
    cur_path = os.path.dirname(os.path.realpath(__file__))
    if compare_CoDi_metrics:
        files = glob.glob(os.path.join('../test/') + f'{dataset_prefix}_benchmark_CoDi_*.csv')
        output_path = f'comparison_CoDi_{dataset_name}{output_suffix}'
    else:
        files = glob.glob(os.path.join(cur_path, '../test/') + f'{dataset_prefix}_benchmark_*.csv')
        output_path = f'comparison_{dataset_name}{output_suffix}'
    dfs = {}
    for metric in ['Accuracy', 'F-score']:
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
            if name == 'CoDi':
                dfs[name].loc[:, 'Subsample_rand'] = dfs[name]['Subsample'] + random.uniform(0, 0.02) #To avoid covering close markers
            else:
                dfs[name].loc[:, 'Subsample_rand'] = dfs[name]['Subsample']
            sns.scatterplot(data=dfs[name], x='Subsample_rand', y=metric)
        plt.legend(labels=dfs.keys(), fontsize="7", loc ="best")  # center right, upper right
        print('********', list(dfs.keys()))
        plt.xticks(dfs[list(dfs.keys())[0]]['Subsample'].values, size=7)
        plt.title(f'Cell Type Detection {metric} for subsampled {title_dataset_name}')
        plt.box(False)
        plt.xlabel('Subsample factor')
        plt.ylim([0.2, 1.05])
        plt.grid(linestyle='--', linewidth=0.5)
        plt.savefig(f'{output_path}_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()