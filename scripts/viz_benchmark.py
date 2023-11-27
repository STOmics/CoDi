import glob as glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


files = glob.glob('../test/' + 'benchmark_*.csv')
dfs = {}
for f in files:
    # if 'ssi' in f:
    #     if not 'KLD' in f:
    #         continue
    name = f.split('_')[-1].replace('.csv', '')
    dfs[name] = pd.read_csv(f)
    sns.scatterplot(data=dfs[name], x='Subsample', y='Accuracy')
plt.legend(labels=dfs.keys())
plt.xticks(dfs[name]['Subsample'].values, size=7)
plt.title('Cell type detection using different distances on Mouse brain 4K dataset')
plt.savefig('comparison.png', dpi=150)