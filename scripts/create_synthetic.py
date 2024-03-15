import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
import argparse as ap
import os

def create_synthetic(path_to_anndata, percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
    adata = sc.read_h5ad(path_to_anndata)
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index).copy()
    adata_copy = adata.copy()
    
    a = np.mean(np.count_nonzero(df != 0, axis=1)) / df.shape[1]
    print(a)

    for perc in percentages:
        df_copy = df.copy()
        for ind, row in df_copy.iterrows():
            non_zero_indices = np.nonzero(row.values)[0]
            num_elements_to_zero = int(perc * len(non_zero_indices))
            selected_indices = np.random.choice(non_zero_indices, num_elements_to_zero, replace=False)
            row[selected_indices] = 0
        adata_copy.X = df_copy.values
        adata_copy.X = scipy.sparse.csr_matrix(adata_copy.X)
        adata_fname = os.path.basename(path).rstrip('.h5ad') + '_' + str(perc) + '.h5ad'
        adata_copy.write(adata_fname)
        print(f'Writing {adata_fname}...')




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--path', help='Path to anndata object.', type=str, required=True)
    parser.add_argument('--percentages', help='Comma separated list of percentages', type=str, required=False, default="0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9")
    
    args = parser.parse_args()
    path = args.path
    perc = [float(s) for s in args.percentages.split(",")]

    create_synthetic(path, perc)
