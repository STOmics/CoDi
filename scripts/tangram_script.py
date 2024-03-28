import os
import time
import subprocess

import warnings
import argparse as ap
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tangram as tg


if __name__ == '__main__':
    start_time = time.perf_counter()
    
    parser = ap.ArgumentParser(description='A script that performs Tangram cell type annotation on ST data, given an SC reference.')
    parser.add_argument('--sc_path', help='Path to .h5ad file with scRNA data.', type=str, required=True)
    parser.add_argument('--st_path', help='Path to .h5ad file with ST data.', type=str, required=True)
    parser.add_argument('-a','--annotation', help='Label of SC .obs column containing SC cell types', type=str, required=True)
    parser.add_argument('--annotation_st', help='Label of ST .obs column containing ST cell types', type=str, required=False, default=None)
    parser.add_argument('--num_epochs', help='Number of epochs for mapping process.', type=int, required=False, default=500)
    parser.add_argument('--common_ct', help='Flag indicating that only cell types existing in ST should be used from SC.', required=False, default=False, action='store_true')
    parser.add_argument('--plotting', help='Level of plotting (images are saved). 0 - none, >0 - spatial plots', type=int, required=False, default=0)
    parser.add_argument('--spot_size', help='Spot size for plotting', type=float, required=False, default=30)

    args = parser.parse_args()
    
    if args.common_ct == True and args.annotation_st == None:
        raise ValueError("ST annotation label is needed for finding intersection of cell types.")

    mem_logger_fname = os.path.basename(args.st_path).replace(
        ".h5ad", "_cpu_gpu_memlog.csv"
    )
    if os.path.isfile(mem_logger_fname):
        os.remove(mem_logger_fname)

    logger_pid = subprocess.Popen(
        [
            "python",
            "scripts/log_gpu_cpu_stats.py",
            mem_logger_fname,
        ]
    )
    print("Started logging compute utilisation")

    # read the .h5ad files
    adata_sc = sc.read(args.sc_path)
    adata_sc
    adata_sc.obs['cell_subclass'] = adata_sc.obs[args.annotation].astype('str')
    adata_st = sc.read(args.st_path)
    if args.annotation_st is not None:
        adata_st.obs[args.annotation_st] = adata_st.obs[args.annotation_st].astype('str')

    # data normalization - log1p
    sc.pp.log1p(adata_sc)
    sc.pp.log1p(adata_st)

    # Subset of scRNA data with cells belonging to the same types as st data
    # [TODO] make this a intersection of cell types, do not assume that sc has all cell types of ST
    if args.common_ct:
        cell_types_st = np.unique(adata_st.obs[args.annotation_st].values)
        adata_sc = adata_sc[adata_sc.obs['cell_subclass'].isin(cell_types_st),:]
    
    # place cell spatial coordinates in .obsm['spatial']
    # coordinates are expected in 'spatial', 'X_spatial', and 'spatial_stereoseq'
    if 'X_spatial' in adata_st.obsm:
        adata_st.obsm['spatial'] = adata_st.obsm['X_spatial'].copy()
    elif 'spatial_stereoseq' in adata_st.obsm:
        adata_st.obsm['spatial'] = np.array(adata_st.obsm['spatial_stereoseq'].copy())
    elif 'spatial' in adata_st.obsm:
        pass
    else:
        warnings.warn('''Spatial coordinates not found. Labels expected in:
            .obsm["spatial"] or\n
            .obsm["X_spatial"] or\n
            .obsm["spatial_stereoseq"]''')
        args.plotting = 0

    # PREPROCESSING
    # marker gene selection
    sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
    markers = list(np.unique(markers_df.melt().value.values))

    # We prepare the data using `pp_adatas`, which does the following:
    # - Takes a list of genes from user via the `genes` argument. These genes are used as training genes.
    # - Annotates training genes under the `training_genes` field, in `uns` dictionary, of each AnnData. 
    # - Ensure consistent gene order in the datasets (_Tangram_ requires that the the $j$-th column in each matrix correspond to the same gene).
    # - If the counts for a gene are all zeros in one of the datasets, the gene is removed from the training genes.
    # - If a gene is not present in both datasets, the gene is removed from the training genes.
    # - In the pp_adatas function, the gene names are converted to lower case to get rid of the inconsistent capitalization.
    #   If this is not wanted, you can set the parameter gene_to_lowercase = False
    tg.pp_adatas(adata_sc, adata_st, genes=markers)

    # find alignment
    ad_map = tg.map_cells_to_space(adata_sc, adata_st,
        # mode="cells",
        mode="clusters",
        cluster_label='cell_subclass',  # .obs field w cell types
        density_prior='uniform',
        num_epochs=args.num_epochs,
        # device="cuda:0",
        device='cpu',
    )

    # select the types with highest probability for cell type annotation
    adata_st.obs['tangram'] = np.nan
    for cell_name in ad_map.var_names:
        probabilities = ad_map.X[:, ad_map.var_names == cell_name]
        index_of_max = np.argmax(probabilities)
        adata_st.obs.loc[cell_name, 'tangram'] = ad_map.obs['cell_subclass'].values[index_of_max]

    # save the .h5ad file with tangram annotation
    # Write CSV and H5AD
    adata_st.obs.index.name = 'cell_id'
    adata_st.obs[["tangram"]].to_csv(os.path.basename(args.st_path).replace(".h5ad", "_tangram.csv"))
    adata_st.write_h5ad(os.path.basename(args.st_path).replace(".h5ad", "_tangram.h5ad"))

    # record execution time
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # End the background process logging the CPU and GPU utilisation.
    logger_pid.terminate()
    print("Terminated the compute utilisation logger background process")

    # read cpu and gpu memory utilization
    logger_df = pd.read_csv(mem_logger_fname)

    max_cpu_mem = logger_df.loc[:, "RAM"].max()
    max_gpu_mem = logger_df.loc[:, "GPU 0"].max()

    with open(
        os.path.basename(args.st_path).replace(".h5ad", "_tangram_time_mem.txt"), "w+"
    ) as text_file:
        text_file.write(
        f"Peak RAM Usage: {max_cpu_mem} MiB\nPeak GPU Usage: {max_gpu_mem} MiB\n Total time: {total_time:.4f} s"
    )

    # plot the mapping results compared to ST annotation
    if args.plotting > 0:
        # three cases
        # 1. no ST labels, plot only result
        if args.annotation_st == None:
            figure, axes = plt.subplots(nrows=1, ncols=1)
            figure.set_size_inches(10, 10)
            figure.set_dpi(200)
            sc.pl.spatial(adata_st, color='tangram', palette=None, frameon=False, show=False, ax=axes, spot_size=args.spot_size)
        # 2. ST labels exist but all SC cell types used
        elif not args.common_ct:
            figure, axes = plt.subplots(nrows=1, ncols=2)
            figure.set_size_inches(10, 20)
            figure.set_dpi(200)
            sc.pl.spatial(adata_st, color=args.annotation_st, palette=None, frameon=False, show=False, ax=axes[0], spot_size=args.spot_size)
            sc.pl.spatial(adata_st, color='tangram', palette=None, frameon=False, show=False, ax=axes[1], spot_size=args.spot_size)
        # 3. ST lables exist and only ST cell types used
        else:
            celltype_pred_cells = np.sort(adata_st.obs[args.annotation_st].unique())
            if f'{args.annotation_st}_colors' in adata_st.uns.keys():
                color_palette = {cellt: adata_st.uns[f'{args.annotation_st}_colors'][i] for i, cellt in enumerate(celltype_pred_cells)}
            else:
                default_color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
                color_palette = {cellt: default_color_palette[i] for i, cellt in enumerate(celltype_pred_cells)}
            tangram_color_palette = {cellt: color_palette[cellt] if cellt in color_palette.keys() else '#CCCCCC' for cellt in adata_sc.obs['cell_subclass'].unique()}
            figure, axes = plt.subplots(nrows=1, ncols=2)
            figure.set_size_inches(10, 20)
            figure.set_dpi(200)
            sc.pl.spatial(adata_st, color=args.annotation_st, palette=color_palette, frameon=False, show=False, ax=axes[0], spot_size=args.spot_size)
            sc.pl.spatial(adata_st, color='tangram', groups=list(celltype_pred_cells), palette=tangram_color_palette, frameon=False, show=False, ax=axes[1], spot_size=args.spot_size)
        figure.savefig(f'{os.path.splitext(os.path.split(args.st_path)[1])[0]}_spatial_ann_vs_tangram.png', dpi=200, bbox_inches='tight')