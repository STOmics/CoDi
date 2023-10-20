import os
import argparse as ap
import logging
import time

import scanpy as sc
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

start_time = time.time()
logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = ap.ArgumentParser(description="A script that performs SSI.")
parser.add_argument(
    "--sc_path", help="A single cell reference dataset", type=str, required=True
)
parser.add_argument(
    "--st_path", help="A spatially resolved dataset", type=str, required=True
)
parser.add_argument(
    "-a",
    "--annotation",
    help="Annotation label for cell types",
    type=str,
    required=False,
    default="cell_subclass",
)
parser.add_argument(
    "-c",
    "--annotation_st",
    help="Annotation label for ST cell types",
    type=str,
    required=False,
    default="celltype",
)

args = parser.parse_args()

adata_sc = sc.read_h5ad(args.sc_path)
adata_st = sc.read_h5ad(args.st_path)
annotation = args.annotation
annotation_st = args.annotation_st


if "markers_per_type_reduced_dict" not in adata_sc.uns:
    # Calculate marker genes
    if "rank_genes_groups" not in adata_sc.uns:   
        sc.pp.filter_cells(adata_sc, min_genes=200)
        sc.pp.filter_genes(adata_sc, min_cells=50)
        sc.pp.normalize_total(adata_sc, target_sum=1e4)
        sc.pp.log1p(adata_sc)
        adata_sc.var_names_make_unique()
        print('Start calculating of marker genes.')
        sc.tl.rank_genes_groups(adata_sc, groupby=annotation, use_raw=False)
    else:
        logger.info("Found rank_genes_groups in adata_sc.uns")
    
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"])
    markers = list(np.unique(markers_df.melt().value.values))
    pval_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["pvals_adj"])
    markers_per_type_dict = {}
    for col in markers_df.columns:
        markers_per_type_dict[col] = markers_df.loc[pval_df[col] < 0.05, col].values

    markers_per_type_reduced_dict = {}
    for col in markers_df.columns:
        reduced_gene_set = set()
        for gene in markers_per_type_dict[col]:
            if gene_marker_times[gene] <= int(len(markers_df.columns) * 0.25):  # 5
                reduced_gene_set.add(gene)
        markers_per_type_reduced_dict[col] = reduced_gene_set.copy()
        print(col, len(markers_per_type_reduced_dict[col]))

    # After this for loop all marker genes will be sorted by p-value per cell type
    for col, genes in markers_per_type_reduced_dict.items():
        df = pd.DataFrame(markers_df[markers_df[col].isin(genes)][col])
        df.loc[:, 'pval'] = pval_df.loc[df.index, col]
        df.sort_values(by='pval', inplace=True)  # Sort by p-val to obtain ranks
        markers_per_type_reduced_dict[col] = list(df[col])
else:
    markers_per_type_reduced_dict = adata_sc.uns["markers_per_type_reduced_dict"]
    logger.info("Found markers_per_type_reduced_dict in adata_sc.uns")
    
# Calculate ST marker genes
if "rank_genes_groups" not in adata_st.uns:
    sc.pp.filter_cells(adata_st, min_genes=100)
    sc.pp.filter_genes(adata_st, min_cells=5)
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st.var_names_make_unique()
    sc.tl.rank_genes_groups(adata_st, groupby=annotation_st, use_raw=False)
    markers_st_df = pd.DataFrame(adata_st.uns["rank_genes_groups"]["names"])
    markers_st = list(np.unique(markers_st_df.melt().value.values))
    pval_st_df = pd.DataFrame(adata_st.uns["rank_genes_groups"]["pvals_adj"])

# Remove genes that does not exist in ST
for ctype, genes in markers_per_type_reduced_dict.items():
    st_genes = list(adata_st.var.index)
    markers_per_type_reduced_dict[ctype] = \
    list(filter(lambda x: x in st_genes, markers_per_type_reduced_dict[ctype]))

markers_per_type_st_dict = dict()
for col in markers_st_df.columns:
    df = pd.DataFrame(markers_st_df[col])
    df.loc[:, 'pval'] = pval_st_df.loc[df.index, col]
    df.sort_values(by='pval', inplace=True)  # Sort by p-val to obtain ranks
    df = df[df['pval'] <= 0.05]  # Keep only significant marker genes
    df.reset_index(inplace=True, drop=True)
    sc_st_df = pd.DataFrame(markers_per_type_reduced_dict[col], columns=['Gene'])
    sc_st_df = sc_st_df.merge(df[col].reset_index(), left_on='Gene', right_on=col)[['Gene', 'index']]
    sc_st_df.columns = ['Gene', 'Rank']
    markers_per_type_st_dict[col] = list(sc_st_df['Gene'])
    # display(sc_st_df)
    # sns.histplot(sc_st_df['Rank'])
    # plt.show()

lost_genes = 0
total_marker_genes_sc = 0
for ctype, genes in markers_per_type_st_dict.items():  # Can be merged with previous for loop but it is more clear
    marker_genes_sc = len(markers_per_type_reduced_dict[ctype])
    dif = marker_genes_sc - len(genes)
    lost_genes += dif
    total_marker_genes_sc += marker_genes_sc
    print(ctype, dif)

logger.info(f'Total marker genes in scRNA that also exist in ST (for {len(markers_st_df.columns)} cell types): {total_marker_genes_sc}')
logger.info(f'Remained marker genes in ST: {total_marker_genes_sc - lost_genes}')

end_time = time.time()
total_time = np.round(end_time - start_time, 3)
logger.info(
    f"Execution took {total_time}"
)


