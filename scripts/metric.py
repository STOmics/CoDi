import os
import argparse as ap
import logging
import time
from collections import Counter

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

parser = ap.ArgumentParser(description="A script that calculates the preservance of marker genes in ST data.")
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
    default=None,
)

parser.add_argument(
    "--st_cell_type_path", help="CSV with spatially resolved dataset cell types", type=str, required=True
)


args = parser.parse_args()

adata_sc = sc.read_h5ad(args.sc_path)
adata_st = sc.read_h5ad(args.st_path)
annotation = args.annotation
annotation_st = args.annotation_st
st_cell_type_path = args.st_cell_type_path

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
    # keep only the genes with pval < 0.05
    for col in markers_df.columns:
        markers_per_type_dict[col] = markers_df.loc[pval_df[col] < 0.05, col].values

    # Marker gene count (number of cell types in which a gene is a marker gene)
    gene_marker_times = {gene: 0 for gene in adata_sc.var.index}
    for col in markers_df.columns:
        for gene in adata_sc.var.index:
            if gene in markers_per_type_dict[col]:
                gene_marker_times[gene] += 1
        print(f'Processed {col}')

    # reduce marker gene lists by removing non-selective ones
    specificity_thr = 0.25
    markers_per_type_reduced_dict = {}
    for col in markers_df.columns:
        reduced_gene_set = set()
        for gene in markers_per_type_dict[col]:
            if gene_marker_times[gene] <= int(len(markers_df.columns) * specificity_thr):  # 5
                reduced_gene_set.add(gene)
        markers_per_type_reduced_dict[col] = reduced_gene_set.copy()
        print(col, len(markers_per_type_reduced_dict[col]))

    # After this for loop all marker genes will be sorted by p-value per cell type
    for col, genes in markers_per_type_reduced_dict.items():
        df = pd.DataFrame(markers_df[markers_df[col].isin(genes)][col])
        df.loc[:, 'pval'] = pval_df.loc[df.index, col]
        df.sort_values(by='pval', inplace=True)  # Sort by p-val to obtain ranks
        markers_per_type_reduced_dict[col] = list(df[col])

    adata_sc.uns["markers_per_type_reduced_dict"] = markers_per_type_reduced_dict
    adata_sc.write_h5ad(os.path.basename(args.sc_path), compression='gzip')
else:
    markers_per_type_reduced_dict = adata_sc.uns["markers_per_type_reduced_dict"]
    logger.info("Found markers_per_type_reduced_dict in adata_sc.uns")

# Extract top 100 genes
markers_per_type_top = {}
markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"])
pval_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["pvals_adj"])
for col in markers_df.columns:
    markers_per_type_top[col] = markers_df.loc[pval_df[col] < 0.05, col].values
# After this for loop all marker genes will be sorted by p-value per cell type
for col, genes in markers_per_type_top.items():
    df = pd.DataFrame(markers_df[markers_df[col].isin(genes)][col])
    df.loc[:, 'pval'] = pval_df.loc[df.index, col]
    df.sort_values(by='pval', inplace=True)  # Sort by p-val to obtain ranks
    markers_per_type_top[col] = list(df[col])[:100]


# Read ST cell type annotations from CSV
st_cell_types_df = pd.read_csv(st_cell_type_path)
st_cell_types_df.set_index(st_cell_types_df.columns[0], inplace=True)
st_annotation = annotation_st if annotation_st in st_cell_types_df.columns else st_cell_types_df.columns[-1]

# Exclude cell types with less than <min_cells> cells
min_cells = 5
c = Counter(st_cell_types_df[st_annotation])
exclude_types = {el for el in c.elements() if c[el] <= min_cells}
st_cell_types_df.loc[:, st_annotation] = st_cell_types_df[st_annotation].apply(lambda x: x if x not in exclude_types else "FILTERED")

# Add annotation from CSV to AnnData so we can calculate marker genes
adata_st.obs = pd.merge(adata_st.obs, st_cell_types_df, left_index=True, right_index=True)

adata_st = adata_st[adata_st.obs[st_annotation] != "FILTERED"]

st_marker_time = time.time()
marker_time = np.round(st_marker_time - start_time, 3)
logger.info(
    f"Calculate ST marker genes - execution took {marker_time} so far"
)
# Calculate ST marker genes
logger.info("Calculating ST marker genes.")
sc.pp.filter_cells(adata_st, min_genes=100)
sc.pp.filter_genes(adata_st, min_cells=5)
sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)
adata_st.var_names_make_unique()
# Ensure 'st_annotation' column is of category dtype
if st_annotation in adata_st.obs.columns:
    adata_st.obs[st_annotation] = adata_st.obs[st_annotation].astype('category')
else:
    raise ValueError(f"Column '{st_annotation}' not found in adata_st.obs")
sc.tl.rank_genes_groups(adata_st, groupby=st_annotation, use_raw=False)
markers_st_df = pd.DataFrame(adata_st.uns["rank_genes_groups"]["names"])
markers_st = list(np.unique(markers_st_df.melt().value.values))
pval_st_df = pd.DataFrame(adata_st.uns["rank_genes_groups"]["pvals_adj"])

st_marker_time2 = time.time()
marker_time = np.round(st_marker_time2 - st_marker_time, 3)
logger.info(
    f"Calculate of ST marker genes took {marker_time} so far"
)

report_df = pd.DataFrame(columns=["retention type", "sc_marker_genes", "st_marker_genes", "st_cell_types", "retention percentage"])
report_per_ctype_df = pd.DataFrame(columns=["Retention type", "Cell type", "Number of cells", "Number of SC marker genes", "Number of ST marker genes", "Number of ST lost genes", "Retention percentage"])
for sc_dict_name, sc_dict in zip(["unique marker genes", "top 100 marker genes"],
                                 [markers_per_type_reduced_dict, markers_per_type_top]):

    # Remove genes that do not exist in ST
    for ctype, genes in sc_dict.items():
        st_genes = list(adata_st.var.index)
        sc_dict[ctype] = list(filter(lambda x: x in st_genes, sc_dict[ctype]))

    markers_per_type_st_dict = dict()
    for col in set(markers_st_df.columns).intersection(set(sc_dict.keys())):
        df = pd.DataFrame(markers_st_df[col])
        df.loc[:, 'pval'] = pval_st_df.loc[df.index, col]
        df.sort_values(by='pval', inplace=True)  # Sort by p-val to obtain ranks
        df = df[df['pval'] <= 0.05]  # Keep only significant marker genes
        df.reset_index(inplace=True, drop=True)
        sc_st_df = pd.DataFrame(sc_dict[col], columns=['Gene'])
        sc_st_df = sc_st_df.merge(df[col].reset_index(), left_on='Gene', right_on=col)[['Gene', 'index']]
        sc_st_df.columns = ['Gene', 'Rank']
        markers_per_type_st_dict[col] = list(sc_st_df['Gene'])
        # display(sc_st_df)
        # sns.histplot(sc_st_df['Rank'])
        # plt.show()

    lost_genes = 0
    total_marker_genes_sc = 0
    for ctype, genes in markers_per_type_st_dict.items():  # Can be merged with previous for loop but it is more clear
        marker_genes_sc = len(sc_dict[ctype])
        dif = marker_genes_sc - len(genes)
        lost_genes += dif
        total_marker_genes_sc += marker_genes_sc
        num_of_cells = len(st_cell_types_df[st_cell_types_df[st_annotation] == ctype])
        report_per_ctype_df.loc[sc_dict_name+ctype, :] = [sc_dict_name, ctype, num_of_cells, marker_genes_sc, len(genes), dif, 
                                                          np.round(100 * (marker_genes_sc - dif) / marker_genes_sc, 2)]

    logger.info(f'Total scRNA {sc_dict_name} that also exist in ST (for {len(markers_st_df.columns)} cell types): {total_marker_genes_sc}')
    logger.info(f'Remained {sc_dict_name}  in ST: {total_marker_genes_sc - lost_genes}')
    report_df.loc[sc_dict_name, :] = [sc_dict_name, total_marker_genes_sc, (total_marker_genes_sc - lost_genes), len(markers_st_df.columns), np.round(100 * (total_marker_genes_sc - lost_genes) / total_marker_genes_sc, 2)]
    st_marker_time3 = time.time()
    marker_time = np.round(st_marker_time3 - st_marker_time2, 3)
    logger.info(
        f"Calculate of ST marker genes for {sc_dict_name} took {marker_time}"
    )

reports_path = "data/reports_final/"
if not os.path.exists(reports_path):
    os.makedirs(reports_path)
outfname = reports_path + st_cell_type_path.split("/")[-1]
report_df.to_csv(outfname, index=False)
outfname_ctype = outfname.replace('.csv', '_per_ctype.csv')
report_per_ctype_df.to_csv(outfname_ctype, index=False)
logger.info(f"Results written to {outfname} and {outfname_ctype}.") 
end_time = time.time()
total_time = np.round(end_time - start_time, 3)
logger.info(
    f"Execution took {total_time}"
)
