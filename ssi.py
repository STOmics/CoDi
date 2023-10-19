import argparse as ap
from collections import Counter
import logging
import multiprocessing as mp
import random
import time
import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
import seaborn as sns
from tqdm import tqdm


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
    "-d",
    "--distance",
    help="Distance metric used to measure the distance between a point and a distribution of points",
    type=str,
    required=False,
    default="KLD",
    choices={"mahalanobis", "KLD", "wasserstein"},
)
parser.add_argument(
    "--num_markers",
    help="Number of marker genes",
    type=int,
    required=False,
    default=100,
)
parser.add_argument(
    "--n_jobs",
    help="Number of jobs to run in parallel. -1 means using all available processors",
    type=int,
    required=False,
    default=-1,
)

args = parser.parse_args()

adata_sc = sc.read_h5ad(args.sc_path)
adata_st = sc.read_h5ad(args.st_path)

# Filter cells and genes
# sc.pp.filter_cells(adata_sc, min_genes=100)
# sc.pp.filter_genes(adata_sc, min_cells=5)
# sc.pp.filter_cells(adata_st, min_genes=100)
# sc.pp.filter_genes(adata_st, min_cells=5)

# Read datasets and check if matrix is sparse
sc_df_raw = pd.DataFrame(
    adata_sc.X.toarray() if issparse(adata_sc.X) else adata_sc.X,
    index=adata_sc.obs.index, columns=adata_sc.var.index
).copy()
st_df_raw = pd.DataFrame(
    adata_st.X.toarray() if issparse(adata_st.X) else adata_st.X,
    index=adata_st.obs.index, columns=adata_st.var.index
).copy()

# Calculate marker genes
start_marker = time.time()
sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)
adata_sc.var_names_make_unique()
sc.pp.highly_variable_genes(adata_sc, inplace=True, n_top_genes=200)

sc.tl.rank_genes_groups(adata_sc, groupby=args.annotation, use_raw=False)
markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[
    0 : args.num_markers, :
]
markers = list(np.unique(markers_df.melt().value.values))

# visualize results
# sc.pl.rank_genes_groups(adata_sc, key='rank_genes_groups_filtered')
# visualize results using dotplot
# sc.pl.rank_genes_groups_dotplot(adata_sc, key='rank_genes_groups_filtered')
# hvgs = list(adata_sc.var[adata_sc.var.highly_variable].index)
markers_intersect = list(set(markers).intersection(adata_st.var.index))
logger.info(
    f"Using {len(markers_intersect)} single cell marker genes that exist in ST dataset"
)
end_marker = time.time()
marker_time = np.round(end_marker - start_marker, 3)
logger.info(
    f"Calculation of marker genes took {marker_time}"
)
sc_df = sc_df_raw.loc[:, markers_intersect]
st_df = st_df_raw.loc[:, markers_intersect]
cell_types = set(adata_sc.obs[args.annotation])


# Algo
def create_subsets(gene_set, num_of_subsets=10):
    num_of_elem = len(gene_set)
    subsets = []
    for s in range(num_of_subsets):
        subset_size = random.randint(int(num_of_elem * 0.3), int(num_of_elem * 0.8))
        subsets.append(random.sample(gene_set, subset_size))
    return subsets


# *****************************************
# Precalculate inverse covariance matrices
# *****************************************
sc_dfs = {}
sc_icms = {}
sc_mean = {}
num_of_subsets = 50
subsets = create_subsets(markers_intersect, num_of_subsets=num_of_subsets)
for ty in cell_types:
    sc_dfs[ty] = []
    sc_icms[ty] = []
    sc_mean[ty] = []
    for sub_id, subset in enumerate(subsets):
        subset_df = sc_df[adata_sc.obs[args.annotation] == ty][subset]
        sc_dfs[ty].append(subset_df)
        cm = np.cov(subset_df.values, rowvar=False)  # Calculate covariance matrix
        sc_icms[ty].append(
            np.linalg.pinv(cm)
        )  # Use pseudo inverse to avoid error singular matrix (determinant = 0)
        sc_mean[ty].append(
            subset_df.mean()
        )  # ...because of presence of genes with all zero values per cell type

# *****************************************
# For all ST cells, for all subsets, for all cell types
# *****************************************
start = time.time()
num_cpus_used = mp.cpu_count() if args.n_jobs == -1 else args.n_jobs
assigned_types = []

iis = [ii for ii in range(len(st_df))]

def per_cell(ii):
    best_matches_subsets = []
    for subset_id, subset in enumerate(subsets):
        best_match = {"cell_type": "", "dist": 9999999}
        for cell_type in cell_types:
            if args.distance == "mahalanobis":
                distance = mahalanobis(
                    st_df.iloc[ii, :][subset].values,
                    sc_mean[cell_type][subset_id].values,
                    sc_icms[cell_type][subset_id],
                )
            elif args.distance == "KLD":
                distance = entropy(
                    st_df.iloc[ii, :][subset].values,
                    sc_mean[cell_type][subset_id].values,
                )
            elif args.distance == "wasserstein":
                distance = wasserstein_distance(
                    st_df.iloc[ii, :][subset].values,
                    sc_mean[cell_type][subset_id].values,
                )

            if distance < best_match["dist"]:
                best_match = {"cell_type": cell_type, "dist": distance}
        best_matches_subsets.append(best_match)

    # Majority voting
    cn = Counter([x["cell_type"] for x in best_matches_subsets])
    best_match_subset = {
        "cell_type": cn.most_common(1)[0][0],
        "confidence": np.round(cn.most_common(1)[0][1] / num_of_subsets, 3),
    }
    pbar.update(1)  # global variable
    return (ii, best_match_subset)

pbar = tqdm(total=len(st_df))
with mp.Pool(processes=num_cpus_used) as pool:
    assigned_types = pool.map(per_cell, iis)
    
assigned_types.sort(key=lambda x: x[0])
assigned_types = [at[1] for at in assigned_types]
end = time.time()
logger.info(f"Execution took: {end - start}s")
adata_st.obs["sc_type"] = [x["cell_type"] for x in assigned_types]
sns.histplot([x["confidence"] for x in assigned_types])
plt.savefig(f"ssi_confidence_hist__{args.distance}.png", dpi=120, bbox_inches="tight")
adata_st.write_h5ad(os.path.basename(args.st_path).replace(".h5ad", "_ssi.h5ad"))

# Visualisation
def plot_spatial(
    adata, annotation, ax: Axes, spot_size: float, palette=None, title: str = ""
):
    """
    Scatter plot in spatial coordinates.

    Parameters:
        - adata (AnnData): Annotated data object which represents the sample
        - annotation (str): adata.obs column used for grouping
        - ax (Axes): Axes object used for plotting
        - spot_size (int): Size of the dot that represents a cell. We are passing it as a diameter of the spot, while
                the plotting library uses radius therefore it is multiplied by 0.5
        - palette (dict): Dictionary that represents a mapping between annotation categories and colors
        - title (str): Title of the figure

    """
    s = spot_size * 0.5
    data = adata
    ax = sns.scatterplot(
        data=data.obs,
        hue=annotation,
        x=data.obsm["spatial"][:, 0],
        y=data.obsm["spatial"][:, 1],
        ax=ax,
        s=s,
        linewidth=0,
        palette=palette,
        marker=".",
    )
    ax.invert_yaxis()
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)

# TODO: Add visualize parameter
# palette = {
#     "DGGRC2": "#ffffcc",
#     "ACNT1": "r",
#     "TEGLU10": "#a1dab4",
#     "TEGLU17": "#41b6c4",
# }
# fig, axs = plt.subplots(2, 2, figsize=(14, 14))
# sns.histplot(adata_st.obs["sc_type"], ax=axs[0][0])
# sns.histplot(adata_sc.obs["cell_subclass"], ax=axs[0][1])
# plot_spatial(
#     adata_st,
#     annotation=f"sc_type",
#     spot_size=30,
#     palette=palette,
#     ax=axs[1][0],
# )
# plot_spatial(
#     adata_st, annotation="celltype", spot_size=30, palette=palette, ax=axs[1][1]
# )
# plt.savefig(f"ssi_{args.distance}.png", dpi=120, bbox_inches="tight")
