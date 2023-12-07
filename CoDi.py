import argparse as ap
from collections import Counter
import logging
import multiprocessing as mp
import sys
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
from scipy.special import rel_entr, kl_div
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
import seaborn as sns
from tqdm import tqdm

# import core

random.seed(3)


def create_subsets(gene_set, num_of_subsets=10):
    num_of_elem = len(gene_set)
    subsets = []
    for s in range(num_of_subsets):
        subset_size = random.randint(int(num_of_elem * 0.3), int(num_of_elem * 0.8))
        subsets.append(random.sample(gene_set, subset_size))
    return subsets


def hellinger(p, q):
    """Hellinger distance between distributions"""
    p = np.array(p)
    q = np.array(q)
    result = np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / np.sqrt(2)
    return result


def binary_distance(p, q):
    """Binary distance between distributions.
    Sum all the positional pairs in both distributions which are
    of different status: one zero, while other nonzero."""
    return np.sum(p.astype(bool) ^ q.astype(bool))


def per_cell(ii):
    best_matches_subsets = []
    for subset_id, subset in enumerate(subsets):
        best_match = {"cell_type": "", "dist": 9999999}
        for cell_type in cell_types:
            st_distrib = st_df.iloc[ii, :][subset].values.astype(float)
            sc_distrib = sc_mean[cell_type][subset_id].values.astype(float)
            # normalize to sum 1.0 if sum is not 0
            st_distrib_norm = (
                st_distrib / np.sum(st_distrib, axis=0, keepdims=True)
                if np.sum(st_distrib, axis=0, keepdims=True) != 0
                else st_distrib
            )
            sc_distrib_norm = (
                sc_distrib / np.sum(sc_distrib, axis=0, keepdims=True)
                if np.sum(sc_distrib, axis=0, keepdims=True) != 0
                else sc_distrib
            )
            if args.distance == "mahalanobis":
                distance = mahalanobis(
                    st_distrib_norm,
                    sc_distrib_norm,
                    sc_icms[cell_type][subset_id],
                )
            elif args.distance == "relativeEntropy":
                distance = rel_entr(
                    st_distrib_norm,
                    sc_distrib_norm,
                ).sum()
            elif args.distance == "KLD":
                distance = kl_div(
                    st_distrib_norm,
                    sc_distrib_norm,
                ).sum()
            elif args.distance == "wasserstein":
                distance = wasserstein_distance(
                    st_distrib_norm,
                    sc_distrib_norm,
                )
            elif args.distance == "hellinger":
                distance = hellinger(
                    st_distrib_norm,
                    sc_distrib_norm,
                )
            elif args.distance == "binary":
                distance = binary_distance(
                    np.floor(st_distrib),
                    np.floor(sc_distrib),
                )

            if distance < best_match["dist"]:
                best_match = {"cell_type": cell_type, "dist": distance}
        best_matches_subsets.append(best_match)

    # Majority voting
    cn = Counter([x["cell_type"] for x in best_matches_subsets])
    best_match_subset = {
        "cell_type": cn.most_common(1)[0][0],
        "confidence": np.round(cn.most_common(1)[0][1] / num_of_subsets, 3),
        "ct_probabilities": [np.round(cn[cellt]/num_of_subsets, 3) if cellt in cn.keys() else 0.0 for cellt in adata_st.obsm['probabilities_dist'].columns],
    }
    pbar.update(1)  # global variable
    return (ii, best_match_subset)


def plot_spatial(
    adata, annotation, ax: plt.Axes, spot_size: float, palette=None, title: str = ""
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
    palette = sns.color_palette("coolwarm", as_cmap=True)
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
        palette=palette
        if ("float" in str(type(adata.obs[annotation][0])).lower())
        else None,
        marker=".",
    )
    ax.invert_yaxis()
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)


logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = ap.ArgumentParser(description="A script that performs CoDi.")
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
    choices={
        "mahalanobis",
        "KLD",
        "wasserstein",
        "relativeEntropy",
        "hellinger",
        "binary",
        "none",
    },
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
parser.add_argument("-c", "--contrastive", action="store_true")

args = parser.parse_args()


adata_sc = sc.read_h5ad(args.sc_path)
adata_st = sc.read_h5ad(args.st_path)
adata_sc.var_names_make_unique()
adata_st.var_names_make_unique()

# Contrastive part
if args.contrastive:
    import core
    
    queue = mp.Queue()

    contrastive_proc = mp.Process(
        target=core.contrastive_process,
        kwargs=dict(
            sc_path=args.sc_path,
            st_path=args.st_path,
            adata_sc=adata_sc,
            adata_st=adata_st,
            annotation_sc=args.annotation,
            epochs=50,
            embedding_dim=32,
            encoder_depth=4,
            classifier_depth=2,
            queue=queue,
        ),
        name="Contrastive process",
    )
    contrastive_proc.start()


if args.distance == "none":
    if args.contrastive:
        df_probabilities = queue.get()  # FIFO (ordering in contrastive.py)
        adata_st.obsm["probabilities_contrastive"] = df_probabilities
        predictions = queue.get()
        adata_st.obs["pred_contrastive"] = predictions
        contrastive_proc.join()
        # Write CSV and H5AD
        adata_st.obs.index.name = "cell_id"
        adata_st.obs["pred_contrastive"].to_csv(
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_contrastive_{args.distance}.csv"
            )
        )
        adata_st.write_h5ad(
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_ssi_{args.distance}.h5ad"
            )
        )
    logger.info(f"No distance metric specified, exiting...")
    sys.exit()

# Filter cells and genes
# sc.pp.filter_cells(adata_sc, min_genes=100)
# sc.pp.filter_genes(adata_sc, min_cells=5)
# sc.pp.filter_cells(adata_st, min_genes=100)
# sc.pp.filter_genes(adata_st, min_cells=5)

# Read datasets and check if matrix is sparse
# sc_df_raw = pd.DataFrame(
#     adata_sc.X.toarray() if issparse(adata_sc.X) else adata_sc.X,
#     index=adata_sc.obs.index, columns=adata_sc.var.index
# ).copy()
# st_df_raw = pd.DataFrame(
#     adata_st.X.toarray() if issparse(adata_st.X) else adata_st.X,
#     index=adata_st.obs.index, columns=adata_st.var.index
# ).copy()

# Calculate marker genes
start_marker = time.time()
if "rank_genes_groups" not in adata_sc.uns:
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.highly_variable_genes(adata_sc, inplace=True, n_top_genes=200)
    sc.tl.rank_genes_groups(adata_sc, groupby=args.annotation, use_raw=False)
else:
    logger.info(f"***d Using precalculated marker genes in input h5ad.")

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

# Extract gene expressions only from marker genes
select_ind = [np.where(adata_sc.var.index == gene)[0][0] for gene in markers_intersect]
sc_df = adata_sc.X.tocsr()[:, select_ind].todense() if issparse(adata_sc.X) else adata_sc.X[:, select_ind]
sc_df = pd.DataFrame(sc_df, columns=markers_intersect, index=adata_sc.obs.index)
select_ind = [np.where(adata_st.var.index == gene)[0][0] for gene in markers_intersect]
st_df = adata_st.X.tocsr()[:, select_ind].todense() if issparse(adata_st.X) else adata_st.X[:, select_ind]
st_df = pd.DataFrame(st_df, columns=markers_intersect, index=adata_st.obs.index)
cell_types = set(adata_sc.obs[args.annotation])
adata_st.obsm['probabilities_dist'] = pd.DataFrame(index=adata_st.obs.index, columns=cell_types)

# Algo
# *****************************************
# Precalculate necessary subset matrices
# *****************************************
# sc_dfs = {}
sc_icms = {}
sc_mean = {}
num_of_subsets = 20
subsets = create_subsets(markers_intersect, num_of_subsets=num_of_subsets)
for ty in cell_types:
    # sc_dfs[ty] = []
    sc_icms[ty] = []
    sc_mean[ty] = []
    for sub_id, subset in enumerate(subsets):
        subset_df = sc_df[adata_sc.obs[args.annotation] == ty][subset]
        # sc_dfs[ty].append(subset_df)
        if args.distance == "mahalanobis":
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

logger.info(f"Starting parallel per cell calculation of distances.")
pbar = tqdm(total=len(st_df))
with mp.Pool(processes=num_cpus_used) as pool:
    assigned_types = pool.map(per_cell, iis)

assigned_types.sort(key=lambda x: x[0])
assigned_types = [at[1] for at in assigned_types]
adata_st.obs["CoDi_dist"] = [x["cell_type"] for x in assigned_types]
adata_st.obs["confidence_dist"] = [x["confidence"] for x in assigned_types]
adata_st.obsm['probabilities_dist'].iloc[:, :] = [x["ct_probabilities"] for x in assigned_types]

# sns.histplot([x["confidence"] for x in assigned_types])
# plt.savefig(f"CoDi_confidence_hist__{args.distance}.png", dpi=120, bbox_inches="tight")

if args.contrastive:
    df_probabilities = queue.get()  # FIFO (ordering in contrastive.py)
    adata_st.obsm["probabilities_contrastive"] = df_probabilities
    predictions = queue.get()
    adata_st.obs["CoDi_contrastive"] = predictions
    contrastive_proc.join()

# combine contrastive and distance results
dist_weight = 0.5
if not args.contrastive:
    assert 'probabilities_contrastive' in adata_st.obsm, "Missing 'probabilities_contrastive' in adata_st.obsm."
    adata_st.obsm['probabilities'] = adata_st.obsm['probabilities_contrastive'].add(adata_st.obsm['probabilities_dist'] * dist_weight)
    adata_st.obs['CoDi'] = np.array([prow.idxmax(axis=1) for _, prow in adata_st.obsm['probabilities'].iterrows()]).astype('str') 
else:
    adata_st.obs['CoDi'] = adata_st.obs['CoDi_dist']

end = time.time()
logger.info(
    f"CoDi execution took: {end - start}s"
    )

# Write CSV and H5AD
adata_st.obs.index.name = "cell_id"
if args.contrastive:
    # Write CSV of contrastive results
    adata_st.obs["CoDi_contrastive"].to_csv(
        os.path.basename(args.st_path).replace(".h5ad", "_contrastive.csv")
    )
# Write CSV and H5AD of final combined results
adata_st.obs[["CoDi"]].to_csv(
    os.path.basename(args.st_path).replace(".h5ad", f"_CoDi_{args.distance}.csv")
)
adata_st.write_h5ad(
    os.path.basename(args.st_path).replace(".h5ad", f"_CoDi_{args.distance}.h5ad")
)

if "spatial" in adata_st.obsm_keys():
    fig, axs = plt.subplots(1, 2, figsize=(14, 14))
    plot_spatial(
        adata_st, annotation=f"CoDi", spot_size=50, ax=axs[0], title="Cell types"
    )
    plot_spatial(
        adata_st,
        annotation=f"confidence_dist",
        spot_size=50,
        ax=axs[1],
        title="Confidence map",
    )
    plt.savefig(
        os.path.basename(args.st_path).replace(".h5ad", f"_CoDi_{args.distance}.png"),
        dpi=120,
        bbox_inches="tight",
    )

end = time.time()
logger.info(
    f"Total execution time: {end - start}s"
    )