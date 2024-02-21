import datetime
import logging
import sys
import random
import time
import os
import warnings
import subprocess

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from memory_profiler import memory_usage
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import multiprocessing as mp
import argparse as ap

# from tqdm import tqdm

from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
from scipy.special import rel_entr, kl_div
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
from collections import Counter
from itertools import repeat


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


def per_cell(ii, subsets, cell_types, st_df, sc_mean, sc_icms):
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
        "confidence": np.round(cn.most_common(1)[0][1] / len(subsets), 3),
        "ct_probabilities": [
            np.round(cn[cellt] / len(subsets), 3) if cellt in cn.keys() else 0.0
            for cellt in cell_types
        ],
    }
    # pbar.update(1)
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
        palette=(
            palette
            if ("float" in str(type(adata.obs[annotation].iloc[0])).lower())
            else None
        ),
        marker=".",
    )
    ax.invert_yaxis()
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)


def main(args):
    start = time.time()

    adata_sc = sc.read_h5ad(args.sc_path)
    adata_st = sc.read_h5ad(args.st_path)
    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()
    adata_sc.obs_names_make_unique()
    adata_st.obs_names_make_unique()

    # place cell spatial coordinates in .obsm['spatial']
    # coordinates are expected in 'spatial', 'X_spatial', and 'spatial_stereoseq'
    if "X_spatial" in adata_st.obsm:
        adata_st.obsm["spatial"] = adata_st.obsm["X_spatial"].copy()
    elif "spatial_stereoseq" in adata_st.obsm:
        adata_st.obsm["spatial"] = np.array(adata_st.obsm["spatial_stereoseq"].copy())
    elif "spatial" in adata_st.obsm:
        pass
    else:
        warnings.warn(
            'Spatial coordinates not found. Labels expected in: \
                .obsm["spatial"] or\n \
                .obsm["X_spatial"] or\n \
                .obsm["spatial_stereoseq"]'
        )

    # Calculate marker genes
    start_marker = time.time()
    adata_sc.layers["counts"] = adata_sc.X.copy()
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

    markers_intersect = list(set(markers).intersection(adata_st.var.index))
    logger.info(
        f"Using {len(markers_intersect)} single cell marker genes that exist in ST dataset"
    )
    end_marker = time.time()
    marker_time = np.round(end_marker - start_marker, 3)
    logger.info(f"Calculation of marker genes took {marker_time:.2f}")

    # Contrastive part
    if args.contrastive:
        import core

        queue = mp.Queue()

        contrastive_proc = mp.Process(
            target=core.contrastive_process,
            kwargs=dict(
                sc_path=args.sc_path,
                st_path=args.st_path,
                adata_sc=adata_sc[:, markers_intersect],
                adata_st=adata_st[:, markers_intersect],
                annotation_sc=args.annotation,
                epochs=args.epochs,
                embedding_dim=args.emb_dim,
                encoder_depth=args.enc_depth,
                classifier_depth=args.class_depth,
                filename=filename,
                augmentation_perc=args.augmentation_perc,
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
        end = time.time()
        logger.info(f"Total execution time: {(end - start):.2f}s")
        return

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

    # Extract gene expressions only from marker genes
    select_ind = [
        np.where(adata_sc.var.index == gene)[0][0] for gene in markers_intersect
    ]
    sc_df = (
        adata_sc.X.tocsr()[:, select_ind].todense()
        if issparse(adata_sc.X)
        else adata_sc.X[:, select_ind]
    )
    sc_df = pd.DataFrame(sc_df, columns=markers_intersect, index=adata_sc.obs.index)
    select_ind = [
        np.where(adata_st.var.index == gene)[0][0] for gene in markers_intersect
    ]
    st_df = (
        adata_st.X.tocsr()[:, select_ind].todense()
        if issparse(adata_st.X)
        else adata_st.X[:, select_ind]
    )
    st_df = pd.DataFrame(st_df, columns=markers_intersect, index=adata_st.obs.index)
    cell_types = list(sorted(adata_sc.obs[args.annotation].unique()))
    adata_st.obsm["probabilities_dist"] = pd.DataFrame(
        index=adata_st.obs.index, columns=cell_types
    ).astype("float32")

    # Algo
    # *****************************************
    # Precalculate necessary subset matrices
    # *****************************************
    sc_icms = {}
    sc_mean = {}
    num_of_subsets = 10
    subsets = create_subsets(markers_intersect, num_of_subsets=num_of_subsets)
    for ty in cell_types:
        sc_icms[ty] = []
        sc_mean[ty] = []
        for sub_id, subset in enumerate(subsets):
            subset_df = sc_df[adata_sc.obs[args.annotation] == ty][subset]
            if args.distance == "mahalanobis":
                cm = np.cov(
                    subset_df.values, rowvar=False
                )  # Calculate covariance matrix
                sc_icms[ty].append(
                    np.linalg.pinv(cm)
                )  # Use pseudo inverse to avoid error singular matrix (determinant = 0)
            sc_mean[ty].append(
                subset_df.mean()
            )  # ...because of presence of genes with all zero values per cell type

    # *****************************************
    # For all ST cells, for all subsets, for all cell types
    # *****************************************
    num_cpus_used = mp.cpu_count() if args.n_jobs == -1 else args.n_jobs
    assigned_types = []

    iis = [ii for ii in range(len(st_df))]

    logger.info(f"Starting parallel per cell calculation of distances.")
    # pbar = tqdm(total=len(st_df))
    with mp.Pool(processes=num_cpus_used) as pool:
        assigned_types = pool.starmap(
            per_cell,
            zip(
                iis, repeat(subsets), repeat(cell_types), repeat(st_df), repeat(sc_mean), repeat(sc_icms)
            ),
            chunksize=30,
        )

    assigned_types.sort(key=lambda x: x[0])
    assigned_types = [at[1] for at in assigned_types]
    adata_st.obs["CoDi_dist"] = [x["cell_type"] for x in assigned_types]
    adata_st.obs["confidence_dist"] = [x["confidence"] for x in assigned_types]
    adata_st.obsm["probabilities_dist"].iloc[:, :] = [
        x["ct_probabilities"] for x in assigned_types
    ]

    # sns.histplot([x["confidence"] for x in assigned_types])
    # plt.savefig(f"CoDi_confidence_hist__{args.distance}.png", dpi=120, bbox_inches="tight")

    if args.contrastive:
        df_probabilities = queue.get()  # FIFO (ordering in contrastive.py)
        adata_st.obsm["probabilities_contrastive"] = df_probabilities
        predictions = queue.get()
        adata_st.obs["CoDi_contrastive"] = predictions
        adata_st.obs["confidence_contrastive"] = [
            np.round(prow.max(), 3)
            for _, prow in adata_st.obsm["probabilities_contrastive"].iterrows()
        ]
        contrastive_proc.join()

    # combine contrastive and distance results
    if args.contrastive:
        assert (
            "probabilities_contrastive" in adata_st.obsm
        ), "Missing 'probabilities_contrastive' in adata_st.obsm."
        adata_st.obsm["probabilities"] = (
            adata_st.obsm["probabilities_contrastive"] * (1.0 - args.dist_prob_weight)
        ).add(adata_st.obsm["probabilities_dist"] * args.dist_prob_weight)
        adata_st.obs["CoDi"] = np.array(
            [prow.idxmax() for _, prow in adata_st.obsm["probabilities"].iterrows()]
        ).astype("str")
        adata_st.obs["confidence"] = [
            np.round(prow.max(), 3)
            for _, prow in adata_st.obsm["probabilities"].iterrows()
        ]
    else:
        adata_st.obs["CoDi"] = adata_st.obs["CoDi_dist"]
        adata_st.obs["confidence"] = adata_st.obs["confidence_dist"]

    end = time.time()
    logger.info(f"CoDi execution took: {end - start}s")

    # Write CSV and H5AD
    adata_st.obs.index.name = "cell_id"
    # Write CSV and H5AD of final combined results
    if args.contrastive:
        adata_st.obs[
            [
                "CoDi_dist",
                "confidence_dist",
                "CoDi_contrastive",
                "confidence_contrastive",
                "CoDi",
                "confidence",
            ]
        ].to_csv(
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.csv"
            )
        )
    else:
        adata_st.obs[["CoDi_dist", "confidence_dist", "CoDi", "confidence"]].to_csv(
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.csv"
            )
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
            annotation=f"confidence",
            spot_size=50,
            ax=axs[1],
            title="Confidence map",
        )
        plt.savefig(
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.png"
            ),
            dpi=120,
            bbox_inches="tight",
        )

    end = time.time()
    logger.info(f"Total execution time: {(end - start):.2f}s")


if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")

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
        "--dist_prob_weight",
        help="Weight coefficient for probabilities obtained by distance metric. Weight for contrastive is 1.0 - dist_prob_weight.",
        type=float,
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--batch_size",
        help="Contrastive: Number of samples in the batch. Defaults to 512",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--epochs",
        help="Contrastive: Number of epochs to train deep encoder. Defaults to 50",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--emb_dim",
        help="Contrastive: Dimension of the output embeddings. Defaults to 32.",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--enc_depth",
        help="Contrastive: Number of layers in the encoder MLP. Defaults to 4.",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--class_depth",
        help="Contrastive: Number of layers in the classifier MLP. Defaults to 2.",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--augmentation_perc",
        help="Contrastive: Percentage for the augmentation of SC data. Defaults to 0.7.",
        type=float,
        required=False,
        default=0.7,
    )
    parser.add_argument(
        "--n_jobs",
        help="Number of jobs to run in parallel. -1 means using all available processors",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument("-c", "--contrastive", action="store_true")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable logging by specifying --verbose",
        action="store_const",
        const=logging.INFO,
        default=logging.WARNING,
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=args.verbose,
    )
    logger = logging.getLogger(__name__)

    filename = None
    if args.verbose != logging.WARNING:
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        filename = os.path.basename(args.st_path).replace(".h5ad", "")
        filename = f"logs/{filename}_{timestamp}.log"
        file_handler = logging.FileHandler(filename)
        logger.addHandler(file_handler)
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

    main(args=args)

    # End the background process logging the CPU and GPU utilisation.
    logger_pid.terminate()
    print("Terminated the compute utilisation logger background process")

    # read cpu and gpu memory utilization
    logger_df = pd.read_csv(mem_logger_fname)

    max_cpu_mem = logger_df.loc[:, "RAM"].max()
    max_gpu_mem = logger_df.loc[:, "GPU 0"].max()

    logger.info(
        f"Peak RAM Usage: {max_cpu_mem} MiB\nPeak GPU Usage: {max_gpu_mem} MiB\n"
    )
