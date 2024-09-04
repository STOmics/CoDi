import datetime
import logging
import sys
import random
import time
import os
import warnings
import subprocess
import multiprocessing as mp
import argparse as ap
from collections import Counter
from itertools import repeat
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from memory_profiler import memory_usage
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
from scipy.special import rel_entr, kl_div
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse

random.seed(3)
# Suppress the warning
warnings.filterwarnings("ignore", message=".*Some cells have zero counts.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log1p*.")
warnings.filterwarnings("ignore", message=".*is_categorical_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*Variable names are not unique.*")


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


def per_cell(ii, subsets, cell_types, st_df, sc_mean, sc_icms, args):
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


def main_proc(args, logger, filename):
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

    # Calculate marker genes  TODO: Add to separate function in preprocessing.py
    start_marker = time.time()
    adata_sc.layers["counts"] = adata_sc.X.copy()  # Used in contrastive learning
    if "rank_genes_groups" not in adata_sc.uns:
        if adata_sc.X.min() >= 0:  # If already logaritmized skip
            sc.pp.normalize_total(adata_sc, target_sum=1e4)
            sc.pp.log1p(adata_sc)
        sc.tl.rank_genes_groups(adata_sc, groupby=args.annotation, use_raw=False, method='t-test')
    else:
        logger.info(f"***d Using precalculated marker genes in input h5ad.")

    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[
        0 : args.num_markers, :
    ]

    markers = list(np.unique(markers_df.melt().value.values))

    markers_intersect = list(set(markers).intersection(adata_st.var.index))
    logger.info(
        f"Using {len(markers_intersect)} unique single cell marker genes that exist in ST dataset ({args.num_markers} per cell type)"
    )
    end_marker = time.time()
    marker_time = np.round(end_marker - start_marker, 3)
    logger.info(f"Calculation of marker genes took {marker_time:.2f}")

    # Contrastive part
    if not args.no_contrastive:
        from contrastive import contrastive_process

        queue = mp.Queue()

        contrastive_proc = mp.Process(
            target=contrastive_process,
            kwargs=dict(
                sc_path=args.sc_path,
                st_path=args.st_path,
                adata_sc=adata_sc[:, markers_intersect],
                adata_st=adata_st[:, markers_intersect],
                annotation_sc=args.annotation,
                batch_size=args.batch_size,
                epochs=args.epochs,
                embedding_dim=args.emb_dim,
                encoder_depth=args.enc_depth,
                classifier_depth=args.class_depth,
                filename=filename,
                augmentation_perc=args.augmentation_perc,
                logger=logger,
                queue=queue,
            ),
            name="Contrastive process",
        )
        contrastive_proc.start()

    if args.distance == "none":
        if not args.no_contrastive:
            df_probabilities = queue.get()  # FIFO (ordering in contrastive.py)
            adata_st.obsm["probabilities_contrastive"] = df_probabilities
            predictions = queue.get()
            adata_st.obs["CoDi_contrastive"] = predictions
            contrastive_proc.join()
            # Write CSV and H5AD  TODO: Add to separate function in core/util.py
            adata_st.obs.index.name = "cell_id"
            adata_st.obs["CoDi_contrastive"].to_csv(os.path.join(args.out_path,
                os.path.basename(args.st_path).replace(
                    ".h5ad", f"_CoDi_{args.distance}.csv"
                )
            ))
            adata_st.write_h5ad(os.path.join(args.out_path,
                os.path.basename(args.st_path).replace(
                    ".h5ad", f"_CoDi_{args.distance}.h5ad"
                )
            ))
        logger.info(f"No distance metric specified, exiting...")
        end = time.time()
        logger.info(f"Total execution time: {(end - start):.2f}s")
        return # TODO: Remove this and unite with same code for saving results

    # Extract gene expressions only from marker genes  TODO: add to separate function in core/preprocessing.py
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

    # Distance based algorithm TODO: Add to separate function in core/distance.py
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
                iis,
                repeat(subsets),
                repeat(cell_types),
                repeat(st_df),
                repeat(sc_mean),
                repeat(sc_icms),
                repeat(args)
            ),
            chunksize=30,
        )

    assigned_types.sort(key=lambda x: x[0])
    assigned_types = [at[1] for at in assigned_types]
    adata_st.obs["CoDi_dist"] = [x["cell_type"] for x in assigned_types]
    adata_st.obs["CoDi_confidence_dist"] = [x["confidence"] for x in assigned_types]
    adata_st.obsm["probabilities_dist"].iloc[:, :] = [
        x["ct_probabilities"] for x in assigned_types
    ]

    # sns.histplot([x["confidence"] for x in assigned_types])
    # plt.savefig(f"CoDi_confidence_hist__{args.distance}.png", dpi=120, bbox_inches="tight")

    if not args.no_contrastive:
        df_probabilities = queue.get()  # FIFO (ordering in contrastive.py)
        adata_st.obsm["probabilities_contrastive"] = df_probabilities
        predictions = queue.get()
        adata_st.obs["CoDi_contrastive"] = predictions
        adata_st.obs["CoDi_confidence_contrastive"] = [
            np.round(prow.max(), 3)
            for _, prow in adata_st.obsm["probabilities_contrastive"].iterrows()
        ]
        contrastive_proc.join()

    # combine contrastive and distance results TODO: Add to separate function in core/util.py
    if not args.no_contrastive:
        assert (
            "probabilities_contrastive" in adata_st.obsm
        ), "Missing 'probabilities_contrastive' in adata_st.obsm."
        adata_st.obsm["probabilities"] = (
            adata_st.obsm["probabilities_contrastive"] * (1.0 - args.dist_prob_weight)
        ).add(adata_st.obsm["probabilities_dist"] * args.dist_prob_weight)
        adata_st.obs["CoDi"] = np.array(
            [prow.idxmax() for _, prow in adata_st.obsm["probabilities"].iterrows()]
        ).astype("str")
        adata_st.obs["CoDi_confidence"] = [
            np.round(prow.max(), 3)
            for _, prow in adata_st.obsm["probabilities"].iterrows()
        ]
    else:
        adata_st.obs["CoDi"] = adata_st.obs["CoDi_dist"]
        adata_st.obs["CoDi_confidence"] = adata_st.obs["CoDi_confidence_dist"]

    end = time.time()
    logger.info(f"CoDi execution took: {end - start}s")

    # Write CSV and H5AD
    # TODO: Add to separate function in core/util.py that will work with or without contrastive or distance
    adata_st.obs.index.name = "cell_id"
    # Write CSV and H5AD of final combined results
    if not args.no_contrastive:
        adata_st.obs[
            [
                "CoDi",
                "CoDi_confidence",
                "CoDi_dist",
                "CoDi_confidence_dist",
                "CoDi_contrastive",
                "CoDi_confidence_contrastive"
            ]
        ].to_csv(os.path.join(args.out_path,
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.csv"
            ))
        )
    else:
        adata_st.obs[["CoDi_dist", "CoDi_confidence_dist", "CoDi", "CoDi_confidence"]].to_csv(
            os.path.join(args.out_path, os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.csv"
            ))
        )
    adata_st.write_h5ad(os.path.join(args.out_path,
        os.path.basename(args.st_path).replace(".h5ad", f"_CoDi_{args.distance}.h5ad")
    ))

    if "spatial" in adata_st.obsm_keys():  # TODO: Add to separate function in core/util.py
        fig, axs = plt.subplots(1, 2, figsize=(14, 14))
        plot_spatial(
            adata_st, annotation=f"CoDi", spot_size=50, ax=axs[0], title="Cell types"
        )
        plot_spatial(
            adata_st,
            annotation=f"CoDi_confidence",
            spot_size=50,
            ax=axs[1],
            title="Confidence map",
        )
        plt.savefig(os.path.join(args.out_path,
            os.path.basename(args.st_path).replace(
                ".h5ad", f"_CoDi_{args.distance}.png"
        )),
            dpi=120,
            bbox_inches="tight",
        )

    end = time.time()
    logger.info(f"Total execution time: {(end - start):.2f}s")


def main(args=None):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    parser = ap.ArgumentParser(
        description="A script that performs CoDi - Contrastive-Distance reference based cell type annotation.")
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
        required=True,
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
        help="Number of marker genes per cell type.",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--dist_prob_weight",
        help="Weight coefficient for probabilities obtained by distance metric. \
            Weight for contrastive is 1.0 - dist_prob_weight. Default is 0.5.",
        type=float,
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--batch_size",
        help="Contrastive: Number of samples in the batch. Defaults to 512.",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--epochs",
        help="Contrastive: Number of epochs to train deep encoder. Default is 50.",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--emb_dim",
        help="Contrastive: Dimension of the output embeddings. Default is 32.",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--enc_depth",
        help="Contrastive: Number of layers in the encoder MLP. Default is 4.",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--class_depth",
        help="Contrastive: Number of layers in the classifier MLP. Default is 2.",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--augmentation_perc",
        help="Contrastive: Percentage for the augmentation of scRNA reference data. \
            If not provided it will be calculated automatically. Defaults to None.",
        type=float,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--n_jobs",
        help="Number of jobs to run in parallel. -1 means using all available processors.",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument("--no_contrastive", 
        action="store_true",
        default=False,
        help="Turn off contrastive prediction of cell types."
    )
    parser.add_argument("-l", "--log_mem", action="store_true", default=False)

    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable logging by specifying --verbose",
        action="store_const",
        const=logging.INFO,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--out_path",
        help="Output path to store results.",
        type=str,
        required=False,
        default='',
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
        filename = os.path.join(args.out_path, f"logs/{filename}_{timestamp}.log")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        file_handler = logging.FileHandler(filename)
        logger.addHandler(file_handler)
    if args.log_mem:
        mem_logger_fname = os.path.join(args.out_path, os.path.basename(args.st_path).replace(".h5ad", "_cpu_gpu_memlog.csv"))
        if os.path.isfile(mem_logger_fname):
            os.remove(mem_logger_fname)

        logger_pid = subprocess.Popen(
            [
                "python",
                "scripts/log_gpu_cpu_stats.py",
                mem_logger_fname,
            ]
        )
        logger.info("Started logging compute resource utilisation")

    main_proc(args=args, logger=logger, filename=filename)

    if args.log_mem:
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


if __name__ == "__main__":
    main()