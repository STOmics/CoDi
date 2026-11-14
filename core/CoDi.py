import argparse as ap
import datetime
import logging
import multiprocessing as mp
import os
import random
import subprocess
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse
from scipy.spatial.distance import mahalanobis
from scipy.special import kl_div, rel_entr
from scipy.stats import wasserstein_distance

random.seed(3)
warnings.filterwarnings("ignore", message=".*Some cells have zero counts.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log1p*.")
warnings.filterwarnings("ignore", message=".*is_categorical_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*Variable names are not unique.*")

DISTANCE_METRICS = {
    "mahalanobis",
    "KLD",
    "wasserstein",
    "relativeEntropy",
    "hellinger",
    "binary",
    "none",
}
NUM_SUBSETS = 10
MAX_DISTANCE = 9999999  # A cell type matches only if its distance is below this
POOL_CHUNKSIZE = 30


# ---------------------------------------------------------------------------
# Distance-based annotation
# ---------------------------------------------------------------------------
def create_subsets(gene_set, num_of_subsets=NUM_SUBSETS):
    """Draw random gene subsets, each covering 30%-80% of `gene_set`."""
    num_of_elem = len(gene_set)
    subsets = []
    for _ in range(num_of_subsets):
        subset_size = random.randint(int(num_of_elem * 0.3), int(num_of_elem * 0.8))
        subsets.append(random.sample(gene_set, subset_size))
    return subsets


def hellinger(p, q):
    """Hellinger distance between distributions (row-wise if `q` is 2D)."""
    return np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=-1) / np.sqrt(2)


def binary_distance(p, q):
    """Binary distance between distributions (row-wise if `q` is 2D).

    Counts positional pairs of different status: one zero, the other nonzero."""
    return np.sum(p.astype(bool) ^ q.astype(bool), axis=-1)


def normalize_distrib(distrib):
    """Normalize to sum 1.0 if sum is not 0."""
    total = np.sum(distrib, axis=0, keepdims=True)
    return distrib / total if total != 0 else distrib


def distances_to_cell_types(distance, st_distrib, st_distrib_norm, ref):
    """Distances from one ST cell to the mean profile of every cell type.

    Args:
        distance: Name of the distance metric.
        st_distrib: Raw expression of the cell on the subset genes, shape (k,).
        st_distrib_norm: `st_distrib` normalized to sum 1.0, shape (k,).
        ref: Subset reference from `build_references`.

    Returns:
        One distance per cell type.
    """
    if distance == "KLD":
        return kl_div(st_distrib_norm, ref["norm"]).sum(axis=1)
    if distance == "relativeEntropy":
        return rel_entr(st_distrib_norm, ref["norm"]).sum(axis=1)
    if distance == "hellinger":
        return hellinger(st_distrib_norm, ref["norm"])
    if distance == "binary":
        return binary_distance(np.floor(st_distrib), np.floor(ref["raw"]))
    if distance == "mahalanobis":
        return [
            mahalanobis(st_distrib_norm, sc_norm, icm)
            for sc_norm, icm in zip(ref["norm"], ref["icms"])
        ]
    if distance == "wasserstein":
        return [wasserstein_distance(st_distrib_norm, sc_norm) for sc_norm in ref["norm"]]
    raise ValueError(f"Unknown distance metric: {distance}")


def closest_cell_type(cell_types, dists):
    """First cell type with the smallest distance below MAX_DISTANCE, else ""."""
    best_type, best_dist = "", MAX_DISTANCE
    for cell_type, dist in zip(cell_types, dists):
        if dist < best_dist:
            best_type, best_dist = cell_type, dist
    return best_type


# Read-only state shared with pool workers once, via the pool initializer
_worker_state = {}


def init_worker(state):
    global _worker_state
    _worker_state = state


def per_cell(ii):
    """Annotate ST cell `ii` by majority vote over the gene subsets.

    Returns:
        (cell type, confidence, per cell type vote share)
    """
    cell_types = _worker_state["cell_types"]
    st_row = _worker_state["st"][ii]

    votes = []
    for subset_idx, ref in zip(_worker_state["subset_idx"], _worker_state["refs"]):
        st_distrib = st_row[subset_idx].astype(float)
        dists = distances_to_cell_types(
            _worker_state["distance"], st_distrib, normalize_distrib(st_distrib), ref
        )
        votes.append(closest_cell_type(cell_types, dists))

    num_votes = len(votes)
    counts = Counter(votes)
    winner, winner_count = counts.most_common(1)[0]
    vote_shares = [
        np.round(counts[ct] / num_votes, 3) if ct in counts else 0.0 for ct in cell_types
    ]
    return winner, np.round(winner_count / num_votes, 3), vote_shares


def marker_expression(adata, markers):
    """Dense DataFrame of `adata.X` restricted to `markers` (in that order)."""
    select_ind = adata.var.index.get_indexer(markers)
    X = (
        adata.X.tocsr()[:, select_ind].todense()
        if issparse(adata.X)
        else adata.X[:, select_ind]
    )
    return pd.DataFrame(X, columns=markers, index=adata.obs.index)


def build_references(sc_df, sc_labels, cell_types, subsets, distance):
    """Per gene subset: mean profile of every cell type, raw and normalized.

    Each reference holds (n_cell_types, subset_size) matrices "raw" and "norm",
    plus inverse covariance matrices "icms" for the Mahalanobis distance.
    """
    sc_mean = {}
    sc_icms = {}
    for ty in cell_types:
        ty_df = sc_df[sc_labels == ty]
        sc_mean[ty] = []
        sc_icms[ty] = []
        for subset in subsets:
            subset_df = ty_df[subset]
            if distance == "mahalanobis":
                # Pseudo inverse avoids singular matrices caused by genes
                # with all zero values per cell type
                cov = np.cov(subset_df.values, rowvar=False)
                sc_icms[ty].append(np.linalg.pinv(cov))
            sc_mean[ty].append(subset_df.mean().values.astype(float))

    refs = []
    for sub_id in range(len(subsets)):
        raw = [sc_mean[ty][sub_id] for ty in cell_types]
        refs.append(
            {
                "raw": np.vstack(raw),
                "norm": np.vstack([normalize_distrib(r) for r in raw]),
                "icms": [sc_icms[ty][sub_id] for ty in cell_types]
                if distance == "mahalanobis"
                else None,
            }
        )
    return refs


def run_distance(args, adata_sc, adata_st, markers, logger):
    """Distance-based annotation of all ST cells; results go into `adata_st`."""
    sc_df = marker_expression(adata_sc, markers)
    st_df = marker_expression(adata_st, markers)
    cell_types = list(sorted(adata_sc.obs[args.annotation].unique()))
    adata_st.obsm["probabilities_dist"] = pd.DataFrame(
        index=adata_st.obs.index, columns=cell_types
    ).astype("float32")

    subsets = create_subsets(markers, num_of_subsets=NUM_SUBSETS)
    worker_state = {
        "distance": args.distance,
        "cell_types": cell_types,
        "st": st_df.to_numpy(),
        "subset_idx": [st_df.columns.get_indexer(subset) for subset in subsets],
        "refs": build_references(
            sc_df, adata_sc.obs[args.annotation], cell_types, subsets, args.distance
        ),
    }

    num_cpus_used = mp.cpu_count() if args.n_jobs == -1 else args.n_jobs
    logger.info("Starting parallel per cell calculation of distances.")
    with mp.Pool(
        processes=num_cpus_used, initializer=init_worker, initargs=(worker_state,)
    ) as pool:
        results = pool.map(per_cell, range(len(st_df)), chunksize=POOL_CHUNKSIZE)

    labels, confidences, vote_shares = zip(*results)
    adata_st.obs["CoDi_dist"] = list(labels)
    adata_st.obs["CoDi_confidence_dist"] = list(confidences)
    adata_st.obsm["probabilities_dist"].iloc[:, :] = list(vote_shares)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def load_data(sc_path, st_path):
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)
    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()
    adata_sc.obs_names_make_unique()
    adata_st.obs_names_make_unique()

    # Spatial coordinates are expected in 'spatial', 'X_spatial' or 'spatial_stereoseq'
    if "X_spatial" in adata_st.obsm:
        adata_st.obsm["spatial"] = adata_st.obsm["X_spatial"].copy()
    elif "spatial_stereoseq" in adata_st.obsm:
        adata_st.obsm["spatial"] = np.array(adata_st.obsm["spatial_stereoseq"].copy())
    elif "spatial" not in adata_st.obsm:
        warnings.warn(
            'Spatial coordinates not found. Labels expected in: \
                .obsm["spatial"] or\n \
                .obsm["X_spatial"] or\n \
                .obsm["spatial_stereoseq"]'
        )
    return adata_sc, adata_st


def compute_marker_genes(adata_sc, adata_st, annotation, num_markers, logger):
    """Top `num_markers` marker genes per cell type that also exist in ST."""
    start = time.time()
    adata_sc.layers["counts"] = adata_sc.X.copy()  # Used in contrastive learning
    if "rank_genes_groups" not in adata_sc.uns:
        if adata_sc.X.min() >= 0:  # If already logarithmized skip
            sc.pp.normalize_total(adata_sc, target_sum=1e4)
            sc.pp.log1p(adata_sc)
        sc.tl.rank_genes_groups(
            adata_sc, groupby=annotation, use_raw=False, method="t-test"
        )
    else:
        logger.info("***d Using precalculated marker genes in input h5ad.")

    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[
        0:num_markers, :
    ]
    markers = list(np.unique(markers_df.melt().value.values))
    markers_intersect = list(set(markers).intersection(adata_st.var.index))
    logger.info(
        f"Using {len(markers_intersect)} unique single cell marker genes that exist in ST dataset ({num_markers} per cell type)"
    )
    logger.info(f"Calculation of marker genes took {np.round(time.time() - start, 3):.2f}")
    return markers_intersect


def start_contrastive(args, adata_sc, adata_st, markers, filename, logger):
    """Train and predict with the contrastive model in a separate process."""
    from contrastive import contrastive_process

    queue = mp.Queue()
    proc = mp.Process(
        target=contrastive_process,
        kwargs=dict(
            sc_path=args.sc_path,
            st_path=args.st_path,
            adata_sc=adata_sc[:, markers],
            adata_st=adata_st[:, markers],
            annotation_sc=args.annotation,
            batch_size=args.batch_size,
            epochs=args.epochs,
            embedding_dim=args.emb_dim,
            encoder_depth=args.enc_depth,
            classifier_depth=args.class_depth,
            filename=filename,
            augmentation_perc=args.augmentation_perc,
            logger=logger,
            log_embeddings=args.log_embeddings,
            queue=queue,
        ),
        name="Contrastive process",
    )
    proc.start()
    return proc, queue


def collect_contrastive(adata_st, proc, queue, with_confidence=True):
    """Store contrastive results (FIFO order is defined in contrastive.py)."""
    adata_st.obsm["probabilities_contrastive"] = queue.get()
    adata_st.obs["CoDi_contrastive"] = queue.get()
    if with_confidence:
        adata_st.obs["CoDi_confidence_contrastive"] = np.round(
            np.nanmax(adata_st.obsm["probabilities_contrastive"].to_numpy(), axis=1), 3
        )
    proc.join()


def combine_predictions(adata_st, dist_prob_weight, use_contrastive):
    """Final CoDi label: weighted sum of distance and contrastive probabilities."""
    if not use_contrastive:
        adata_st.obs["CoDi"] = adata_st.obs["CoDi_dist"]
        adata_st.obs["CoDi_confidence"] = adata_st.obs["CoDi_confidence_dist"]
        return

    assert (
        "probabilities_contrastive" in adata_st.obsm
    ), "Missing 'probabilities_contrastive' in adata_st.obsm."
    probabilities = (
        adata_st.obsm["probabilities_contrastive"] * (1.0 - dist_prob_weight)
    ).add(adata_st.obsm["probabilities_dist"] * dist_prob_weight)
    adata_st.obsm["probabilities"] = probabilities

    prob_values = probabilities.to_numpy()
    adata_st.obs["CoDi"] = np.array(
        probabilities.columns[np.nanargmax(prob_values, axis=1)]
    ).astype("str")
    adata_st.obs["CoDi_confidence"] = np.round(np.nanmax(prob_values, axis=1), 3)


def output_path(args, extension):
    """<out_path>/<ST file name>_CoDi_<distance>.<extension>"""
    return os.path.join(
        args.out_path,
        os.path.basename(args.st_path).replace(
            ".h5ad", f"_CoDi_{args.distance}.{extension}"
        ),
    )


def save_results(adata_st, args, csv_columns):
    adata_st.obs.index.name = "cell_id"
    adata_st.obs[csv_columns].to_csv(output_path(args, "csv"))
    adata_st.write_h5ad(output_path(args, "h5ad"))


def plot_spatial(
    adata, annotation, ax: plt.Axes, spot_size: float, palette=None, title: str = ""
):
    """
    Scatter plot in spatial coordinates.

    Parameters:
        - adata (AnnData): Annotated data object which represents the sample
        - annotation (str): adata.obs column used for grouping
        - ax (Axes): Axes object used for plotting
        - spot_size (int): Diameter of the dot that represents a cell (the plotting
                library uses radius, therefore it is multiplied by 0.5)
        - palette: Unused; float annotations always use the "coolwarm" colormap
        - title (str): Title of the figure
    """
    is_float = "float" in str(type(adata.obs[annotation].iloc[0])).lower()
    ax = sns.scatterplot(
        data=adata.obs,
        hue=annotation,
        x=adata.obsm["spatial"][:, 0],
        y=adata.obsm["spatial"][:, 1],
        ax=ax,
        s=spot_size * 0.5,
        linewidth=0,
        palette=sns.color_palette("coolwarm", as_cmap=True) if is_float else None,
        marker=".",
    )
    ax.invert_yaxis()
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)


def plot_results(adata_st, args):
    fig, axs = plt.subplots(1, 2, figsize=(14, 14))
    plot_spatial(adata_st, annotation="CoDi", spot_size=50, ax=axs[0], title="Cell types")
    plot_spatial(
        adata_st,
        annotation="CoDi_confidence",
        spot_size=50,
        ax=axs[1],
        title="Confidence map",
    )
    plt.savefig(output_path(args, "png"), dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main_proc(args, logger, filename):
    start = time.time()
    use_contrastive = not args.no_contrastive

    adata_sc, adata_st = load_data(args.sc_path, args.st_path)
    markers = compute_marker_genes(
        adata_sc, adata_st, args.annotation, args.num_markers, logger
    )

    if use_contrastive:
        contrastive_proc, queue = start_contrastive(
            args, adata_sc, adata_st, markers, filename, logger
        )

    if args.distance == "none":
        if use_contrastive:
            collect_contrastive(adata_st, contrastive_proc, queue, with_confidence=False)
            save_results(adata_st, args, "CoDi_contrastive")
        logger.info("No distance metric specified, exiting...")
        logger.info(f"Total execution time: {(time.time() - start):.2f}s")
        return

    run_distance(args, adata_sc, adata_st, markers, logger)
    if use_contrastive:
        collect_contrastive(adata_st, contrastive_proc, queue)
    combine_predictions(adata_st, args.dist_prob_weight, use_contrastive)
    logger.info(f"CoDi execution took: {time.time() - start}s")

    if use_contrastive:
        csv_columns = [
            "CoDi",
            "CoDi_confidence",
            "CoDi_dist",
            "CoDi_confidence_dist",
            "CoDi_contrastive",
            "CoDi_confidence_contrastive",
        ]
    else:
        csv_columns = ["CoDi_dist", "CoDi_confidence_dist", "CoDi", "CoDi_confidence"]
    save_results(adata_st, args, csv_columns)

    if "spatial" in adata_st.obsm_keys():
        plot_results(adata_st, args)

    logger.info(f"Total execution time: {(time.time() - start):.2f}s")


def parse_args():
    parser = ap.ArgumentParser(
        description="A script that performs CoDi - Contrastive-Distance reference based cell type annotation."
    )
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
        default="KLD",
        choices=DISTANCE_METRICS,
    )
    parser.add_argument(
        "--num_markers",
        help="Number of marker genes per cell type.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--dist_prob_weight",
        help="Weight coefficient for probabilities obtained by distance metric. \
            Weight for contrastive is 1.0 - dist_prob_weight. Default is 0.5.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--batch_size",
        help="Contrastive: Number of samples in the batch. Defaults to 512.",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--epochs",
        help="Contrastive: Number of epochs to train deep encoder. Default is 50.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--emb_dim",
        help="Contrastive: Dimension of the output embeddings. Default is 32.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--enc_depth",
        help="Contrastive: Number of layers in the encoder MLP. Default is 4.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--class_depth",
        help="Contrastive: Number of layers in the classifier MLP. Default is 2.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--augmentation_perc",
        help="Contrastive: Percentage for the augmentation of scRNA reference data. \
            If not provided it will be calculated automatically. Defaults to None.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--n_jobs",
        help="Number of jobs to run in parallel. -1 means using all available processors.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--no_contrastive",
        action="store_true",
        default=False,
        help="Turn off contrastive prediction of cell types.",
    )
    parser.add_argument(
        "--log_embeddings",
        action="store_true",
        default=False,
        help="Save embeddings to disk as np array and create UMAP plots of ST and SC embeddings.",
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
        default="",
    )
    return parser.parse_args()


def setup_logging(args):
    """Configure logging. Returns (logger, log file path or None)."""
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=args.verbose,
    )
    logger = logging.getLogger(__name__)

    filename = None
    if args.verbose != logging.WARNING:
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        st_name = os.path.basename(args.st_path).replace(".h5ad", "")
        filename = os.path.join(args.out_path, f"logs/{st_name}_{timestamp}.log")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.addHandler(logging.FileHandler(filename))
    return logger, filename


def start_resource_logger(args, logger):
    """Start the background CPU/GPU memory logger. Returns (process, CSV path)."""
    mem_logger_fname = os.path.join(
        args.out_path,
        os.path.basename(args.st_path).replace(".h5ad", "_cpu_gpu_memlog.csv"),
    )
    if os.path.isfile(mem_logger_fname):
        os.remove(mem_logger_fname)
    proc = subprocess.Popen(["python", "scripts/log_gpu_cpu_stats.py", mem_logger_fname])
    logger.info("Started logging compute resource utilisation")
    return proc, mem_logger_fname


def stop_resource_logger(proc, mem_logger_fname, logger):
    proc.terminate()
    print("Terminated the compute utilisation logger background process")
    logger_df = pd.read_csv(mem_logger_fname)
    max_cpu_mem = logger_df.loc[:, "RAM"].max()
    max_gpu_mem = logger_df.loc[:, "GPU 0"].max()
    logger.info(
        f"Peak RAM Usage: {max_cpu_mem} MiB\nPeak GPU Usage: {max_gpu_mem} MiB\n"
    )


def main(args=None):
    os.makedirs("logs", exist_ok=True)
    args = parse_args()
    logger, filename = setup_logging(args)

    if args.log_mem:
        mem_proc, mem_logger_fname = start_resource_logger(args, logger)

    main_proc(args=args, logger=logger, filename=filename)

    if args.log_mem:
        stop_resource_logger(mem_proc, mem_logger_fname, logger)


if __name__ == "__main__":
    main()
