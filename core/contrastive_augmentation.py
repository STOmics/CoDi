import logging
import random

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def auto_augmentation_perc_estimation(adata_sc: ad.AnnData, adata_st: ad.AnnData):
    """Estimate the augmentation (dropout) percentage.

    Uses the relative difference between the mean number of expressed genes
    per cell in SC and in ST data.

    Args:
        adata_sc: SC reference data
        adata_st: ST data
    """
    st_exp_mean = np.mean((adata_st.X != 0).sum(axis=1))
    sc_exp_mean = np.mean((adata_sc.X != 0).sum(axis=1))
    perc = np.round((sc_exp_mean - st_exp_mean) / sc_exp_mean, 2)
    return perc if perc > 0 else 0


def normalize(max_count, min_count, new_max, new_min, num):
    """Map a cell type count from [min_count, max_count] to [new_min, new_max].

    Args:
        max_count: Max number of some celltype in original dataset
        min_count: Min number of some celltype in original dataset
        new_max: Max number of some celltype in augmented dataset
        new_min: Min number of some celltype in augmented dataset
        num: Number of cells of certain celltype in original dataset

    Returns:
        Corrected number of cells of certain celltype in augmented dataset
    """
    norm = (num - min_count) / (max_count - min_count)
    return int(norm * (new_max - new_min) + new_min)


def resampled_class_sizes(labels: pd.Series):
    """New size of every cell type, scaled to 0.25-0.75 of the largest class.

    Returns:
        List of (new size, original size, cell type), largest class first.
    """
    value_counts = labels.value_counts()
    counts_per_ct = value_counts.values
    cts = list(value_counts.index)
    max_count, min_count = max(counts_per_ct), min(counts_per_ct)
    upper_bound, lower_bound = int(max_count * 0.75), int(max_count * 0.25)

    normalized_counts_per_ct = [
        normalize(max_count, min_count, upper_bound, lower_bound, count)
        for count in counts_per_ct
    ]
    # Rescale so the total number of cells stays roughly the same
    scaling_factor = sum(counts_per_ct) / sum(normalized_counts_per_ct)
    return [
        (int(new_size * scaling_factor), original_size, label)
        for new_size, original_size, label in zip(
            normalized_counts_per_ct, counts_per_ct, cts
        )
    ]


def drop_out_counts(row: np.ndarray, percentage: float):
    """Set a random `percentage` of the non-zero counts in `row` to zero (in place)."""
    non_zero_indices = np.nonzero(row)[0]
    num_elements_to_zero = int(percentage * len(non_zero_indices))
    selected_indices = random.sample(list(non_zero_indices), k=num_elements_to_zero)
    row[selected_indices] = 0
    return row


def augment_data(
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    annotation: str,
    percentage: float = None,
    logger: logging.Logger = None,
):
    """Rebalance cell types to 0.25-0.75 of the largest class and augment.

    Cell types are subsampled (without replacement) or oversampled (with
    replacement) to their new size. Augmentation sets a percentage of each
    cell's non-zero gene counts to zero to better resemble ST data.

    Args:
        adata_sc: AnnData object with SC gene exp and annotation
        adata_st: AnnData object with ST gene exp used in case of auto percentage calculation
        annotation: Column in adata.obs that represents cell type annotation
        percentage: A percentage of genes with non-zero counts that will be set to zero
    """
    if percentage is None:
        percentage = auto_augmentation_perc_estimation(adata_sc, adata_st)
        logger.info(f"Auto calculated augmentation percentage is {percentage}\n")

    class_sizes = resampled_class_sizes(adata_sc.obs[annotation])
    total_new_cells = sum(new_size for new_size, _, _ in class_sizes)
    counts = np.zeros((total_new_cells, adata_sc.shape[1]), dtype=np.int16)
    labels = []

    X_sc = csr_matrix(adata_sc.X)
    obs_names = adata_sc.obs_names
    ct_labels = adata_sc.obs[annotation].values

    ind = 0
    for new_size, original_size, label in class_sizes:
        ct_obs_names = obs_names[ct_labels == label]
        if new_size <= original_size:
            selected_cells = random.sample(list(ct_obs_names), k=new_size)
        else:
            selected_cells = random.choices(ct_obs_names, k=new_size)

        for row in obs_names.get_indexer(selected_cells):
            counts[ind] = drop_out_counts(
                X_sc[row].toarray().astype(np.int16)[0], percentage
            )
            labels.append(label)
            ind += 1

    new_adata = ad.AnnData(counts)
    new_adata.var_names = adata_sc.var_names
    new_adata.obs[annotation] = pd.Series(
        labels, index=new_adata.obs_names, dtype=object
    ).astype("category")
    new_adata.X = csr_matrix(new_adata.X)
    return new_adata
