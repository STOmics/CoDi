import anndata as ad
from scipy.sparse import csr_matrix
import numpy as np
import random
import pandas as pd


def normalize(max_count, min_count, new_max, new_min, num):
    """Normalization to new count values

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


def augment_data(adata_sc: ad.AnnData, annotation: str, percentage: float):
    """Scale original gene expression abundance to fit 0.25-0.75 of maximum celltype count and augment

    Augmentation is performed by setting a percentage of non-zero gene counts to zero to
    better resemble ST data.

    Args:
        adata: AnnData object that contains SC gene expression and annotation
        annotation: Column in adata.obs that represents cell type annotation
        percentage: A percentage of genes with non-zero counts that will be set to zero
    """
    counts_per_ct = adata_sc.obs[annotation].value_counts().values
    cts = list(adata_sc.obs[annotation].value_counts().index)
    max_count, min_count = max(counts_per_ct), min(counts_per_ct)
    upper_bound, lower_bound = int(max_count * 0.75), int(max_count * 0.25)

    normalized_counts_per_ct = [
        normalize(max_count, min_count, upper_bound, lower_bound, el)
        for el in counts_per_ct
    ]
    resampling_class_size = list(
        zip(normalized_counts_per_ct, counts_per_ct, cts)
    )  # (new, original, cell_type)

    total_new_cells = sum(normalized_counts_per_ct)
    genes = adata_sc.shape[1]
    counts = np.zeros((total_new_cells, genes), dtype=np.int16)

    new_adata = ad.AnnData(counts)
    new_adata.var_names = adata_sc.var_names
    new_adata.obs[annotation] = pd.Series(np.empty(total_new_cells), dtype=str)

    ind = 0
    genes = adata_sc.shape[1]

    for new_size, original_size, label in resampling_class_size:
        if new_size <= original_size:
            # take unique cells and mutate
            adata_single_ct = adata_sc[adata_sc.obs[annotation] == label, :]
            for obs_name in random.sample(list(adata_single_ct.obs_names), k=new_size):
                new_adata.obs[annotation].iloc[ind] = label
                to_mutate = adata_sc[obs_name].X.toarray().astype(np.int16)[0]
                non_zero_indices = np.nonzero(to_mutate)
                num_elements_to_zero = int(percentage * len(non_zero_indices[0]))
                selected_indices = random.sample(
                    list(non_zero_indices[0]), k=num_elements_to_zero
                )
                to_mutate[selected_indices] = 0
                new_adata[ind].X = to_mutate
                ind += 1
        else:
            # you have to oversample by duplicating, draw random samples with replacement
            adata_single_ct = adata_sc[adata_sc.obs[annotation] == label, :]
            for obs_name in random.choices(adata_single_ct.obs_names, k=new_size):
                # mutate here
                new_adata.obs[annotation].iloc[ind] = label
                to_mutate = adata_sc[obs_name].X.toarray().astype(np.int16)[0]
                non_zero_indices = np.nonzero(to_mutate)
                num_elements_to_zero = int(percentage * len(non_zero_indices[0]))
                selected_indices = random.sample(
                    list(non_zero_indices[0]), k=num_elements_to_zero
                )
                to_mutate[selected_indices] = 0
                new_adata[ind].X = to_mutate
                ind += 1

    new_adata.obs[annotation] = new_adata.obs[annotation].astype("category")
    new_adata.X = csr_matrix(new_adata.X)

    # perform some basic testing to confirm that number of
    # genes with non-zero counts decreased in augmented dataset
    # cnt_new = 0
    # cnt_old = 0
    # for i in range(adata_sc.shape[0]):
    #     cnt_old += len(np.nonzero(adata_sc[i].X)[1])
    # for i in range(new_adata.shape[0]):
    #     cnt_new += len(np.nonzero(new_adata[i].X)[1])

    # print("avg non-zero count in original", cnt_old / adata_sc.shape[0])
    # print("avg non-zero count in augmented", cnt_new / new_adata.shape[0])

    return new_adata
