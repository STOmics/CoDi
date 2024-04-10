import os
import subprocess
import json
import sys
import time
import multiprocessing

import scanpy as sc
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

# Check if input JSON exists and read it
try:
    print(sys.argv[1])
    config = json.load(open(sys.argv[1]))
except (NameError, FileNotFoundError) as error:
    print("Please add JSON with configuration!")
    exit(1)
results_dir_all = config["results_dir"].split(',')
hq_path = config["hq_path"]
lq_paths = config["lq_paths"]
annotation = config["annotation"]
algo_suffix_all = config["output_suffix"].split(',')
distance_metrics = config["distance_metric"].split(',')

start = time.time()
# Read HQ data
adata_sc = sc.read_h5ad(hq_path)
adata_sc.var_names_make_unique()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def calc_metric(actual_labels, pred, ind):
    # Calculate the F-score per class (macro) since micro is the same as accuracy
    f_score = f1_score(actual_labels, pred, average="macro")

    # Calculate accuracy
    acc_score = accuracy_score(actual_labels, pred)

    # Create a pandas DataFrame to store the results
    results_df = pd.DataFrame(
        {"Subsample": [ind], "F-score": [f_score], "Accuracy": [acc_score]}
    )
    return results_df

for results_dir, algo_suffix, distance_metric in zip(results_dir_all, algo_suffix_all, distance_metrics):
    print('*'*30, algo_suffix, distance_metric) 
    out_df = pd.DataFrame()
    for lq_path in lq_paths:
        print(f"Processing {lq_path}")
    
        if algo_suffix == "CoDi":
            path_dist_name = distance_metric if distance_metric != 'CoDi_contrastive' else 'KLD'
            lq_path_pred = os.path.join(
                results_dir, os.path.basename(lq_path).replace(
                    ".h5ad", f"_{algo_suffix}_{path_dist_name}.h5ad"
                )
            )
            subsample_factor_pos = -3
            out_file_name = f"{os.path.basename(sys.argv[1]).split('.')[0]}_benchmark_{algo_suffix}_{distance_metric}.csv"
        else:
            lq_path_pred = os.path.join(
                results_dir, os.path.basename(lq_path).replace(
                    ".h5ad", f"_{algo_suffix}.h5ad"
                )
            )
            subsample_factor_pos = -2
            out_file_name = f"{os.path.basename(sys.argv[1]).split('.')[0]}_benchmark_{algo_suffix}.csv"
        if not os.path.exists(lq_path_pred):
            # raise ValueError(f"no file {lq_path_pred}")
            print(f"no file {lq_path_pred}")
            continue
        print('Reading ' + lq_path_pred)
        adata_st = sc.read_h5ad(lq_path_pred)

        adata_st.var_names_make_unique()
        subsample_factor = (
            lq_path_pred.split("_")[subsample_factor_pos]
            if is_number(lq_path_pred.split("_")[subsample_factor_pos])
            else 0.0
        )
        algo_annotation = 'ssi' if (algo_suffix == 'CoDi' and 'ssi' in adata_st.obs.columns) else ('CoDi_contrastive' if distance_metric == 'CoDi_contrastive' else algo_suffix)
        res_df = calc_metric(
            adata_sc.obs[annotation], adata_st.obs[f'{algo_annotation}'], subsample_factor
        )
        out_df = pd.concat([out_df, res_df])
        print(out_df)
    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(out_file_name)
    print('Writing ' + out_file_name)
end = time.time()
print(f"benchmark_from_results.py took: {end-start}s")


