import os
import subprocess
import json
import sys
import time

import scanpy as sc
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc


# Check if input JSON exists and read it
try:
    print(sys.argv[1])
    config = json.load(open(sys.argv[1]))
except (NameError, FileNotFoundError) as error:
    print("Please add JSON with configuration!")
    exit(1)
hq_path = config["hq_path"]
lq_paths = config["lq_paths"]
annotation = config["annotation"]
command_gen = config["command"]
algo_suffix = config["output_suffix"]

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
    results_df = pd.DataFrame({
        'Subsample': [ind],
        'F-score': [f_score],
        'Accuracy': [acc_score]
    })
    return results_df

out_df = pd.DataFrame()
for lq_path in lq_paths:
    print(f"Processing {lq_path}")
    
    command = command_gen.replace("hq_path", hq_path).replace("lq_path", lq_path).replace("annotation", annotation)
    command_split = command.split(" ")
    if algo_suffix == "ssi":
        if '-d' in command_split:
            distance_metric = command_split[[h for h, value in enumerate(command_split) if value == '-d'][0]+1]
        else:
            distance_metric = "KLD"
        lq_path_pred = os.path.basename(lq_path).replace(".h5ad", f"_{algo_suffix}_{distance_metric}.h5ad")
        subsample_factor_pos = -3
        out_file_name = f"../test/benchmark_{algo_suffix}_{distance_metric}.csv"
    else:
        lq_path_pred = os.path.basename(lq_path).replace(".h5ad", f"_{algo_suffix}.h5ad")
        subsample_factor_pos = -2
        out_file_name = f"../test/benchmark_{algo_suffix}.csv"
    if not os.path.exists(lq_path_pred):
        return_code = subprocess.call(command.split(" "))
    adata_st = sc.read_h5ad(lq_path_pred)
    adata_st.var_names_make_unique()
    subsample_factor = lq_path_pred.split('_')[subsample_factor_pos] \
        if is_number(lq_path_pred.split('_')[subsample_factor_pos]) else 0.0
    res_df = calc_metric(adata_sc.obs[annotation], adata_st.obs[algo_suffix], subsample_factor)
    out_df = pd.concat([out_df, res_df])
    print(out_df)
out_df = out_df.reset_index(drop=True)
out_df.to_csv(out_file_name)
end = time.time()
print(f"benchmark.py took: {end-start}s")
