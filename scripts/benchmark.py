import os
import subprocess
import json
import sys

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

# Read HQ data
adata_sc = sc.read_h5ad(hq_path)

def calc_metric(actual_labels, pred, ind):
    # Calculate the F-score per class (macro) since micro is the same as accuracy
    f_score = f1_score(actual_labels, pred, average="macro")  

    # Calculate accuracy
    acc_score = accuracy_score(actual_labels, pred)

    # Create a pandas DataFrame to store the results
    results_df = pd.DataFrame({
        'F-score': [f_score],
        'Accuracy': [acc_score]
    }, index=[ind])
    return results_df

out_df = pd.DataFrame()
for lq_path in lq_paths:
    print(f"Processing {lq_path}")
    
    command = command_gen.replace("hq_path", hq_path).replace("lq_path", lq_path).replace("annotation", annotation)
    lq_path_pred = os.path.basename(lq_path).replace(".h5ad", f"_{algo_suffix}.h5ad")
    if not os.path.exists(lq_path_pred):
        return_code = subprocess.call(command.split(" "))
    adata_st = sc.read_h5ad(lq_path_pred)
    subsample_factor = lq_path_pred.split('_')[-2]
    res_df = calc_metric(adata_sc.obs[annotation], adata_st.obs["sc_type"], subsample_factor)
    out_df = pd.concat([out_df, res_df])
    print(out_df)
out_df.to_csv(f"benchmark_{algo_suffix}.csv")
