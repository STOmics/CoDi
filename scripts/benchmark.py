import os
import subprocess

import scanpy as sc
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

# st_paths = ['/home/vlada/ssi/data/4K/Mouse_brain_SC_0.05.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.1.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.2.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.3.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.5.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.7.h5ad',
#             '/home/vlada/ssi/data/4K/Mouse_brain_SC_0.9.h5ad']
# sc_path = '/home/vlada/ssi/data/4K/Mouse_brain_SC.h5ad'

st_paths = ['/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.05.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.1.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.2.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.3.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.5.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.7.h5ad',
            '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.9.h5ad']
sc_path = '/goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/sc_raw.h5ad'
annotation = "annotation_1"

adata_sc = sc.read_h5ad(sc_path)

def calc_metric(actual_labels, pred, ind):
    # Calculate the F-score
    f_score = f1_score(actual_labels, pred, average="micro")

    # Create a confusion matrix
    conf_mat = confusion_matrix(actual_labels, pred).ravel()

    # Calculate AUC
    # fpr, tpr, _ = roc_curve(actual_labels, pred)
    # roc_auc = auc(fpr, tpr)

    # Calculate accuracy
    acc_score = accuracy_score(actual_labels, pred)

    # Create a pandas DataFrame to store the results
    results_df = pd.DataFrame({
        'F-score': [f_score],
        'Accuracy': [acc_score]
    }, index=[ind])
    return results_df

out_df = pd.DataFrame()
for st_path in st_paths:
    print(f"Processing {st_path}")
    command = ["python", "ssi.py", "--sc_path", sc_path, "--st_path", st_path, "-a", annotation]
    st_path_pred = os.path.basename(st_path).replace(".h5ad", "_ssi.h5ad")
    if not os.path.exists(st_path_pred):
        return_code = subprocess.call(command)
    adata_st = sc.read_h5ad(st_path_pred)
    subsample_factor = st_path_pred.split('_')[-2]
    res_df = calc_metric(adata_sc.obs[annotation], adata_st.obs["sc_type"], subsample_factor)
    out_df = pd.concat([out_df, res_df])
    print(out_df)
out_df.to_csv("benchmark_visium_res.csv")
