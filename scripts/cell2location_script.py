import os
import time

import torch

import numpy as np
import scanpy as sc
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32,force_device=True"
torch.set_float32_matmul_precision("medium")

import cell2location

from matplotlib import rcParams
from scipy.sparse import issparse, csr_matrix

rcParams["pdf.fonttype"] = 42  # enables correct plotting of text for PDFs


def test_cell2location(args):
    if torch.cuda.is_available():
        use_gpu = int(torch.cuda.is_available())
        accelerator = "gpu"
    else:
        use_gpu = None
        accelerator = "cpu"

    sc_dataset = sc.read_h5ad(args.sc_path)
    sc_dataset.var_names_make_unique()
    sc_dataset.obs_names_make_unique()
    if issparse(sc_dataset.X):
        sc_dataset.X = sc_dataset.X.tocsr().todense()
        sc_dataset.uns["is_sparse"] = True
    else:
        sc_dataset.uns["is_sparse"] = False
    if ("log1p" in sc_dataset.uns.keys()) and np.max(sc_dataset.X) != np.round(
        np.max(sc_dataset.X)
    ):
        # gene expression matrix is logged
        # perform exponential correction
        sc_dataset.uns["logged"] = True
        print(
            "Cell2location requires data values before log transform, trying to revert SC dataset..."
        )
        sc_dataset.X = np.exp(sc_dataset.X) - 1
    else:
        sc_dataset.uns["logged"] = False
    if np.max(sc_dataset.X) != np.round(np.max(sc_dataset.X)):
        # unnormalize the data for cell2location
        # cell2location can only work with unnormalized expression values
        print(
            "Cell2location requires unnormalized data, trying to unnormalize SC dataset..."
        )
        # find lowest nonzero value
        tmp = sc_dataset.X.copy()
        tmp[tmp == 0] = 1
        sc_dataset.X = sc_dataset.X / np.min(tmp)
        del tmp
        sc_dataset.X = np.ceil(sc_dataset.X)

    sc_dataset.obsm["spatial"] = np.random.normal(0, 1, [sc_dataset.n_obs, 2])
    sc_dataset.obs["batch"] = [0 for _ in sc_dataset.obs_names]

    # read ST data
    st_dataset = sc.read_h5ad(args.st_path)
    st_dataset.var_names_make_unique()
    st_dataset.obs_names_make_unique()
    if issparse(st_dataset.X):
        st_dataset.X = st_dataset.X.tocsr().todense()
        st_dataset.uns["is_sparse"] = True
    else:
        st_dataset.uns["is_sparse"] = False
    if ("log1p" in st_dataset.uns.keys()) and np.max(st_dataset.X) != np.round(
        np.max(st_dataset.X)
    ):
        # gene expression matrix is logged
        # perform exponential correction
        st_dataset.uns["logged"] = True
        print(
            "Cell2location requires data values before log transform, trying to revert ST dataset..."
        )
        st_dataset.X = np.exp(st_dataset.X) - 1
    else:
        st_dataset.uns["logged"] = False
    if np.max(st_dataset.X) != np.round(np.max(st_dataset.X)):
        # unnormalize the data for cell2location
        # cell2location can only work with unnormalized expression values
        print(
            "Cell2location requires unnormalized data, trying to unnormalize ST dataset..."
        )
        st_dataset.X = np.exp(st_dataset.X) - 1
        # find lowest nonzero value
        tmp = st_dataset.X.copy()
        tmp[tmp == 0] = 1
        st_dataset.X = st_dataset.X / np.min(tmp)
        del tmp
        st_dataset.X = np.ceil(st_dataset.X)
    # place cell spatial coordinates in .obsm['spatial']
    # coordinates are expected in 'spatial', 'X_spatial', and 'spatial_stereoseq'
    if "X_spatial" in st_dataset.obsm:
        st_dataset.obsm["spatial"] = st_dataset.obsm["X_spatial"].copy()
    elif "spatial_stereoseq" in st_dataset.obsm:
        st_dataset.obsm["spatial"] = np.array(
            st_dataset.obsm["spatial_stereoseq"].copy()
        )
    elif "spatial" in st_dataset.obsm:
        pass
    else:
        args.plotting = 0
        st_dataset.obsm["spatial"] = np.random.normal(0, 1, [st_dataset.n_obs, 2])
        print(
            'WARNING: Spatial coordinates not found. Labels expected in: \
            .obsm["spatial"] or\n \
            .obsm["X_spatial"] or\n \
            .obsm["spatial_stereoseq"]'
        )

    cell2location.models.RegressionModel.setup_anndata(
        sc_dataset, labels_key=args.annotation, batch_key="batch"
    )

    # train regression model to get signatures of cell types
    sc_model = cell2location.models.RegressionModel(sc_dataset)
    # test full data training
    sc_model.train(max_epochs=args.max_epochs, accelerator=accelerator)
    # export the estimated cell abundance (summary of the posterior distribution)
    # NOTE: index of sc_dataset is changed through export_posterior
    sc_dataset = sc_model.export_posterior(
        sc_dataset,
        sample_kwargs={
            "num_samples": 200,
            "batch_size": args.batch_size,
            "accelerator": accelerator,
        },
    )
    # plot _scvi_labels
    if args.plotting > 1:
        sc_dataset.obs["_scvi_labels"] = sc_dataset.obs["_scvi_labels"].astype("str")
        figure, axes = plt.subplots(nrows=1, ncols=2)
        sc.pl.spatial(
            sc_dataset, color=args.annotation, spot_size=args.spot_size, ax=axes[0]
        )
        sc.pl.spatial(
            sc_dataset, color="_scvi_labels", spot_size=args.spot_size, ax=axes[1]
        )
        figure.savefig(f"{run_name}/sc_scvi_labels.png", dpi=200, bbox_inches="tight")
        plt.close()

    # test plot_QC
    if args.plotting > 2:
        plt.figure
        sc_model.plot_QC()
        plt.savefig(f"{run_name}/sc_qc_metric.png")
        plt.close()

    # # test save/load
    # sc_model.save(ref_run_name, overwrite=True, save_anndata=True)
    # sc_model = cell2location.models.RegressionModel.load(ref_run_name)

    if args.plotting > 0:
        # Save anndata object with results
        adata_file = f"{ref_run_name}/sc.h5ad"
        if sc_dataset.uns["is_sparse"]:
            sc_dataset.X = csr_matrix(sc_dataset.X)
        sc_dataset.write(adata_file)

    # export estimated expression in each cluster
    if "means_per_cluster_mu_fg" in sc_dataset.varm.keys():
        inf_aver = sc_dataset.varm["means_per_cluster_mu_fg"][
            [
                f"means_per_cluster_mu_fg_{i}"
                for i in sc_dataset.uns["mod"]["factor_names"]
            ]
        ].copy()
    else:
        inf_aver = sc_dataset.var[
            [
                f"means_per_cluster_mu_fg_{i}"
                for i in sc_dataset.uns["mod"]["factor_names"]
            ]
        ].copy()
    inf_aver.columns = sc_dataset.uns["mod"]["factor_names"]

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(st_dataset.var_names, inf_aver.index)
    st_dataset = st_dataset[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=st_dataset, batch_key=None)

    # create and train the model
    mod = cell2location.models.Cell2location(
        st_dataset,
        cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=1,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20,
    )
    # mod.view_anndata_setup()

    mod.train(
        max_epochs=args.max_epochs,
        # train using full data (batch_size=None)
        batch_size=args.batch_size,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1,
        accelerator=accelerator,
    )

    # # plot ELBO loss history during training, removing first 100 epochs from the plot
    # if args.plotting > 0:
    #     plt.figure
    #     mod.plot_history(1000)
    #     plt.legend(labels=["full data training"])
    #     plt.savefig(f"{run_name}/ELBO_loss.png")
    #     plt.close()

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    st_dataset = mod.export_posterior(
        st_dataset,
        sample_kwargs={
            "num_samples": args.num_samples,
            "batch_size": (
                mod.adata.n_obs
                if (args.batch_size == None or args.batch_size > mod.adata.n_obs)
                else args.batch_size
            ),
            "accelerator": accelerator,
        },
    )
    # # plot _scvi_labels
    # if args.plotting > 1:
    #     st_dataset.obs["_scvi_labels"] = st_dataset.obs["_scvi_labels"].astype("str")
    #     figure, axes = plt.subplots(nrows=1, ncols=2)
    #     sc.pl.spatial(st_dataset, color=args.annotation_st, spot_size=args.spot_size, ax=axes[0])
    #     sc.pl.spatial(st_dataset, color="_scvi_labels", spot_size=args.spot_size, ax=axes[1])
    #     figure.savefig(f"{run_name}/st_scvi_labels.png", dpi=200, bbox_inches="tight")
    #     plt.close()

    # # Save model
    # mod.save(f"{run_name}", overwrite=True)
    # mod = cell2location.models.Cell2location.load(f"{run_name}", st_dataset)

    # test plot_QC
    if args.plotting > 2:
        plt.figure
        mod.plot_QC()
        plt.savefig(f"{run_name}/st_qc_metric.png")

    # add 5% quantile, representing confident cell abundance, 'at least this amount is present',
    # to adata.obs with nice names for plotting
    # cell_types = st_dataset.uns["mod"]["factor_names"]
    # st_dataset.obs[cell_types] = st_dataset.obsm["q05_cell_abundance_w_sf"]
    # figure, axes = plt.subplots(nrows=1, ncols=len(cell_types))
    # for i, ct in enumerate(cell_types):
    #     sc.pl.spatial(st_dataset, color=ct, spot_size=args.spot_size, ax=axes[i])
    # figure.savefig(f'{run_name}/st_celltypes_05.png', dpi=200, bbox_inches='tight')

    st_dataset.obs["cell2location_q05"] = np.array(
        [
            prow.idxmax().split("q05cell_abundance_w_sf_")[1]
            for _, prow in st_dataset.obsm["q05_cell_abundance_w_sf"].iterrows()
        ]
    )
    st_dataset.obs["cell2location_q95"] = np.array(
        [
            prow.idxmax().split("q95cell_abundance_w_sf_")[1]
            for _, prow in st_dataset.obsm["q95_cell_abundance_w_sf"].iterrows()
        ]
    )
    st_dataset.obs["cell2location"] = np.array(
        [
            prow.idxmax().split("meanscell_abundance_w_sf_")[1]
            for _, prow in st_dataset.obsm["means_cell_abundance_w_sf"].iterrows()
        ]
    )

    # Save anndata object with results in CSV and H5AD format
    if args.annotation_st is not None:
        st_dataset.obs[
            [
                args.annotation_st,
                "cell2location",
                "cell2location_q05",
                "cell2location_q95",
            ]
        ].to_csv(os.path.basename(args.st_path).replace(".h5ad", f"_cell2location.csv"))
    else:
        st_dataset.obs[
            ["cell2location", "cell2location_q05", "cell2location_q95"]
        ].to_csv(os.path.basename(args.st_path).replace(".h5ad", f"_cell2location.csv"))
    # check if data was sparse and return to sparse before saving
    if st_dataset.uns["is_sparse"]:
        st_dataset.X = csr_matrix(st_dataset.X)
    st_dataset.write_h5ad(
        os.path.basename(args.st_path).replace(".h5ad", f"_cell2location.h5ad")
    )

    if args.plotting > 0:
        if args.annotation_st is not None:
            figure, axes = plt.subplots(nrows=1, ncols=2)
            sc.pl.spatial(
                st_dataset,
                color=args.annotation_st,
                spot_size=args.spot_size,
                ax=axes[0],
            )
            sc.pl.spatial(
                st_dataset, color="cell2location", spot_size=args.spot_size, ax=axes[1]
            )
        else:
            figure, axes = plt.subplots(nrows=1, ncols=1)
            sc.pl.spatial(
                st_dataset, color="cell2location", spot_size=args.spot_size, ax=axes
            )

        figure.savefig(
            f"{run_name}/st_cell2location_05.png", dpi=200, bbox_inches="tight"
        )


if __name__ == "__main__":

    start_time = time.perf_counter()

    parser = ap.ArgumentParser(
        description="A script that performs cell2location cell type annotation on ST data, given an SC reference."
    )
    parser.add_argument(
        "--sc_path", help="Path to .h5ad file with scRNA data.", type=str, required=True
    )
    parser.add_argument(
        "--st_path", help="Path to .h5ad file with ST data.", type=str, required=True
    )
    parser.add_argument(
        "-a",
        "--annotation",
        help="Label of SC .obs column containing SC cell types",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--annotation_st",
        help="Label of ST .obs column containing ST cell types",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        help="GPU training batch size. Enables processing with limited GPU memory, but reduces the models accuracy.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--max_epochs",
        help="GPU training/predicting maximum epochs number. Higher number provides more iterations for reducing the function loss.",
        type=int,
        required=False,
        default=200,
    )
    parser.add_argument(
        "--num_samples",
        help="GPU training/predicting number of samples. Lower number enables less memory consumption, but reduces the accuracy.",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--plotting",
        help="Level of plotting (images are saved). 0 - none, >0 - st spatal plot and loss fn, >1 - scvi label plots, >2 - qc metric plots.",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--spot_size",
        help="Spot size for plotting",
        type=float,
        required=False,
        default=0.1,
    )

    args = parser.parse_args()

    results_folder = (
        f'./results/cell2location_{os.path.basename(args.st_path).replace(".h5ad","")}/'
    )

    # create paths and names to results folders for reference regression and cell2location models
    ref_run_name = f"{results_folder}/reference_signatures"
    run_name = f"{results_folder}/cell2location_map"

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    if args.plotting > 0:
        if not os.path.exists(ref_run_name):
            os.mkdir(ref_run_name)
        if not os.path.exists(run_name):
            os.mkdir(run_name)

    import subprocess

    logger_fname = os.path.basename(args.st_path).replace(
        ".h5ad", "_cpu_gpu_memlog.csv"
    )
    if os.path.isfile(logger_fname):
        os.remove(logger_fname)

    logger_pid = subprocess.Popen(
        [
            "python",
            "scripts/log_gpu_cpu_stats.py",
            logger_fname,
        ]
    )
    print("Started logging compute utilisation")

    # test cell2location (main function)
    test_cell2location(args=args)

    # record execution time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total execution time of cell2location: {total_time:.4f} s")
    with open(
        os.path.basename(args.st_path).replace(".h5ad", "_time.txt"), "w"
    ) as text_file:
        text_file.write(f"Total execution time of cell2location: {total_time:.4f} s")

    # End the background process logging the CPU and GPU utilisation.
    logger_pid.terminate()
    print("Terminated the compute utilisation logger background process")

    # read cpu and gpu memory utilization
    logger_df = pd.read_csv(logger_fname)

    max_cpu_mem = logger_df.loc[:, "RAM"].max()
    with open(os.path.basename(args.st_path).replace(".h5ad", "_cpumem.txt"), "w") as f:
        f.write(f"Peak RAM Usage: {max_cpu_mem} MB\n")

    max_gpu_mem = logger_df.loc[:, "GPU 0"].max()
    with open(
        os.path.basename(args.st_path).replace(".h5ad", "_gpumem.txt"), "w+"
    ) as text_file:
        text_file.write(f"GPU: {max_gpu_mem} MB used")
