#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoDi

# Adult mouse brain stereo-seq (celltype_pred)
python core/CoDi.py --sc_path ~/s3/projects/SSI/datasets/sc_reference/mouse_brain_L5_csr.h5ad --st_path ~/s3/projects/SSI/datasets/stereo_seq/mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad -a cell_subclass -v


# Mouse brain 16.5 stereo-seq (sim anno)
python core/CoDi.py --sc_path ~/s3/projects/SSI/datasets/sc_reference/mouse_brain_L5_csr.h5ad --st_path ~/s3/projects/SSI/datasets/stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad  -a cell_subclass -v

# Mouse brain visium 40K cells
python core/CoDi.py --sc_path ~/s3/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad --st_path ~/s3/projects/SSI/datasets/10x/mouse_brain_visium_cell2location_pruned_inplace.h5ad -a annotation_1 -v

# Mouse kidney 43636x31053
python core/CoDi.py --sc_path ~/s3/projects/SSI/datasets/sc_reference/GSE157079_P0_adult.h5ad --st_path ~/s3/projects/SSI/datasets/slide_seq/Puck_191223_19_corr_pruned_inplace.h5ad -a clusters_mod -v

##------------------------------------------------##
# # # Mouse kidney - subsampled
##------------------------------------------------##

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.05.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.1.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.2.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.3.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.5.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.6.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.7.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.8.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.9.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

##------------------------------------------------##
## visium mouse brain subsampled
##------------------------------------------------##

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.05.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.1.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.2.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.3.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose
# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.4.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.5.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.6.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.7.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.8.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.9.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

##------------------------------------------------##
## snRNA cell atlas
##------------------------------------------------##
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.05
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.05.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.1
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.1.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.2
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.2.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.3
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.3.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.4
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.4.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.5
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.5.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.6
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.6.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.7
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.7.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.8
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.8.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# echo ************************* 0.9
# python CoDi.py --st_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005_0.9.h5ad --sc_path /goofys/Samples/sc_reference/single_nucleus_cell_atlas/GTEX-1HSMQ-5005.h5ad -a "Cell types level 2"  --epochs 300 --emb_dim 64 --batch_size 2700 --n_jobs 50 --verbose >> out.log
# 
