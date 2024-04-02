#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoDi_env

##------------------------------------------------##
## 4 ST test pairs
##------------------------------------------------##

# Adult mouse brain stereo-seq (celltype_pred)
python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad  -a cell_subclass --n_jobs 50 --num_markers 50 --contrastive --batch_size 2700 --epochs 300 --emb_dim 64 --verbose

# # Mouse brain 16.5 stereo-seq (sim anno)
python CoDi.py --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad  -a cell_subclass --n_jobs 50 --num_markers 50 --contrastive --batch_size 15000 --epochs 300 --emb_dim 64 --verbose

# # Mouse brain visium 40K cells
python CoDi.py --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location_pruned_inplace.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/visium_mouse_brain_cell2location_sc_unlog1p_pruned_inplace.h5ad -a annotation_1 --n_jobs 50 --contrastive --batch_size 2700 --epochs 300 --emb_dim 64 --verbose

# # Mouse kidney 43636x31053
python CoDi.py --st_path /goofys/Samples/slide_seq/cellxgene_kidney_slide_seq_v2/Puck_191223_19_corr_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --n_jobs 50 --contrastive --batch_size 2700 --epochs 300 --emb_dim 64 --verbose

##------------------------------------------------##
##------------------------------------------------##

# # Adult mouse brain stereo-seq 4K (celltype_pred)
# python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/adult_mouse_brain_ST4k.h5ad --sc_path /goofys/Samples/sc_reference/adult_mouse_brain_SC4k.h5ad -a cell_subclass --n_jobs 50

# # Adult mouse brain stereo-seq 7K
# python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/adult_mouse_brain_ST7k.h5ad --sc_path /goofys/Samples/sc_reference/adult_mouse_brain_SC7k.h5ad -a cell_subclass --n_jobs 50

# # Human heart dataset
# python CoDi.py --st_path /goofys/Samples/sc_reference/GSE121893_heart_4933_cells.h5ad --sc_path /goofys/Samples/sc_reference/GSE183852_heart.h5ad -a cell_type --n_jobs 50

# # Mop MERFISH (all 12 slices)
# python CoDi.py --st_path /goofys/Samples/MERFISH_dataset/mop/merfish13864.h5ad --sc_path /goofys/Samples/sc_reference/mop_merfish.h5ad -a Allen.cluster_label --n_jobs 50

# Benchmark synthetic visium
#python scripts/benchmark.py test/config_40k_visium.json

# Benchmark synthetic 4K
#python scripts/benchmark.py test/config_4k_adult_brain.json

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

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.5.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.6.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.7.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.8.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose

# python CoDi.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.9.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --epochs 300 --emb_dim 64 --batch_size 2700 --contrastive --n_jobs 50 --verbose