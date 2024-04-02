#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cell2loc_env

################## CELL2LOCATION  ##########################

##------------------------------------------------##
## 4 ST test pairs
##------------------------------------------------##

# # Adult mouse brain stereo-seq (celltype_pred)
# python scripts/cell2location_script.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad  -a cell_subclass --annotation_st celltype_pred --batch_size 15000 --max_epochs 300 --num_samples 1000

# # Mouse brain 16.5 stereo-seq (sim anno)
# python scripts/cell2location_script.py --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad  -a cell_subclass --annotation_st "sim anno" --batch_size 15000 --max_epochs 300 --num_samples 1000

# Mouse brain visium 40K cells
python scripts/cell2location_script.py --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location_pruned_inplace.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/visium_mouse_brain_cell2location_sc_unlog1p_pruned_inplace.h5ad -a annotation_1 --batch_size 15000 --max_epochs 300 --num_samples 1000

# # Mouse kidney 43636x31053
# python scripts/cell2location_script.py --st_path /goofys/Samples/slide_seq/cellxgene_kidney_slide_seq_v2/Puck_191223_19_corr_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st author_cell_type --batch_size 15000 --max_epochs 300 --num_samples 1000

##------------------------------------------------##
## Visium mouse brain - subsampled - CELL2LOCATION
##------------------------------------------------##

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.05.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.1.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.2.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.3.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.5.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.6.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.7.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.8.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.9.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1 --batch_size 9000 --max_epochs 300 --num_samples 1000


# ##### 33 cell types test for adult mouse brain
# # Adult mouse brain stereo-seq (celltype_pred)
# python scripts/cell2location_script.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr_pruned_inplace_33celltypes.h5ad  -a cell_subclass --annotation_st celltype_pred --batch_size 15000 --max_epochs 300 --num_samples 1000

##------------------------------------------------##
# Mouse kidney - subsampled - CELL2LOCATION
##------------------------------------------------##

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.05.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.1.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.2.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.3.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.5.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.6.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.7.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.8.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000

# python scripts/cell2location_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.9.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters --batch_size 9000 --max_epochs 300 --num_samples 1000