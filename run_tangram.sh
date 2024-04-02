#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangram-env

################## TANGRAM  ##########################

##------------------------------------------------##
## 4 ST test pairs
##------------------------------------------------##

# # Adult mouse brain stereo-seq (celltype_pred)
# python scripts/tangram_script.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad -a cell_subclass --annotation_st celltype_pred

# # Mouse brain 16.5 stereo-seq (sim anno)
# python scripts/tangram_script.py --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad -a cell_subclass --annotation_st "sim anno"

# # Mouse brain visium 40K cells
# python scripts/tangram_script.py --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location_pruned_inplace.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/visium_mouse_brain_cell2location_sc_unlog1p_pruned_inplace.h5ad -a annotation_1

# # Mouse kidney 43636x31053
# python scripts/tangram_script.py --st_path /goofys/Samples/slide_seq/cellxgene_kidney_slide_seq_v2/Puck_191223_19_corr_pruned_inplace.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st author_cell_type


##------------------------------------------------##
## Mouse kidney - synthetic TANGRAM
##------------------------------------------------##

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.05.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.1.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.2.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.3.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.5.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.6.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.7.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.8.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

# python scripts/tangram_script.py --st_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult_0.9.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters --annotation_st clusters 

##------------------------------------------------##
## visium mouse brain subsampled - TANGRAM
##------------------------------------------------##

python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.05.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.1.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.2.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.3.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.5.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.6.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.7.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.8.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1

# python scripts/tangram_script.py --st_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc_0.9.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/synthetic_sc.h5ad -a annotation_1 --annotation_st annotation_1
