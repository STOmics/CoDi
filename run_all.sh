#!/bin/bash

# Adult mouse brain stereo-seq 4K (celltype_pred)
python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/adult_mouse_brain_ST4k.h5ad --sc_path /goofys/Samples/sc_reference/adult_mouse_brain_SC4k.h5ad -a cell_subclass

# Adult mouse brain stereo-seq 7K
python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/adult_mouse_brain_ST7k.h5ad --sc_path /goofys/Samples/sc_reference/adult_mouse_brain_SC7k.h5ad -a cell_subclass

# Adult mouse brain stereo-seq (celltype_pred)
python CoDi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_L5.h5ad  -a cell_subclass --n_jobs 36 --num_markers 50

# Mouse brain 16.5 stereo-seq (sim anno)
python CoDi.py --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad  -a cell_subclass --num_markers 50

# Mouse brain visium 40K cells
python CoDi.py --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location.h5ad --sc_path /goofys/projects/SSI/datasets/mouse_brain_visium_cell2location/visium_mouse_brain_cell2location_sc.h5ad -a annotation_1

# Mouse kidney 43636x31053
python CoDi.py --st_path /goofys/Samples/slide_seq/cellxgene_kidney_slide_seq_v2/Puck_191223_19_corr.h5ad --sc_path /goofys/Samples/sc_reference/mouse_kidney/GSE157079_P0_adult.h5ad -a clusters

# Human heart dataset
python CoDi.py --st_path /goofys/Samples/sc_reference/GSE121893_heart_4933_cells.h5ad --sc_path /goofys/Samples/sc_reference/GSE183852_heart.h5ad -a cell_type --n_jobs 24

# Mop MERFISH (all 12 slices)
python CoDi.py --st_path /goofys/Samples/MERFISH_dataset/mop/merfish13864.h5ad --sc_path /goofys/Samples/sc_reference/mop_merfish.h5ad -a Allen.cluster_label

# Benchmark synthetic visium
#python scripts/benchmark.py test/config_40k_visium.json

# Benchmark synthetic 4K
#python scripts/benchmark.py test/config_4k_adult_brain.json

