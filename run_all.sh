#!/bin/bash

# Adult mouse brain stereo-seq 4K (celltype_pred)
python ssi.py --st_path /goofys/projects/SSI/datasets/4K/Mouse_brain_ST.h5ad --sc_path /goofys/projects/SSI/datasets/4K/Mouse_brain_SC.h5ad -a cell_subclass

# Adult mouse brain stereo-seq (celltype_pred)
python ssi.py --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5.h5ad  -a cell_subclass

# Mouse brain 16.5 stereo-seq (sim anno)
python ssi.py --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct.h5ad --sc_path /goofys/Samples/sc_reference/mouse_brain_L5.h5ad  -a cell_subclass

# Mouse brain visium 40K cells
python ssi.py --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location.h5ad --sc_path /goofys/Samples/sc_reference/visium_mouse_brain_cell2location.h5ad -a annotation_1

# Benchmark synthetic visium
#python scripts/benchmark.py test/config_40k_visium.json

# Benchmark synthetic 4K
#python scripts/benchmark.py test/config_4k_adult_brain.json

