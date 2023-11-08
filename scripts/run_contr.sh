#!/bin/bash

python scripts/contrastive.py --sc_path /home/nikola_dev/ssi/data/Mouse_brain_SC.h5ad --st_path /home/nikola_dev/ssi/data/Mouse_brain_ST.h5ad --annotation cell_subclass --annotation_st celltype --epochs 10
# python contrastive.py --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad --annotation cell_subclass --annotation_st celltype --epochs 100
# python contrastive.py --sc_path /goofys/Samples/sc_reference/mouse_brain_L5_csr.h5ad --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad --annotation cell_subclass --annotation_st celltype --epochs 100
