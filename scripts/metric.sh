#!/bin/bash
BASE_PATH="data/tools_predictions/"

FILES_L5_B5=(
        "${BASE_PATH}adult_mouse_brain_B5_cell2location.csv"
        "${BASE_PATH}adult_mouse_brain_B5_CoDi.csv"
        "${BASE_PATH}adult_mouse_brain_B5_seurat.csv"
        "${BASE_PATH}adult_mouse_brain_B5_tangram.csv"
        "${BASE_PATH}adult_mouse_brain_B5_cytospace.csv"
)

FILES_WHOLE_MOUSE=(
       "${BASE_PATH}whole_brain_mouse_embryo_CoDi.csv"
       "${BASE_PATH}whole_brain_mouse_embryo_cell2location.csv"
       "${BASE_PATH}whole_brain_mouse_embryo_tangram.csv"
       "${BASE_PATH}whole_brain_mouse_embryo_cytospace.csv"
       "${BASE_PATH}whole_brain_mouse_embryo_seurat.csv"
)

FILES_10X=(
        "${BASE_PATH}visium_mouse_brain_CoDi.csv"
        "${BASE_PATH}visium_mouse_brain_cell2location.csv"
        "${BASE_PATH}visium_mouse_brain_tangram.csv"
        "${BASE_PATH}visium_mouse_brain_cytospace.csv"
        "${BASE_PATH}visium_mouse_brain_seurat.csv"
)

FILES_KIDNEY=(
       "${BASE_PATH}mouse_kidney_cell2location.csv"
       "${BASE_PATH}mouse_kidney_cell2location.csv"
       "${BASE_PATH}mouse_kidney_CoDi.csv"
       "${BASE_PATH}mouse_kidney_seurat.csv"
       "${BASE_PATH}mouse_kidney_cytospace.csv"
       "${BASE_PATH}mouse_kidney_tangram.csv"
)

{
for f in "${FILES_L5_B5[@]}"
do
   echo "Processing $f"
   python scripts/metric.py --sc_path mouse_brain_L5_csr.h5ad --st_path ~/s3/projects/SSI/datasets/stereo_seq/mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad -a cell_subclass --st_cell_type_path $f --annotation_st CoDi
done
} &

{
for f in "${FILES_WHOLE_MOUSE[@]}"
do

   echo "Processing $f"
   python scripts/metric.py --sc_path ~/s3/projects/SSI/datasets/sc_reference/mouse_brain_L5_csr.h5ad --st_path ~/s3/projects/SSI/datasets/stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad  -a cell_subclass --st_cell_type_path $f --annotation_st CoDi
done
} &

{
for f in "${FILES_10X[@]}"
do
   echo "Processing $f"
   python scripts/metric.py --sc_path synthetic_sc.h5ad --st_path ~/s3/projects/SSI/datasets/10x/mouse_brain_visium_cell2location_pruned_inplace.h5ad -a annotation_1  --st_cell_type_path $f --annotation_st CoDi
done
} &

{
for f in "${FILES_KIDNEY[@]}"
do
    echo "Processing $f"
    python scripts/metric.py --sc_path ~/s3/projects/SSI/datasets/sc_reference/GSE157079_P0_adult.h5ad --st_path ~/s3/projects/SSI/datasets/slide_seq/Puck_191223_19_corr_pruned_inplace.h5ad -a clusters_mod  --st_cell_type_path $f --annotation_st CoDi
done
} &

wait
echo "DONE"
