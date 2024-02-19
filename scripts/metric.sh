#!/bin/bash
BASE_PATH="/goofys/projects/SSI/testing/"
# Run from ssi directory to create reports in ssi/data/reports

FILES_L5_B5="${BASE_PATH}CoDi/adult_mouse_brain_B5_CoDi.csv
${BASE_PATH}cell2location/adult_mouse_brain_B5_cell2location.csv
${BASE_PATH}tangram/adult_mouse_brain_B5_tangram.csv
${BASE_PATH}cytospace/adult_mouse_brain_B5_cytospace.csv
${BASE_PATH}seurat/adult_mouse_brain_B5_seurat.csv"

FILES_WHOLE_MOUSE="${BASE_PATH}CoDi/whole_brain_mouse_embryo_CoDi.csv
${BASE_PATH}cell2location/whole_brain_mouse_embryo_cell2location.csv
${BASE_PATH}tangram/whole_brain_mouse_embryo_tangram.csv
${BASE_PATH}cytospace/whole_brain_mouse_embryo_cytospace.csv
${BASE_PATH}seurat/whole_brain_mouse_embryo_seurat.csv"

FILES_10X="${BASE_PATH}CoDi/visium_mouse_brain_CoDi.csv
${BASE_PATH}cell2location/visium_mouse_brain_cell2location.csv
${BASE_PATH}tangram/visium_mouse_brain_tangram.csv
${BASE_PATH}cytospace/visium_mouse_brain_cytospace.csv
${BASE_PATH}seurat/visium_mouse_brain_seurat.csv"
#/goofys/projects/SSI/testing/RCTD/visium_mouse_brain_RCTD.csv"

FILES_KIDNEY="${BASE_PATH}CoDi/mouse_kidney_CoDi.csv
${BASE_PATH}cell2location/mouse_kidney_cell2location.csv
${BASE_PATH}seurat/mouse_kidney_seurat.csv
${BASE_PATH}tangram/mouse_kidney_tangram.csv
${BASE_PATH}cytospace/mouse_kidney_cytospace.csv"
# /goofys/projects/SSI/testing/ssi/mouse_kidney_Puck_191223_19_auth.csv
{
for f in $FILES_L5_B5
do
	echo "Processing $f"
   python scripts/metric.py --sc_path /goofys/projects/SSI/testing/sc_ref_with_marker_genes/mouse_brain_L5_csr.h5ad --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5_pruned_inplace.h5ad -a cell_subclass --st_cell_type_path $f
done
} 

{
for f in $FILES_WHOLE_MOUSE
do
	echo "Processing $f"
   python scripts/metric.py --sc_path /goofys/projects/SSI/testing/sc_ref_with_marker_genes/mouse_brain_L5_csr.h5ad --st_path /goofys/Samples/Stereo_seq/E16.5_E1S3_cell_bin_whole_brain_noborderct_unlog1p_pruned_inplace.h5ad  -a cell_subclass --st_cell_type_path $f 
done
} 

{
for f in $FILES_10X
do
	echo "Processing $f"
    python scripts/metric.py --sc_path /goofys/projects/SSI/testing/sc_ref_with_marker_genes/visium_mouse_brain_cell2location_sc_unlog1p_pruned_inplace.h5ad --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location_pruned_inplace.h5ad -a annotation_1  --st_cell_type_path $f 
done
} 

{
for f in $FILES_KIDNEY
do
	echo "Processing $f"
    python scripts/metric.py --sc_path /goofys/projects/SSI/testing/sc_ref_with_marker_genes/GSE157079_P0_adult.h5ad --st_path /goofys/Samples/slide_seq/cellxgene_kidney_slide_seq_v2/Puck_191223_19_corr_pruned_inplace.h5ad -a clusters --st_cell_type_path $f 
done
} 


wait
echo "DONE"