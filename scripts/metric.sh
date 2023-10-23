#!/bin/bash
# Run from ssi directory to create reports in ssi/data/reports

FILES_LB_B5="/goofys/projects/SSI/testing/tangram/L5_B5_265ct_tangram.csv
/goofys/projects/SSI/testing/spatialid/L5_B5_265ct_spatialid.csv
/goofys/projects/SSI/testing/cytospace/L5_B5_265ct_cytospace.csv"

FILES_10X="/goofys/projects/SSI/testing/cytospace/visium_mouse_brain_cytospace.csv
/goofys/projects/SSI/testing/seurat/visium_mouse_brain_seurat.csv"

for f in $FILES_LB_B5
do
	echo "Processing $f"
    python scripts/metric.py --sc_path /goofys/projects/SSI/datasets/mouse_brain_L5.h5ad --st_path /goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad --st_cell_type_path $f & 
done

for f in $FILES_10X
do
	echo "Processing $f"
    python scripts/metric.py --sc_path /goofys/Samples/sc_reference/visium_mouse_brain_cell2location.h5ad  --st_path /goofys/Samples/10X/mouse_brain_visium_cell2location.h5ad -a annotation_1  --st_cell_type_path $f & 
done


wait
echo "DONE"