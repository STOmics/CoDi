# Contrastive Distance (CoDi) reference-based cell type annotation

Accurate cell type annotation utilizing neural network based on contrastive learning and advanced distance calculation methods and reference single-cell datasets

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)
- [License](#license)


## Requirements

- Python 3.x
- Required Python libraries can be installed using:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

```bash
python CoDi.py --sc_path <single_cell_dataset.h5ad> --st_path <spatial_dataset.h5ad> -a <sc_annotation>
```

## Parameters

- `--sc_path`: A single cell reference dataset (required)
- `--st_path`: A spatially resolved dataset (required)
- `-a, --annotation`: Annotation label for cell types (optional, default: "cell_subclass")
- `-d, --distance`: Distance metric used to measure the distance between a point and a distribution of points (optional, default: "KLD") 
  - Choices: "mahalanobis", "KLD", "wasserstein", "relativeEntropy", "hellinger", "binary", "none"
- `--num_markers`: Number of marker genes (optional, default: 100)
- `--dist_prob_weight`: Weight coefficient for probabilities obtained by distance metric. Weight for contrastive is 1.0 - dist_prob_weight. (optional, default: 0.5)
- `--batch_size`: Contrastive: Number of samples in the batch. Defaults to 512. (optional, default: 512)
- `--epochs`: Contrastive: Number of epochs to train deep encoder. Defaults to 50. (optional, default: 50)
- `--emb_dim`: Contrastive: Dimension of the output embeddings. Defaults to 32. (optional, default: 32)
- `--enc_depth`: Contrastive: Number of layers in the encoder MLP. Defaults to 4. (optional, default: 4)
- `--class_depth`: Contrastive: Number of layers in the classifier MLP. Defaults to 2. (optional, default: 2)
- `--augmentation_perc`: Contrastive: Percentage for the augmentation of SC data. If not provided it will be calculated automatically. Defaults to None. (optional, default: None)
- `--n_jobs`: Number of jobs to run in parallel. -1 means using all available processors. (optional, default: -1)
- `-c, --contrastive`: Enable contrastive mode (optional)
- `-v, --verbose`: Enable logging by specifying --verbose. (optional, default: logging.WARNING)
```
Example:

```bash
python CoDi.py --sc_path data/single_cell_dataset.h5ad --st_path data/spatial_dataset.h5ad -a celltype -d KLD --num_markers 150 --n_jobs 4
```

## Output

The script generates an output h5ad file containing the annotated spatial dataset (`<spatial_dataset>_CoDi_KLD.h5ad`) and CSV also contating the cell type annotation in the second column and cell IDs in the first column (`<spatial_dataset>_CoDi_KLD.csv`).. Additionally, a histogram of confidence scores and a confidence histogram plot are saved.

## Benchmark

The list of available paired datasets is available in `run_all.sh`.
This library can calculate retention of the marker genes in spatialy resolved datasets comparing to scRNA cell types. CSV generated by CoDi can be direct input to `scripts/metrics.py'. The command lines for our datasets are available in `scripts/metrics.sh'. Its output contains CSV where first row is for non-promiscuete (unique marker genes) and second is for top 100 marker genes.
The second benchmark uses scRNA downsampled datasets created using `scripts/create_synthetic.py'. `scripts/benchmark.py` can run CoDi for pairs of original and subsampled dataset and generate metrics for it. It receives configuration containing paths in input JSON file (e.g. `test/config_4k_adult_brain.json` and `test/config_40k_visium.json`). When benchmark for several tools is available it can be cisualized using 'scripts/viz_benchmark.py`.


## License

This project is licensed under the [MIT License](LICENSE).

