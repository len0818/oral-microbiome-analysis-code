# Reproducible analysis code for the SNU oral microbiome study

This repository contains the author-generated code used for alpha-diversity visualization and genus-level modeling analyses.

## Important note

This is a **code-only** public release.

- No raw data are included in this repository.
- No local absolute paths are retained in the public code.
- All input and output locations must be supplied by the user at run time through command-line arguments.

## Repository structure

```text
.
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── src/
│   ├── alpha_diversity_boxplots.py
│   └── genus_model_benchmark.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── environment.yml
├── requirements.txt
└── README.md
```

## What this repository contains

- `src/alpha_diversity_boxplots.py`
  - loads alpha-diversity tables and sample metadata
  - merges tables by sample identifier
  - generates Shannon and observed-features boxplots
  - saves a combined figure

- `src/genus_model_benchmark.py`
  - loads genus-level abundance data from Excel or CSV
  - runs Kruskal-Wallis tests across phenotype groups
  - trains several classifiers for binary comparison
  - saves ROC curves and performance summaries

## Data policy for this repository

This public repository does **not** distribute the underlying study data.

If someone wants to execute the scripts, they must prepare their own input files with compatible column names and formats. You can also describe separate data-access conditions in the manuscript or Data Availability Statement if needed.

## Expected input layout

The scripts do not require a fixed repository path. One possible local layout is:

```text
my_project/
├── data/
│   ├── alpha/
│   │   ├── shannon/
│   │   │   ├── alpha-diversity.tsv
│   │   │   └── sample-metadata.tsv
│   │   └── observed_features/
│   │       ├── alpha-diversity.tsv
│   │       └── sample-metadata.tsv
│   └── genus/
│       └── SNU_oral_genus.xlsx
└── results/
```

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate snu-oral-microbiome
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Example usage

### 1) Alpha-diversity boxplots

```bash
python src/alpha_diversity_boxplots.py \
  --shannon-table data/alpha/shannon/alpha-diversity.tsv \
  --shannon-metadata data/alpha/shannon/sample-metadata.tsv \
  --observed-table data/alpha/observed_features/alpha-diversity.tsv \
  --observed-metadata data/alpha/observed_features/sample-metadata.tsv \
  --output results/Shannon_and_Observedfeature_boxplot.png
```

### 2) Genus-level benchmark

```bash
python src/genus_model_benchmark.py \
  --input data/genus/SNU_oral_genus.xlsx \
  --target status \
  --positive severe \
  --negative mild \
  --output-dir results/genus_benchmark
```

## Reproducibility notes

- Hard-coded absolute paths from the original working files were removed from the public scripts.
- The public release focuses on the cleaned executable scripts rather than exploratory notebooks.
- The genus modeling script explicitly applies preprocessing within a pipeline so scaling is handled reproducibly during model training and evaluation.
- If your column names differ from the expected defaults, update the script arguments or helper functions accordingly.


## License

This repository uses the MIT License. Replace it if your institution requires another open-source license.
