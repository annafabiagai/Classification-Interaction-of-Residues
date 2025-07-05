# Contact predictor - Group 1

A command-line tool to download mmCIF files, extract 3Di and feature data predict residue-residue interactions.

## Features
This script performs the following steps:
1. Parse command-line arguments.
2. Create the output directory if it doesn't exist.
3. Download the mmCIF file for the specified PDB ID.
4. Run the 3Di extraction script.
5. Run the feature extraction script (optionally with a config file).
6. Load and merge the 3Di and feature TSV outputs into a single DataFrame.
7. Clean up intermediate files (*.cif, *_3di.tsv, *_features.tsv).
8. Load pre-trained scaler and ML models.
9. Prepare the feature matrix (dummy-encode categorical cols, scale).
10. Run multi-class and multi-label predictions.
11. Save the final merged TSV with interaction predictions.

## Requirements
- pandas  
- numpy  
- scikit-learn  
- joblib  

and

- Internet access to download CIF file.

## Usage

Run the pipeline with a PDB ID:

```bash
python extractdata.py <PDB_ID> [options]
```

### Command-line Arguments

| Option             | Description                               | Default            |
| ------------------ | ----------------------------------------- | ------------------ |
| `<PDB_ID>`         | PDB identifier (e.g., `1ABA`).            | (required)         |
| `-out_dir`         | Output directory for all generated files. | `outputs`          |
| `-script`          | Path to the 3Di extraction script.        | `calc_3di.py`      |
| `-features_script` | Path to the feature extraction script.    | `calc_features.py` |
| `-config`          | JSON config for feature extraction.       | _None_             |

### Example

```bash
python extractdata.py 1ABA \
    -out_dir results \
    -script scripts/calc_3di.py \
    -features_script scripts/calc_features.py \
    -config config/features.json
```

This will:
1. Download `1aba.cif` into `results/`.
2. Extract 3Di states.
3. Compute structural features.
4. Merge and clean intermediate data.
5. Prepare features and run interaction predictions.
6. Save final output: `results/1aba_predicted.tsv`.

## Project Structure

```
./
├── calc_3di.py                # 3Di extraction script
├── calc_features.py           # Feature extraction script
├── extractdata.py             # Pipeline
├── configuration.json         # Configuration file
├── arrays.py                  # Columns Arrays
├── ....                       # Other helper files
├── mappings.py                # id-label mappings
├── models/                    # Pre-trained models and scaler
│   ├── multi_class_weights.joblib
│   ├── multi_label_weights.joblib
│   └── scaler.pkl
├── taining/                   # Training material
└── outputs/                   # Generated outputs (after running)
    ├── <PDB_ID>.cif           # (deleted)
    ├── <PDB_ID>_3di.tsv       # (deleted)
    ├── <PDB_ID>_features.tsv  # (deleted)
    ├── <PDB_ID>.tsv
    └── <PDB_ID>_predicted.tsv
```

## Pipeline Details

1. **Download mmCIF**: `download_mmCIF(pdb_id, out_dir)`
2. **3Di Extraction**: `run_3di_extraction(cif_path, out_dir, script_path)`
3. **Feature Extraction**: `run_features_extraction(cif_path, out_dir, features_script, config)`
4. **Merge & Clean**: Load TSVs, merge on residue indices, compute distances, drop unused columns.
5. **Model Prediction**:
   - Load `scaler.pkl`, `multi_class_weights.joblib`, `multi_label_weights.joblib`.
   - One-hot encode categorical features (secondary structure states).
   - Scale, predict interaction class and multi-label interactions.
6. **Output**: Final TSV with predictions.

## Team members
- Annafabia Gai
- Riccardo Zanatta
- Filippo Tiberio

## Training Setup
The training of the models was performed using the entire provided dataset, split into 80% for training and 20% for testing/validation, on a cloud computing machine equipped with:
 - L40S GPU
 - 16 vCPUs
 - 94 GB of RAM.

Final models training is carried out after hyperparameter optimization via Randomized Cross-Validation and takes approximately 2 hours per model.



