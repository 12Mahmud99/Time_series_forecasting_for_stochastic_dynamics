# Forecast Distributions, Not Trajectories: Rethinking Time Series Forecasting Benchmarks for Stochastic Dynamics

> Anonymous submission for NeurIPS double-blind review.
> Author and funding information omitted for anonymous review.

This repository is the official implementation of the paper *Forecast Distributions, Not Trajectories: Rethinking
Time Series Forecasting Benchmarks for Stochastic
Dynamics*.


---

## Repository Layout (after Zenodo download)

The full reproduction environment is split between this repository (code) and a
Zenodo record (data, checkpoints, pre-computed results). After cloning this repo
and downloading the Zenodo archive, the repository root **must** look like this:

```
TSF_For_StochasticDynamics/
├── NFTSF/                    ← normalizing-flow sub-module (own train/eval scripts)
├── architectures/            ← vendored baselines: CSDI, NF, TSDiff
├── utils/compare.py                ← main plotting entry point (this branch)
├── configs/                  ← training & forecast YAML/JSON configs per model & split
├── eval/                     ← evaluation scripts (ARIMA, CSDI, NF, TSDiff)
├── figures/                  ← output directories for generated figures
├── requirements.txt          ← Python dependencies
├── scripts/                  ← SLURM/bash runner scripts for each model × split
├── train/                    ← training scripts (CSDI, NFTSF, TSDiff)
├── __init__.py
├── checkpoints_25_25/        ← from Zenodo (context=25, prediction=25)
├── checkpoints_50_50/        ← from Zenodo (context=50, prediction=50)
├── data/                     ← from Zenodo (normalized .npz train/test splits)
├── data_raw/                 ← from Zenodo (raw .npy trajectories before normalization)
├── data_unnorm/              ← from Zenodo (unnormalized .npz, intermediate step)
├── gluonts_datasets/         ← from Zenodo (GluonTS-formatted datasets for TSDiff)
└── results/                  ← from Zenodo (pre-computed forecast .npz files)
```

Brief description of each Zenodo-hosted folder:

| Folder | Contents | Used by |
|---|---|---|
| `checkpoints_25_25/` | Saved model weights for context\_length=25 / prediction\_length=25. Sub-dirs: `csdi_with_early_stopping/`, `nf/`, `tsdiff/`, `tsdiff_cond_with_early_stopping/` each with one sub-dir per dataset. | `eval/CSDI/forecast_csdi.py`, `eval/NF/forecast_nf.py`, `eval/TSDiff/forecast_tsdiff*.py` |
| `checkpoints_50_50/` | Saved model weights for context\_length=50 / prediction\_length=50. Same structure as above. | Same eval scripts with 50\_50 configs |
| `data/` | Normalized `.npz` files (train/test splits) for all four datasets: `single_well`, `double_well`, `alanine_phi`, `alanine_psi`. | `eval/CSDI/`, `eval/NF/`, `eval/ARIMA/`, training scripts |
| `data_raw/` | Raw `.npy` trajectory files before normalization (input to `preprocessing/normalize.py`). | `preprocessing/normalize.py` |
| `data_unnorm/` | Unnormalized `.npz` files (output of `preprocessing/_concatenate.py`, input to `preprocessing/normalize.py`). | `preprocessing/normalize.py` |
| `gluonts_datasets/` | GluonTS-formatted datasets (one sub-dir per dataset with `train/`, `test/`, `metadata.json`). | `train/TSDiff/train_tsdiff.py`, `train/TSDiff/train_cond_tsdiff*.py`, `eval/TSDiff/forecast_tsdiff*.py` |
| `results/` | Pre-computed forecast `.npz` files produced by all evaluation scripts. Sub-dirs: `csdi_with_early_stopping/`, `nf/`, `tsdiff_cond_with_early_stopping/`, `tsdiff_mse/`, `tsdiff_q/`. **Input to `compare.py` for figure generation.** | `compare.py` |

---

## Downloading the Zenodo Data Archive

The data, checkpoints, and results required to reproduce this paper are bundled
into a single ZIP archive hosted anonymously on Zenodo. The archive contains all
seven folders and preserves their internal structure.

**Anonymous preview link (for reviewers):**

```
https://zenodo.org/records/20031454?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIxM2ZhMWFkLTE2NmYtNGMxYi04MzY3LTYwYTcxZWUyYjU4NSIsImRhdGEiOnt9LCJyYW5kb20iOiI5Y2Q0NWY4Mjc4ODE4ODE4MjQ4ZjczZjUxZDVjMDJlOCJ9.nXG6o-7ulOPwX4zbeSPZ7cDvWKp3d6uSZL0BiSCl22mf6RG08B4RJVI6GzsAmM4yuvt9L6JY4Zbp6DZG2LzisA
```

**Reconstruction steps:**

1. Open the link above in a browser. You will see one ZIP file on the record page.
2. Download the ZIP to the repository root.
3. Extract it in place so that all seven folders land directly at the repository
   root (at the same level as the source code), then remove the ZIP:

```bash
# From the repository root (nftsf_pipeline/)
unzip neurips_weights_data.zip -d .
rm neurips_weights_data.zip
```

4. Verify the layout matches the *Repository Layout* tree above. The seven
   folders must sit at the repository root.
5. If the ZIP extracts into a single wrapper directory (e.g.
   `nftsf_pipeline_data/checkpoints_25_25/...` rather than
   `checkpoints_25_25/...`), move the contents up one level:

```bash
# Only if the archive contains a wrapper directory:
mv <wrapper_dir>/* .
mv <wrapper_dir>/.[!.]* . 2>/dev/null || true
rmdir <wrapper_dir>
```

6. (Optional) Verify integrity with `sha256sum` against any `MANIFEST.txt` on
   the Zenodo record, if provided.

---

## Requirements

Python 3.10+ and a CUDA-capable GPU are recommended for training and evaluation.

```bash
# 1. Create and activate a conda environment
# Create Python 3.10 environment
conda create -n venv310 python=3.10
conda init
conda activate venv310

# 2. Install pip and all dependencies (pinned versions used in the paper)
conda install pip
pip install -r requirements.txt
```

**TSDiff baseline requires an additional editable install:**

```bash
cd architectures/unconditional_time_series_diffusion
pip install -e .
cd ../..
```

**Key packages** (see `requirements.txt` for pinned versions):

| Package | Version used |
|---|---|
| Python | 3.10 |
| PyTorch | 2.10.0 |
| pytorch-lightning | 1.9.4 |
| GluonTS | 0.12.3 |
| normflows | 1.7.3 |
| matplotlib | 3.10.8 |
| numpy | 1.23.5 |

Hardware used in the paper:

1× NVIDIA A30
---

## Training

All models are trained independently per dataset and per split configuration.
The `scripts/` directory contains ready-to-use SLURM/bash scripts for each
combination. Edit the `# TODO` lines at the top of each script to match your
cluster environment before submitting.

### Split configurations

| Config | `context_length` | `prediction_length` | Checkpoint dir |
|---|---|---|---|
| `25_25` | 25 | 25 | `checkpoints_25_25/` |
| `50_50` | 50 | 50 | `checkpoints_50_50/` |
| `100_100` | 100 | 100 | `checkpoints_100_100/` |

### NFTSF (proposed model)

```bash
python train/NFTSF/train_nftsf_with_encoder.py   \
    --data_path data/alanine_phi_train.npz  \
    --output_dir ./checkpoints_25_25/nf_encoder_k3/alanine_phi   \
    --model_name alanine_phi --n_past 25 --n_future 25   --epochs 1000 --batch_size 4096   \
    --val_fraction 0.1        --device cuda
```

Replace `alanine_phi` with `alanine_psi`, `double_well`, or `single_well` for
other datasets. Replace `25` with `50` or `100` for the 50\_50 split and 100\_100 respectively.
### CSDI

```bash
python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  data/alanine_phi_train.npz \
    --out    checkpoints_25_25/csdi/alanine_phi \
    --device cuda:0
```

Per-dataset/split bash scripts: `scripts/csdi/<dataset>_<split>.sh`

### TSDiff-Cond

```bash
module purge
module load cuda-12.8.1-gcc-12.1.0
```

```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"
export PYKEOPS_BUILD_DIR="$SLURM_TMPDIR/pykeops_build"
export KEOPS_CACHE_FOLDER="$SLURM_TMPDIR/keops_cache"
```


```bash
if [ -z "$SLURM_TMPDIR" ]; then
    export PYKEOPS_BUILD_DIR="$HOME/.cache/pykeops_build"
    export KEOPS_CACHE_FOLDER="$HOME/.cache/keops_cache"
fi

mkdir -p "$PYKEOPS_BUILD_DIR"
mkdir -p "$KEOPS_CACHE_FOLDER"
```

or run 

```bash
sbatch scripts/init.sh
```


```bash
python train/TSDiff/train_cond_tsdiff.py \
    --dataset_path gluonts_datasets/alanine_phi \
    --config configs/tsdiff_cond_train/alanine_phi_25_25.yaml \
    --out_dir checkpoints_25_25/tsdiff_cond/alanine_phi
```

Per-dataset/split bash scripts: `scripts/tsdiff_cond/<dataset>_<split>.sh`

### TSDiff (unconditional, MSE and Quantile variants)

```bash
python train/TSDiff/train_tsdiff.py \
    --dataset_path gluonts_datasets/alanine_phi \
    --config configs/tsdiff_train/alanine_phi_25_25.yaml \
    --out_dir checkpoints_25_25/tsdiff/alanine_phi
```

Per-dataset/split bash scripts: `scripts/tsdiff/<dataset>_<split>.sh`

---

## Evaluation

Run evaluation scripts after training (or after downloading pre-trained
checkpoints from Zenodo). Each script writes a `.npz` forecast file to
`results/`.

### NFTSF
python eval/NF/forecast_nf_encoder.py     --model_path ./checkpoints_100_100/nf_encoder_k4/alanine_psi/cnn/alanine_psi.pth     --config ./checkpoints_100_100/nf_encoder_k4/alanine_psi/cnn/config.json     --data_path data/alanine_psi_test.npz     --out results/nf_encoder/alanine_psi_cnn_k4_100_100.npz     --seed 123   --n_samples 1000 --batch_size 64 --device cuda

```bash
python eval/NF/forecast_nf_encoder.py \
    --model_path   checkpoints_25_25/nf_encoder_k3/alanine_phi/model.pth \
    --data_path    data/alanine_phi_test.npz \
    --config ./checkpoints_25_25/nf_encoder_k3/alanine_phi/config.json \
    --out          results/nf/alanine_phi_25_25.npz \
    --context_length 25 --prediction_length 25 \
    --n_samples 1000 --train_test_split 900 --test_size 3000 \
    --seed 42
```

### CSDI

```bash
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  data/alanine_phi_test.npz \
    --ckpt   checkpoints_25_25/csdi/alanine_phi/model.pth \
    --out    results/csdi/alanine_phi_25_25.npz \
    --device cuda:0
```

### TSDiff-Cond

```bash
python eval/TSDiff/forecast_tsdiff_cond.py \
    --config configs/tsdiff_forecast/alanine_phi_cond_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_cond/alanine_phi_25_25.npz
```

### TSDiff-MS (MSE loss)

```bash
python eval/TSDiff/forecast_tsdiff.py \
    --config configs/tsdiff_forecast/alanine_phi_mse_03_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_mse/alanine_phi_mse_03_25_25.npz
```

### TSDiff-Q (Quantile loss)

```bash
python eval/TSDiff/forecast_tsdiff.py \
    --config configs/tsdiff_forecast/alanine_phi_q_4_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_q/alanine_phi_q_4_25_25.npz
```

### ARIMA

```bash
python eval/ARIMA/run_auto_arima.py \
    -c configs/arima/25_25.yaml \
    --input data/alanine_phi_test.npz \
    --out results/arima/alanine_phi_25_25.npz \
    --n_jobs 8
```

---

## Pre-trained Models

Pre-trained checkpoints for both split configurations are included in the
Zenodo archive (see *Downloading the Zenodo Data Archive* above).

| Folder | Models included | Datasets |
|---|---|---|
| `checkpoints_25_25/` | NFTSF (`nf/`), CSDI (`csdi_with_early_stopping/`), TSDiff-Cond (`tsdiff_cond_with_early_stopping/`), TSDiff (`tsdiff/`) | single\_well, double\_well, alanine\_phi, alanine\_psi |
| `checkpoints_50_50/` | Same model set | Same datasets |

Each model/dataset sub-directory contains the saved weights (`.pth` or
`.ckpt`) and any metadata files (config, normalization stats, loss curves)
produced during training.

---

## Reproducing the Plots

This is the `plots` branch. All paper figures are generated by `compare.py`,
which reads the pre-computed `.npz` forecast files from `results/` (available
via Zenodo) and writes SVG/PNG figures to `./comparison/`.

**Full figure reproduction (after Zenodo download):**

```bash
# Example: all-methods comparison for alanine_phi, 25/25 split
python utils/compare.py \
    --results \
        alanine_phi:csdi:results/csdi/alanine_phi_25_25.npz \
        alanine_phi:tsdiff_cond:results/tsdiff_cond/alanine_phi_25_25.npz \
        alanine_phi:nftsf:results/nf/alanine_phi_25_25.npz \
        alanine_phi:tsdiff_q:results/tsdiff_q/alanine_phi_q_4_25_25.npz \
        alanine_phi:tsdiff_mse:results/tsdiff_q/alanine_phi_mse_03_25_25.npz \
    --data_npz \
        alanine_phi:data/alanine_phi_test.npz \
    --output_dir ./comparison_alanine_phi \
    --n_traj_show 3 \
    --seed 42 
```

> Run `python compare.py --help` to see all options. Figures are
> written to the directory specified by `--output_dir` (default: `./comparison/`).

Plots produced per landscape × split:
- `raw_trajectories_{landscape}.svg`
- `trajectory_comparison_{landscape}.svg`
- `histogram2d_comparison_{landscape}_dark.svg`
- `histogram2d_comparison_{landscape}_light.svg`
- `error_metrics_{landscape}_len{length}.svg`
- `summary_{landscape}_len{length}.png` / `.csv`

---

## License

> - `NFTSF` - CC BY 4.0
> - `architectures/CSDI/` — MIT License (Copyright 2021 Yusuke Tashiro)
> - `architectures/unconditional_time_series_diffusion/` — Apache 2.0

---

## Contributing

> Anonymous submission. Contribution guidelines will be added upon de-anonymization.
