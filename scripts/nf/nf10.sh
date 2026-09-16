#!/usr/bin/env bash
#SBATCH --job-name=nf10
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:a30:1 
#SBATCH --mem=32G
#SBATCH --output=logs/nftsf/nf10.%j.out
#SBATCH --error=logs/nftsf/nf10.%j.err
#SBATCH --mail-user=meahmed@asu.edu

source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
conda activate venv310
which python

mkdir -p checkpoints_25_25/nf_encoder_k3/electricity_nips/cnn 
mkdir -p checkpoints_50_50/nf_encoder_k3/electricity_nips/cnn 
mkdir -p checkpoints_100_100/nf_encoder_k3/electricity_nips/cnn 



which python

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge
module load cuda-12.8.1-gcc-12.1.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

rm -rf ~/.cache/keops* 2>/dev/null
rm -rf ~/.cache/pykeops* 2>/dev/null
rm -rf /tmp/keops* 2>/dev/null
rm -rf /tmp/pykeops* 2>/dev/null

export PYKEOPS_BUILD_DIR="$SLURM_TMPDIR/pykeops_build"
export KEOPS_CACHE_FOLDER="$SLURM_TMPDIR/keops_cache"

if [ -z "$SLURM_TMPDIR" ]; then
    export PYKEOPS_BUILD_DIR="$HOME/.cache/pykeops_build"
    export KEOPS_CACHE_FOLDER="$HOME/.cache/keops_cache"
fi

mkdir -p "$PYKEOPS_BUILD_DIR"
mkdir -p "$KEOPS_CACHE_FOLDER"

echo "PyKeOps build dir: $PYKEOPS_BUILD_DIR"
echo "KeOps cache dir: $KEOPS_CACHE_FOLDER"

echo "Checking CUDA installation..."
ls $CUDA_PATH/include/cuda.h || echo "WARNING: cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "WARNING: nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so* || ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "WARNING: libnvrtc.so not found"

python eval/NF/forecast_nf_encoder_nips.py   \
    --model_path ./checkpoints_50_50/nf_encoder_k3/electricity_nips/cnn/model.pth   \
    --config ./checkpoints_50_50/nf_encoder_k3/electricity_nips/cnn/config.json  \
    --data_path data_raw/electricity_nips_test.npz  \
    --out results/nf_encoder/electricity_nips_cnn_k3_50_50_100_samples.npz  \
    --seed 123   --n_samples 100 --batch_size 64 --device cuda

python eval/NF/forecast_nf_encoder_nips.py   \
    --model_path ./checkpoints_100_100/nf_encoder_k3/electricity_nips/cnn/model.pth   \
    --config ./checkpoints_100_100/nf_encoder_k3/electricity_nips/cnn/config.json  \
    --data_path data_raw/electricity_nips_test.npz  \
    --out results/nf_encoder/electricity_nips_cnn_k3_100_100_100_samples.npz  \
    --seed 123   --n_samples 100 --batch_size 64 --device cuda
