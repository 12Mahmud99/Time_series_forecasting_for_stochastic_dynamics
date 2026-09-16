#!/usr/bin/env bash
#SBATCH --job-name=traffic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G

#SBATCH --output=logs/csdi/bash.%j.out
#SBATCH --error=logs/csdi/bash.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/alanine_phi_150_150
mkdir -p checkpoints_150_150/csdi_with_early_stopping/

conda activate venv310
module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"
export KEOPS_CACHE_DIR=$HOME/.cache/keops
mkdir -p $KEOPS_CACHE_DIR

ls $CUDA_PATH/include/cuda.h      || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h     || echo "nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so*  || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"

python eval/CSDI/forecast_csdi_nips.py \
    --config configs/csdi_train/traffic_nips.yaml \
    --input  data_raw/traffic_nips_test.npz \
    --ckpt   checkpoints_custom/csdi/traffic_nips/model.pth \
    --out    results/csdi/traffic_nips.npz \
    --device cuda:0
