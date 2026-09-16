#!/usr/bin/env bash
#SBATCH --job-name=psiCSDI
#SBATCH --mail-type=ALL
#SBATCH --mail-user=###
#SBATCH --time=1-6:00:00
#SBATCH --nodes=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=logs/csdi/alanine_psi_100_100.%j.out
#SBATCH --error=logs/csdi/alanine_psi_100_100.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/alanine_psi_100_100
mkdir -p checkpoints_100_100/csdi_with_early_stopping/

source venv310/bin/activate
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


python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_psi_100_50.yaml \
    --input  data/alanine_psi_train.npz \
    --out    checkpoints_100_50/csdi/alanine_psi \
    --device cuda:0

python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_phi_100_50.yaml \
    --input  data/alanine_phi_train.npz \
    --out    checkpoints_100_50/csdi/alanine_phi \
    --device cuda:0