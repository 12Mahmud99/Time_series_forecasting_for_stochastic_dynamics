#!/usr/bin/env bash
#SBATCH --job-name=CSDI10050
#SBATCH --time=02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a30:1 
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/CSDI10050.%j.out
#SBATCH --error=logs/csdi/CSDI10050.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/alanine_phi_25_25
mkdir -p checkpoints_25_25_dummy/csdi_with_early_stopping/
mkdir -p checkpoints_25_25_dummy/csdi/alanine_phi

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


python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_psi_100_50.yaml \
    --input  data/alanine_psi_test.npz \
    --ckpt   checkpoints_100_50/csdi/alanine_psi/model.pth \
    --out    results/csdi/alanine_psi_100_50.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_phi_100_50.yaml \
    --input  data/alanine_phi_test.npz \
    --ckpt   checkpoints_100_50/csdi/alanine_phi/model.pth \
    --out    results/csdi/alanine_phi_100_50.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/double_well_100_50.yaml \
    --input  data/double_well_test.npz \
    --ckpt   checkpoints_100_50/csdi/double_well/model.pth \
    --out    results/csdi/double_well_100_50.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/single_well_100_50.yaml \
    --input  data/single_well_test.npz \
    --ckpt   checkpoints_100_50/csdi/single_well/model.pth \
    --out    results/csdi/single_well_100_50.npz \
    --device cuda:0