#!/usr/bin/env bash
#SBATCH --job-name=csdi_nips
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1  # TODO: set to your GPU type (e.g. gpu:a30:1, gpu:a100:1)
#SBATCH --mem=24G
#SBATCH --output=logs/csdi/csdi_nips-25.%j.out
#SBATCH --error=logs/csdi/csdi_nips.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/csdi_nips
mkdir -p checkpoints_custom/csdi/electricity_nips
mkdir -p checkpoints_custom/csdi/traffic_nips

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
    --config configs/csdi_train/electricity_nips.yaml \
    --input  data_raw/electricity_nips_train.npz \
    --out    checkpoints_custom/csdi/electricity_nips \
    --device cuda:0


python train/CSDI/train_csdi.py \
    --config configs/csdi_train/traffic_nips.yaml \
    --input  data_raw/traffic_nips_train.npz \
    --out    checkpoints_custom/csdi/traffic_nips \
    --device cuda:0
