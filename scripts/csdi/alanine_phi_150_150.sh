#!/usr/bin/env bash
#SBATCH --job-name=CSDI_150_150_phi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=#####
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=logs/csdi/CSDI_150_150_phi.%j.out
#SBATCH --error=logs/csdi/CSDI_150_150_phi.%j.err

mkdir -p results/csdi
mkdir -p checkpoints_150_150/csdi/alanine_phi
mkdir -p checkpoints_150_150/csdi/alanine_psi
mkdir -p checkpoints_150_150/csdi/double_well
mkdir -p checkpoints_150_150/csdi/single_well

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



python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets_150_150/alanine_phi \
        --config configs/tsdiff_cond_train/alanine_phi_150_150.yaml \
        --out_dir checkpoints_150_150/tsdiff_cond/alanine_phi