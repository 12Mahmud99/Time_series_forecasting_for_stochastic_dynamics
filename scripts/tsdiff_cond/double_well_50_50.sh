#!/usr/bin/env bash
#SBATCH --job-name=phiTSDIff
#SBATCH --mail-type=ALL
#SBATCH --time=1-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=<your-partition>  # TODO: set to your cluster partition
#SBATCH --qos=<your-qos>  # TODO: set to your cluster QOS
#SBATCH --gres=gpu:<type>:1  # TODO: set to your GPU type (e.g. gpu:a30:1, gpu:a100:1)
#SBATCH --mem=32G




source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
conda activate /home/.conda/envs/venv310

mkdir -p checkpoints_25_25/tsdiff_cond/double_well
mkdir -p results/tsdiff_cond/double_well
mkdir -p logs/tsdiff_cond/double_well

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

python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets/double_well \
        --config configs/tsdiff_cond_train/double_well_50_50.yaml \
        --out_dir checkpoints_50_50_dummy/tsdiff_cond/double_well
   
python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/double_well_cond_50_50.yaml  \
        --dataset_path gluonts_datasets/double_well \
        --out results/tsdiff_cond/double_well_50_50.npz