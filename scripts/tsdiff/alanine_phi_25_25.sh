source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh

which python

mkdir -p results/tsdiff
mkdir -p checkpoints_25_25/tsdiff
mkdir -p results/tsdiff/checkpoints_25_25
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

python train/TSDiff/train_tsdiff.py \
        --dataset_path gluonts_datasets/alanine_phi \
        --config configs/tsdiff_train/alanine_phi_25_25.yaml \
        --out_dir checkpoints_25_25_dummy/tsdiff/alanine_phi

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/alanine_phi_mse_03_25_25.yaml  \
        --dataset_path gluonts_datasets/alanine_phi \
        --out results/tsdiff_mse/alanine_phi_mse_03_25_25.npz

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/alanine_phi_q_4_50_50.yaml  \
        --dataset_path gluonts_datasets/alanine_phi \
        --out results/tsdiff_q/alanine_phi_q_4_50_50.npz