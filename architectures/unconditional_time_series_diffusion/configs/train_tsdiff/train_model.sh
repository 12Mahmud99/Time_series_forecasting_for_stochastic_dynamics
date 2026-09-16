  1 #!/usr/bin/env bash
  2 #SBATCH --job-name=uncondTSFdiff
  3 #SBATCH --time=3-00:00:00          # 3 days max
  4 #SBATCH --nodes=1
  5 #SBATCH --ntasks=1
  6 #SBATCH --partition=<your-partition>  # TODO: set to your cluster partition
  7 #SBATCH --qos=<your-qos>  # TODO: set to your cluster QOS
  8 #SBATCH --gres=gpu:2               # Request 1 GPU
  9 #SBATCH --mem=32G                   # Adjust memory as needed
 10 #SBATCH --output=slurm.%j.out      # STDOUT log
 11 #SBATCH --error=slurm.%j.err       # STDERR log
 12 echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
 13 module purge
 14 module load cuda-12.8.1-gcc-12.1.0             # Load correct CUDA version
 15 
 16 export CUDA_HOME=/packages/apps/cuda/12.8.1
 17 export PATH=$CUDA_HOME/bin:$PATH
 18 export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 19 export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE
 20 rm -rf ~/.cache/keops2.1.1
 21 python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml
~                                                                                      
~                                                                                      
~                                                                                      
~                                                              