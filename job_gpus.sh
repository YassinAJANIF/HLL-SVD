#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --gres=gpu:4               # GPUs per node
#SBATCH --constraint=h100          # Request H100 GPUs
#SBATCH --ntasks=4                 # Total number of MPI tasks
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=output_gpu/test.log
#SBATCH --error=output_gpu/test.err
#SBATCH --exclusive                # Use the node exclusively

# Load environment
source ~/.bashrc
# source activate new_env

conda activate new_env             # Change the env name if needed
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run with 4 GPUs (1 MPI rank per GPU)
mpirun -np 4 python3 main_parallel_gpu.py

