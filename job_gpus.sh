#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --cluster=nautilus
#SBATCH --partition=gpu            # Specify the partition
#SBATCH --nodelist=gnode1          # Specify the node name
#SBATCH --ntasks=4                 # Total number of task
#SBATCH --gres=gpu:4               # Number of gpus per node
#SBATCH --qos=gpus
#SBATCH --time=00:40:00
#SBATCH --output=output_gpu/test.log
#SBATCH --error=output_gpu/test.err
#SBATCH --exclusive                   

# Chargement de l'environnement
source ~/.bashrc
#source activate new_env

conda activate new_env
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1



# Execution with 2 GPUs

mpirun -np 2  python3  main_parallel_gpu.py
