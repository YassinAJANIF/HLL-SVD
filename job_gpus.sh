#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --nodes=1                  #Totla number of nodes
#SBATCH --gres=gpu:4               #Number of gpus per node
#SBATCH --constraint=h100          #use of gpus H100
#SBATCH --ntasks=4                 # Total number of task
#SBATCH --cpus-per-task=1             
#SBATCH --time=01:30:00
#SBATCH --output=output_gpu/test.log
#SBATCH --error=output_gpu/test.err
#SBATCH --exclusive                   

# Chargement de l'environnement
source ~/.bashrc
#source activate new_env

conda activate new_env                    # Change the env name
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1



# Execution with 4 GPUs

mpirun -np 4  python3  main_parallel_gpu.py
