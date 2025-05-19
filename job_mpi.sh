#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --cluster=nautilus
#SBATCH --time=00:20:00
#SBATCH --qos=long          
#SBATCH --partition=all     # partition name
#SBATHC --nodes=1           # number of nodes
#SBATCH  --ntasks=96
###SBATCH --nodelist=cnode706     #Specify the node name
#SBATCH --cpus-per-task=1    # Number of core per task
#SBATCH --output=output_mpi/test.log
#SBATCH --error=output_mpi/test.err
#SBATCH --exclusive
# Chargement de l'environnement
source ~/.bashrc
source activate new_env

# Limitation des threads par processus
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Exécution avec mpirun
mpirun --bind-to core --map-by core -np 96 python3 main_parallel_mpi.py
