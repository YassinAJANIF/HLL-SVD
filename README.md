# A new parallel singular value decomposition implementation for hybrid architectures  GPUs+CPUs

## Table of contents


  * [Description](#description)
  * [Installation](#Installation)
  * [Preprocessing](#Preprocessing)
  * [Testing](#Testing)
  * [References](#references)
  * [License](#license)




# Description
Our library provides a comprehensive set of methods for computing the Singular Value Decomposition (SVD), available in both sequential and parallel versions. Designed in a modular and well-structured manner, it ensures simple, flexible, and easily portable usage across various computing environments, including distributed architectures and GPU-accelerated systems. The library is organized as follows:

1. **Streaming SVD**: Our library provides the first implementation of the Levy and Lindenbaum([(Levy and Lindenbaum 1998)](#Levy-and-Lindenbaum-1998) approach on hybrid GPU-CPU architectures, using mpi4py to enable inter-process communication in an HPC environment, and CuPy to perform algebraic operations (QR decomposition, SVD, matrix products, etc.) on GPUs;

2. **Direct Parallel SVD**: Our library provides a variety of direct methods for computing the SVD in parallel, including the approximate snapshot method [(Wang et al 2016)](#Wang-et-al-2016), SVD via EVD, and SVD via the TSQR method [(Wang et al 2016)](#Wang-et-al-2016). These methods use mpi4py to enable communication between processes running on different cores.


3. **Serail SVD** The library also provides classical implementations of SVD in serial, including SVD via QR decomposition, SVD via EVD, and randomized SVD.

# Installation
Use the following command to install the library locally<br>
```bash
$ git clone "https://github.com/YassinAJANIF/New_library.git"
```
**Requirement**:
To run the library, you must have a Conda environment and libraries such as NumPy, CuPy, HDF5,mpi4py...., All requirement are in  the file "requirements.txt", to do so, please run the file:
```bash
python3 setup.py
```
# Preprocessing
To generate the data, there are two directories:
- **Data_parallel**: This repository contains the file **create_data.py**, which allows you to generate data for parallel tests, such as MPI (on CPU nodes) or CuPy (hybrid CPU/GPU nodes). To use it, simply specify the size of the matrix, the number of batches (num_batches), and the number of ranks (num_ranks), then execute the script.
- **Data_serial**: This repository includes the file **create_data.py**, which facilitates the generation of random data. You need to specify the number of columns and rows for the matrix. Once the matrix is created, you can also define the number of batches(num_batches) into which the matrix should be divided.
# Testing
- For the serial version: execute the python script main_serial.py.

```bash
$ python3 main_serial.py
```
  
- For the parallel version, if you are working in interactive mode, you can run the following command — for example, with 2 GPUs.
```bash 
$ export OPENBLAS_NUM_THREADS=1
```
```bash
$ mpirun -np 2 python3 main_parallel.py
```
Otherwise, you can run the code using the Slurm script job_gpus.sh: 

```bash
$ SBATCH  job_gpus.sh
```


## References

#### (Levy and Lindenbaum 1998) 
*Sequential Karhunen–Loeve Basis Extraction and its Application to Images.* [[DOI](https://ieeexplore.ieee.org/abstract/document/723422)]

#### (Wang et al 2016) 
*Approximate partitioned method of snapshots for POD.* [[DOI](https://www.sciencedirect.com/science/article/pii/S0377042715005774)]

#### (Benson et al 2013)
*Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures.* [[DOI](https://ieeexplore.ieee.org/document/6691583)]

#Lien utils
 

