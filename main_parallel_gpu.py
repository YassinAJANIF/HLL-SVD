import numpy as np
import sys
import time
import pytest
from mpi4py import MPI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Parallel_svd')))
# Import de la classe ParSVD_Parallel depuis ton script local


#from Parallel_svd.base_parallel import ParSVD_Base
from parallel_svd.stream_svd_gpus import ParSVD_Parallel



#Path to the data
path = "/home/yaajanif/Mylibrary/Data/Data_parallel/"

# Creation of the ParSVD_Parallel instance with K=10 and forget factor ff=10.
parallel = ParSVD_Parallel(K=10, ff=10)



#Load the data
data0 = np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_0.npy')).astype(np.float64)
data1=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy')).astype(np.float64)




# Initialisation et incorporation des données.
t1=MPI.Wtime()
parallel.initialize(data0)
parallel.incorporate_data(data1)
t2=MPI.Wtime()


# start a second pass over the data to compute Vt
t3=MPI.Wtime()
parallel.initialize_vt(data0)
parallel.compute_vt(data1)
t4=MPI.Wtime()


#Desplay the results on rank 0
if parallel.rank==0:
    print("Whole time to compute U is===",t2-t1)
    print("Whole  time to compute Vt is==", t4-t3)
