import numpy as np
import sys
import time
from mpi4py import MPI
import sys, os
import cupy as cp


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Parallel_svd')))
# Import de la classe ParSVD_Parallel depuis ton script local



from parallel_svd.stream_svd_gpus import HLL_SVD

#Path to the data
path = "data/data_parallel/"

# Creation of the ParSVD_Parallel instance with K=10 and forget factor ff=10.
parallel =HLL_SVD(K=10, ff=10)

#Load the data
data0 = np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_0.npy'))
data1=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy'))
data2=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy'))
data3=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy'))
data4=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy'))


# Initialization .
t1=MPI.Wtime()
parallel.initialize(data0)
#incorporate new data
parallel.incorporate_data(data1)
parallel.incorporate_data(data2)
parallel.incorporate_data(data3)
parallel.incorporate_data(data4)

t2=MPI.Wtime()
#gather the data 

parallel._gather_modes()
# start a second pass over the data to compute Vt

t3=MPI.Wtime()
parallel.initialize_vt(data0)
parallel.compute_vt(data1)
t4=MPI.Wtime()




if parallel.rank==0:
    print(f"Whole time to compute spatial modes is={t2-t1}")
    print(f"Whole  time to compute temporal modes is={t4-t3}")
    #get the spatial modes
    U=parallel._modes
    U=cp.asnumpy(U)
   
    #get the singular_values
    S=parallel._singular_values

    #get the temporal modes
    Vt=parallel.Vt
    
    #Save the singular values
    np.save("results/S.npy",S)

    #Save the spatial modes in the results directory
    np.save("results/U.npy",U)

    print("la taille de Vt est ===", Vt.shape)
    #Save the temporal modes
    np.save("results/Vt.npy",Vt)
   

