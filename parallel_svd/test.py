import numpy as np
import sys
import time
import pytest
from mpi4py import MPI
import sys, os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Parallel_svd')))
# Import de la classe ParSVD_Parallel depuis ton script local


#from Parallel_svd.base_parallel import ParSVD_Base
from Parallel_svd.parallel_svd import Dsvd



#Path to the data
path = "/home/yaajanif/Mylibrary/Data/Data_parallel/"

# Creation of the ParSVD_Parallel instance with K=10 and forget factor ff=10.
parallel = Dsvd(K=10, ff=10)



#Load the data
data0 = np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_0.npy')).astype(np.float64)
data1=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy')).astype(np.float64)

