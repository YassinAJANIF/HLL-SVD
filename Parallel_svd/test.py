import numpy as np
import os
import sys
import time
import pytest
from mpi4py import MPI

# Import de la classe ParSVD_Parallel depuis ton script local
from paralle_stream_mpi import ParSVD_Parallel
from parallel_svd import Dsvd
# Chemin absolu vers les données
path = "/home/yaajanif/Mylibrary/Data/Data_parallel/"

# Création de l'instance ParSVD_Parallel avec K=10, facteur d'oubli ff=10
parallel = ParSVD_Parallel(K=10, ff=10)

# Chargement du fichier correspondant au rang MPI courant
#filename = os.path.join(path, f'points_rank_{parallel.rank}_batch_0.npy')


data0 = np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_0.npy')).astype(np.float64)
data1=  np.load(os.path.join(path, f'points_rank_{parallel.rank}_batch_1.npy')).astype(np.float64)
# Affichage du rang pour vérification
print("Rang MPI :", parallel.rank)

# Initialisation et incorporation des données
t0=time.time()
parallel.initialize(data0)
parallel.incorporate_data(data1)
t1=time.time()

#parallel.incorporate_data(data1)



t2=time.time()
parallel.initialize_vt(data0)
parallel.compute_vt(data1)
t3=time.time()
print("le temps du calcul U ======",t1-t0)
print("le temps du calcul Vt======",t3-t2)

direct=Dsvd(K=10, ff=1.0)

tt=time.time()
direct.tsqr2_svd(data0)
print("le temps d'execution de tsqr est ", time.time()-tt)
