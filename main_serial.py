import os
import sys
import time
import numpy as np

# Add the working directory to the Python search pa


CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from serial_svd.serial_stream  import Serial_SVD


#path to data

path = os.path.join(CFD, './data/data_serial/')
#path = os.path.join('/home/yajanif@ec-nantes.fr/convert/pression/pyparsvd_200_test/tutorials/basic/data/parallel')

# Initialize the SVD class with parameters
SerSVD= Serial_SVD(K=10, ff=1.0)

# Serial data
initial_data_ser = np.load(os.path.join(path, 'Batch_0_data.npy'))
new_data_ser = np.load(os.path.join(path, 'Batch_1_data.npy'))


# Do first modal decomposition -- Serial
s = time.time()
SerSVD.initialize(initial_data_ser)

# Incorporate new data -- Serial
SerSVD.incorporate_data(new_data_ser)
#SerSVD.incorporate_data(newer_data_ser)


print('Elapsed time SERIAL: ', time.time() - s, 's.')

