import os
import sys

import numpy as np

# Add the working directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the base class for ParSVD_Parallel
from serial_base import SVD_Base

# Define the current working directory
CWD = os.getcwd()

# Fix the random seed for reproducibility
np.random.seed(10)

class Serial_SVD(SVD_Base):
    """
    Serial_SVD  class for performing SVD computations in serial.

    :param int K: number of modes to truncate.
    :param int ff: forget factor.
    :param bool low_rank: if True, it uses a low-rank algorithm to speed up computations.
    :param str results_dir: if specified, it saves the results in `results_dir`. \
        Default save path is under a folder called `results` in the current working directory.
    """

    def __init__(self, K, ff, low_rank=False, results_dir='results'):
        # Call the constructor of the base class (ParSVD_Base)
        super().__init__(K, ff, low_rank, results_dir)

    def initialize(self, A):
        """
        Initialize SVD computation with the initial data matrix.

        :param ndarray A: initial data matrix.
        """
        # Step 1: Perform QR decomposition of the input matrix A
        q, r = np.linalg.qr(A)

        # Step 2: Perform SVD of the R matrix (r) and get singular values
        ui, self._singular_values, self.vit = np.linalg.svd(r)

        # Step 3: Multiply Q and U matrices, and truncate the results
        self._modes = np.matmul(q, ui)[:, :self._K]
        self._singular_values = self._singular_values[:self._K]

        return self

    def incorporate_data(self, A):
        """
        Incorporate new data for streaming SVD computation.

        :param ndarray A: new data matrix.
        """
        # Step 3(a): Compute M_ap = forget factor * U * Sigma
        m_ap = self._ff * np.matmul(self._modes, np.diag(self._singular_values))

        # Concatenate the new data matrix A to M_ap
        m_ap = np.concatenate((m_ap, A), axis=-1)

        # Step 3(b): Perform QR decomposition on the concatenated matrix
        udashi, ddashi = np.linalg.qr(m_ap)

        # Step 3(b): Perform SVD on the resulting matrix ddashi
        utildei, dtildei, vtildeti = np.linalg.svd(ddashi)

        # Step 3(c): Select the top K singular values and modes
        max_idx = np.argsort(dtildei)[::-1][:self._K]
        self._singular_values = dtildei[max_idx]
        utildei = utildei[:, max_idx]

        # Update the modes matrix
        self._modes = np.matmul(udashi, utildei)

        return self
        

