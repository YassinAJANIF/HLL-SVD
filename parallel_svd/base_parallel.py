import numpy as np
from mpi4py import MPI

# Import custom Python packages
# import pyparsvd.postprocessing as post

# For shared memory deployment:
# `export OPENBLAS_NUM_THREADS=1`


class SVD_Base(object):
    """
    SVD_Base class. It implements data and methods shared
    across the derived classes(HLL_SVD, SVD_MPI, Dsvd).

    :param int K: number of modes to truncate.
    :param int ff: forget factor.
    :param bool low_rank: if True, it uses a low rank algorithm to speed up computations.
    
    """

    def __init__(self, K, ff, low_rank=False):
        self._K = K
        self._ff = ff
        self._low_rank = low_rank
        self._Vt = None
        self._iteration = 0

        # Initialize MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()
        self._nprocs = self.comm.Get_size()

    # --- Basic Getters ---

    @property
    def K(self):
        """Get the number of modes to truncate."""
        return self._K

    @property
    def ff(self):
        """Get the forget factor."""
        return self._ff

    @property
    def low_rank(self):
        """Get the low rank behaviour."""
        return self._low_rank

    @property
    def modes(self):
        """Get the modes."""
        if self.rank == 0:
            if isinstance(self._modes, np.ndarray):
                return self._modes
            elif isinstance(self._modes, str):
                return np.load(self._modes)
            else:
                raise TypeError("type,", type(self._modes), "not available")

    @property
    def singular_values(self):
        """Get the singular values."""
        if self.rank == 0:
            if isinstance(self._singular_values, np.ndarray):
                return self._singular_values
            elif isinstance(self._singular_values, str):
                return np.load(self._singular_values)
            else:
                raise TypeError("type,", type(self._singular_values), "not available")

    @property
    def iteration(self):
        """Get the number of data incorporation performed in the streaming data ingestion."""
        return self._iteration

    @property
    def n_modes(self):
        """Get the number of modes."""
        return self.singular_values.shape[-1]

    @property
    def comm(self):
        """Get the parallel MPI Communicator."""
        return self._comm

    @property
    def rank(self):
        """
        Get the parallel MPI Rank
        """
        return self._rank
    @property
    def Vt(self):

        return self._Vt

    @property
    def nprocs(self):
        """Get the number of processors."""
        return self._nprocs

    # --- Plotting Methods ---

    
