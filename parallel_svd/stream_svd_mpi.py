import numpy as np
from mpi4py import MPI
import time
import os

from base_parallel import ParSVD_Base


np.random.seed(10)

# Current, parent, and file paths
CWD = os.getcwd()

class SVD_MPI(SVD_Base):

    """
    PyParSVD parallel class.

    :param int K: number of modes to truncate.
    :param int ff: forget factor.
    :param bool low_rank: if True, it uses a low-rank algorithm to speed up computations.
    :param str results_dir: if specified, it saves the results in `results_dir`. \
        Default save path is under a folder called `results` in the current working path.
    """

    def __init__(self, K, ff, low_rank=False, results_dir='results'):
        super().__init__(K, ff, low_rank, results_dir)
     #   self.comm = MPI.COMM_WORLD  # Initialize the MPI communicator
      #  self.rank = self.comm.Get_rank()  # Get the rank of the process
       # self.nprocs = self.comm.Get_size()  # Get the number of processes

    def initialize(self, A):
        """
        Initialize SVD computation with initial data using QR decomposition.
        
        :param ndarray/str A: initial data matrix
        """        
        # Perform parallel QR decomposition
        t1 = MPI.Wtime()
        q,ulocal, self._singular_values = self.parallel_qr(A)
        t2 = MPI.Wtime()
#        print("le temps d'execution de la decomposion QR en parallel est === ",t2-t1)
        t1 = MPI.Wtime()
        self.ulocal=np.matmul(q,ulocal)
        t2= MPI.Wtime()
 #       print("le temps d execution de produit au niveau de l'initialisation est ", t2-t1)
      #  self._gather_modes()

        return self

    def incorporate_data(self, A):
        """
        Incorporate new data in a streaming way for SVD computation.

        :param ndarray/str A: new data matrix.
        """
        self._iteration += 1
    
        ll = self._ff * np.matmul(self.ulocal, np.diag(self._singular_values))
        
      
        ll = np.concatenate((ll, A), axis=-1)
    
        qlocal, utemp, self._singular_values = self.parallel_qr(ll)
        
       
        self.ulocal = np.matmul(qlocal, utemp)
       

        return self

    def end_process(self, A):
        """
        Incorporate new data in a streaming way for SVD computation.

        :param ndarray/str A: new data matrix.
        """
        self._iteration += 1
        ll = self._ff * np.matmul(self.ulocal, np.diag(self._singular_values))              
        ll = np.concatenate((ll, A), axis=-1)
        qlocal, utemp, self._singular_values = self.parallel_qr(ll)
        self.ulocal = np.matmul(qlocal, utemp)
    
        self._gather_modes()
    

        return self





    def parallel_qr(self, A):
        """
        Perform parallel QR decomposition.

        :param ndarray A: data matrix
        :return: qlocal (local Q matrix), unew (updated U matrix), snew (singular values)
        """
        # Perform local QR
        num_rows, num_cols = A.shape
    
        q, r = np.linalg.qr(A)
    
        rlocal_shape_0 = r.shape[0]

        
        r_global = self.comm.gather(r, root=0)

        # Perform QR at rank 0:
        if self.rank == 0:
            temp = r_global[0]
            for i in range(self.nprocs-1):
                temp = np.concatenate((temp, r_global[i+1]), axis=0)
            r_global = temp

            qglobal, rfinal = np.linalg.qr(r_global)
            qglobal = -qglobal  # Trick for consistency
            rfinal = -rfinal

            # For this rank
            qlocal = np.matmul(q, qglobal[:rlocal_shape_0])

            # Send to other ranks
            for rank in range(1, self.nprocs):
                self.comm.send(qglobal[rank*rlocal_shape_0: (rank+1)*rlocal_shape_0], dest=rank, tag=rank+10)

            # Perform SVD on rfinal
            if self._low_rank:
                unew, snew = self.low_rank_svd(rfinal, self._K)  # Use self.low_rank_svd
            else:
                unew, snew, _ = np.linalg.svd(rfinal)  # Discard the third value (Vh)
                unew=unew[:, :self._K]
                snew=snew[:self._K]
        else:
            # Receive qglobal slices from rank 0
            qglobal = self.comm.recv(source=0, tag=self.rank+10)

            # For this rank
            qlocal = np.matmul(q, qglobal)

            # To receive new singular vectors
            unew = None
            snew = None

        unew = self.comm.bcast(unew, root=0)
        snew = self.comm.bcast(snew, root=0)

        return qlocal, unew, snew

    def low_rank_svd(self, A, K):
        """
        Performs randomized SVD.

        :param np.ndarray A: snapshot data matrix.
        :param int K: truncation.

        :return: singular values `unew` and `snew`.
        :rtype: np.ndarray, np.ndarray
        """
        M = A.shape[0]
        N = A.shape[1]

        omega = np.random.normal(size=(N, 2*K))
        omega_pm = np.matmul(A, np.transpose(A))
        Y = np.matmul(omega_pm, np.matmul(A, omega))

        Qred, Rred = np.linalg.qr(Y)

        B = np.matmul(np.transpose(Qred), A)
        ustar, snew, _ = np.linalg.svd(B)

        unew = np.matmul(Qred, ustar)

        unew = unew[:, :K]
        snew = snew[:K]

        return unew, snew

    



    def initialize_vt(self, A):
        
        
        
        n, m = A.shape
        S = np.diag(self._singular_values)
        S = np.linalg.inv(S)
        U = self.ulocal.T
        V = np.matmul(np.matmul(S, U), A).astype(np.float64)
        tmp = np.zeros((self._K, m))
        self.comm.Allreduce(V, tmp, op=MPI.SUM)
        if self.rank == 0:
            self._Vt =tmp
            return self._Vt



    def compute_vt(self, A):
        
      
        n, m = A.shape
        S = np.diag(self._singular_values)
        S = np.linalg.inv(S)
        U = self.ulocal.T
        V = np.matmul(np.matmul(S, U), A)
        tmp = np.zeros((self._K, m))
        self.comm.Allreduce(V, tmp, op=MPI.SUM)
        if self.rank == 0:
            self._Vt = np.hstack((self._Vt,tmp))
            return self._Vt



    def save(self):
        """
        Save data.
        """
        results_dir = os.path.join(CWD, self._results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        pathname_sv = os.path.join(results_dir, 'parallel_singular_values.npy')
        np.save(pathname_sv, self._singular_values)
        pathname_m = os.path.join(results_dir, 'parallel_POD.npy')

        if self.rank == 0:
            np.save(pathname_m, self._modes)

        self._singular_values = pathname_sv
        self._modes = pathname_m

    def _gather_modes(self):
        """
        Gather modes from all ranks.
        """
        modes_global = self.comm.gather(self.ulocal, root=0)
        if self.rank == 0:
            self._modes = modes_global[0]
            for i in range(self.nprocs-1):
                self._modes = np.concatenate((self._modes, modes_global[i+1]), axis=0)


