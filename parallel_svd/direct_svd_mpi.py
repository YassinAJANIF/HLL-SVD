import numpy as np
from mpi4py import MPI
import os, sys
# from utils import low_rank_svd, generate_right_vectors  # Assurez-vous d'importer ces fonctions

CWD = os.getcwd()
from base_parallel import SVD_Base



class Dsvd(SVD_Base):
    def __init__(self, K, ff, low_rank=False, results_dir='results'):
        super().__init__(K, ff, low_rank, results_dir)
  

  

    def tsqr1(self, A):
        """
        Tall-and-skinny QR decomposition followed by global reduction and optional low-rank SVD.
        """
        # Étape 1 : QR locale
        q, r = np.linalg.qr(A)
        rlocal_shape_0 = r.shape[0]

        # Étape 2 : rassemblement des R locaux
        r_global = self.comm.gather(r, root=0)

        if self.rank == 0:
            # Concaténation de R
            r_stack = np.concatenate(r_global, axis=0)
            qglobal, rfinal = np.linalg.qr(r_stack)
            qglobal = -qglobal  # Pour homogénéité
            rfinal = -rfinal

            # Calcul de Qlocal pour le rang 0
            qlocal = np.matmul(q, qglobal[:rlocal_shape_0])

            # Envoi des blocs de Qglobal aux autres rangs
            for rank in range(1, self.nprocs):
                start = rank * rlocal_shape_0
                end = (rank + 1) * rlocal_shape_0
                self.comm.send(qglobal[start:end], dest=rank, tag=rank + 10)

            # Étape SVD (Levy-Lindenbaum)
            if self._low_rank:
                unew, snew, vt = low_rank_svd(rfinal, self._K)
            else:
                unew, snew, vt = np.linalg.svd(rfinal, full_matrices=False)
        else:
            qglobal = self.comm.recv(source=0, tag=self.rank + 10)
            qlocal = np.matmul(q, qglobal)
            unew = None
            snew = None
            vt = None

        # Broadcast à tous les rangs
        unew = self.comm.bcast(unew, root=0)
        snew = self.comm.bcast(snew, root=0)
        vt = self.comm.bcast(vt, root=0)

        Ui = np.matmul(qlocal, unew)
        return Ui, snew, vt

 

    def tsqr1_svd(self, A):
        """
        Computing the SVD using the tsqr1 algorithm
        """
        Ui, S, V = self.tsqr1(A)
        return Ui, S, V





    def tsqr1_randomized(self, A, n_iter):
        """
        Perform parallel randomized  QR decomposition.

        :param ndarray A: data matrix
        :return: qlocal (local Q matrix), unew (updated U matrix), snew (singular values)
        """
        # Perform local QR
        num_rows, num_cols = A.shape        
        np.random.seed(42)      
        omega = np.random.normal(size=(num_cols,self._K))
      
        Y = np.matmul(A, omega)
        for _ in range(n_iter):
            Y = np.matmul(A, np.matmul(A.T, Y))  # Itérations pour renforcer l'orthogonalisation

        t4=MPI.Wtime()
        q, r = np.linalg.qr(Y)
        t5=MPI.Wtime()
        rlocal_shape_0 = r.shape[0]
        # Gather data at rank 0:
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
                unew, snew, vt = self.low_rank_svd(rfinal, self._K)  # Use self.low_rank_svd
            else:
                unew, snew, vt = np.linalg.svd(rfinal)  # Discard the third value (Vh)
                unew=unew[:,:self._K]
                snew=snew[:self._K]
        else:
            # Receive qglobal slices from rank 0
            qglobal = self.comm.recv(source=0, tag=self.rank+10)

            # For this rank
            qlocal = np.matmul(q, qglobal)

            # To receive new singular vectors
            unew = None
            snew = None
            vt = None

        unew = self.comm.bcast(unew, root=0)
        snew = self.comm.bcast(snew, root=0)
        vt = self.comm.bcast(vt, root=0)

        Ui = np.matmul(qlocal, unew)
        return Ui, snew, vt


    def tsqr1_svd_randomized(self, A, n_iter):
        """
        Effectue la SVD via TSQR sur la matrice A locale.
        """
        Ui, S, V = self.tsqr1_randomized(A,n_iter)
        return Ui, S, V




    def APMOS(self, A):
        """
        SVD by APMOS.
        """
        vlocal, slocal = generate_right_vectors(A, self._K)
        wlocal = np.matmul(vlocal, np.diag(slocal).T)
        wglobal = self.comm.gather(wlocal, root=0)

        if self.rank == 0:
            temp = np.concatenate(wglobal, axis=-1)
            if self._low_rank:
                x, s, _ = low_rank_svd(temp, self._K)
            else:
                x, s, _ = np.linalg.svd(temp, full_matrices=False)
        else:
            x = None
            s = None

        x = self.comm.bcast(x, root=0)
        s = self.comm.bcast(s, root=0)

        phi_local = [1.0 / s[i] * np.matmul(A, x[:, i:i+1]) for i in range(self._K)]
        temp = np.concatenate(phi_local, axis=-1)

        return temp, s[:self._K]




    def EVD(self, A):
        """
        SVD using the snapshot approach 
        """
        A = A.astype(np.float64)
        local_rows, n = A.shape
        local_ATA = np.dot(A.T, A).astype(np.float64)
        ATA = np.zeros((n, n), dtype=np.float64)

        self.comm.Allreduce(local_ATA, ATA, op=MPI.SUM)

        if self.rank == 0:
            eig_values, eig_vectors = np.linalg.eigh(ATA)
            idx = np.argsort(eig_values)[::-1]
            eig_values = eig_values[idx]
            eig_vectors = eig_vectors[:, idx]
            eig_values_truncated = eig_values[:self._K]
            V_truncated = eig_vectors[:, :self._K]
            singular_values_truncated = np.sqrt(eig_values_truncated)
        else:
            V_truncated = None
            singular_values_truncated = None

        V_truncated = self.comm.bcast(V_truncated, root=0)
        singular_values_truncated = self.comm.bcast(singular_values_truncated, root=0)
        U_local = np.dot(A, V_truncated) / singular_values_truncated

        return U_local, singular_values_truncated, V_truncated
