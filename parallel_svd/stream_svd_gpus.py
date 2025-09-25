import numpy as np
import cupy as cp
from mpi4py import MPI
import time
from parallel_svd.base_parallel  import ParSVD_Base
import os

np.random.seed(10)
CWD = os.getcwd()

class HLL_SVD(SVD_Base):
    def __init__(self, K, ff, low_rank=False, results_dir='results'):
        super().__init__(K, ff, low_rank, results_dir)
      



    def initialize(self, A):
        """
        Initialize the svd computation, by computing the svd of A,
        we keep only U, and Sigma

        """
        gpu_id = self.rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(gpu_id).use()
        if not isinstance(A, cp.ndarray):
            A = cp.array(A)
        q, ulocal, self._singular_values = self.parallel_qr(A)
    
        self.ulocal = cp.matmul(q, ulocal)
        return self

    def incorporate_data(self, A):
        gpu_id = self.rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(gpu_id).use()
        cp.cuda.Device(gpu_id).synchronize()
        if not isinstance(A, cp.ndarray):
            A = cp.array(A)
        self._iteration += 1
        ll = self._ff * cp.matmul(self.ulocal, cp.diag(self._singular_values))
        ll = cp.concatenate((ll, A), axis=-1)
        qlocal, utemp, self._singular_values = self.parallel_qr(ll)
        self.ulocal = cp.matmul(qlocal, utemp)
        return self



    def parallel_qr(self, A):
        gpu_id = self.rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(gpu_id).synchronize()

        with cp.cuda.Device(gpu_id):
            q_gpu, r_gpu = cp.linalg.qr(A)
            r_local = cp.asnumpy(r_gpu)
            
            r_global = self.comm.gather(r_local, root=0)
            
            if self.rank == 0:
                r_global = np.concatenate(r_global, axis=0)
                r_global_gpu = cp.array(r_global)
                q_global_gpu, r_final_gpu = cp.linalg.qr(r_global_gpu)
                q_global_gpu = -q_global_gpu
                r_final_gpu = -r_final_gpu
                q_global_split = cp.split(q_global_gpu, self.nprocs)
                qlocal_gpu = cp.matmul(q_gpu, q_global_split[self.rank])
                if self._low_rank:
                    tt = MPI.Wtime()
                    unew_gpu, snew_gpu = self.low_rank_svd(r_final_gpu, self._K)
                else:
            
                    unew_gpu, snew_gpu, _ = cp.linalg.svd(r_final_gpu)
                    unew_gpu = unew_gpu[:, :self._K]
                    snew_gpu = snew_gpu[:self._K]
                unew = cp.asnumpy(unew_gpu)
                snew = cp.asnumpy(snew_gpu)
                
                for rank in range(1, self.nprocs):
                    self.comm.send(cp.asnumpy(q_global_split[rank]), dest=rank, tag=rank+10)
        
            else:
                
                q_global_recv = self.comm.recv(source=0, tag=self.rank+10)
                q_global_gpu_recv = cp.array(q_global_recv)
                qlocal_gpu = cp.matmul(q_gpu, q_global_gpu_recv)
            
                unew = None
                snew = None
            
            unew = self.comm.bcast(unew, root=0)
            snew = self.comm.bcast(snew, root=0)
        
            unew = cp.array(unew)
            snew = cp.array(snew)
            
            return qlocal_gpu, unew, snew

    def compute_vt(self, A):
        gpu_id = self.rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(gpu_id).use()
        if not isinstance(A, cp.ndarray):
            A = cp.array(A)
        n, m = A.shape
        S = cp.diag(self._singular_values)
        S = cp.linalg.inv(S)
        U = self.ulocal.T
        V = cp.matmul(cp.matmul(S, U), A)
        V = cp.asnumpy(V).astype(np.float64)
        tmp = np.zeros((self._K, m))
        self.comm.Allreduce(V, tmp, op=MPI.SUM)
        if self.rank == 0:
            self._Vt = np.hstack((self._Vt,tmp))
            return self._Vt




    def initialize_vt(self, A):
        gpu_id = self.rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(gpu_id).use()
        if not isinstance(A, cp.ndarray):
            A = cp.array(A)
        n, m = A.shape
        S = cp.diag(self._singular_values)
        S = cp.linalg.inv(S)
        U = self.ulocal.T
        V = cp.matmul(cp.matmul(S, U), A)
        V = cp.asnumpy(V).astype(np.float64)
        print("la taille de V",V.shape)
        tmp = np.zeros((self._K, m))
        print("le taille s de tmp",tmp.shape)

        self.comm.Allreduce(V, tmp, op=MPI.SUM)
        if self.rank == 0:
            self._Vt = tmp
            return self._Vt




    def low_rank_svd(self, A, K):
        """
        This method compute low rank approximation 
        using randomized SVD, on one GPU

        """
        A_gpu = A
        M, N = A_gpu.shape
        omega_gpu = cp.random.normal(size=(N, 2 * K))
        omega_pm_gpu = cp.matmul(A_gpu, cp.transpose(A_gpu))
        Y_gpu = cp.matmul(omega_pm_gpu, cp.matmul(A_gpu, omega_gpu))
        Qred_gpu, Rred_gpu = cp.linalg.qr(Y_gpu)
        B_gpu = cp.matmul(cp.transpose(Qred_gpu), A_gpu)
        ustar_gpu, snew_gpu, _ = cp.linalg.svd(B_gpu)
        unew_gpu = cp.matmul(Qred_gpu, ustar_gpu)
        unew_gpu = unew_gpu[:, :K]
        snew_gpu = snew_gpu[:K]
        return unew_gpu, snew_gpu




    def save(self):
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
        modes_global = self.comm.gather(self.ulocal, root=0)
        if self.rank == 0:
            self._modes = modes_global[0]
            for i in range(1, self.nprocs):
                self._modes = np.concatenate((self._modes, modes_global[i]), axis=0)


