import numpy as np
from mpi4py import MPI

# For shared memory deployment:
# `export OPENBLAS_NUM_THREADS=1`


class SVD_Base(object):
    """
    PyParSVD base class. It implements data and methods shared
    across the derived classes.

    :param int K: number of modes to truncate.
    :param int ff: forget factor.
    :param bool low_rank: if True, it uses a low rank algorithm to speed up computations.
    :param str results_dir: if specified, it saves the results in `results_dir`.
    """

    def __init__(self, K, ff=1.0, low_rank=False, results_dir='results'):
        self._K = K
        self._ff = ff
        self._low_rank = low_rank
        self._results_dir = results_dir
        self._iteration = 0

        

    # --- Basic Getters ---

    @property
    def K(self):
        return self._K

    @property
    def ff(self):
        return self._ff

    @property
    def low_rank(self):
        return self._low_rank

    @property
    def modes(self):
        if self.rank == 0:
            if isinstance(self._modes, np.ndarray):
                return self._modes
            elif isinstance(self._modes, str):
                return np.load(self._modes)
            else:
                raise TypeError("type,", type(self._modes), "not available")

    @property
    def singular_values(self):
        if self.rank == 0:
            if isinstance(self._singular_values, np.ndarray):
                return self._singular_values
            elif isinstance(self._singular_values, str):
                return np.load(self._singular_values)
            else:
                raise TypeError("type,", type(self._singular_values), "not available")

    @property
    def iteration(self):
        return self._iteration

    @property
    def n_modes(self):
        return self.singular_values.shape[-1]

    @property
    def comm(self):
        return self._comm

    @property
    def rank(self):
        return self._rank



    # --- SVD methods ---

    def svd_via_evd(self, A):
        """
        Truncated SVD via EVD using A.T @ A or A @ A.T (NumPy only).
        Only the top-K modes are returned.
        """
        print(f"[INFO] SVD via EVD decomposition (truncated to K={self.K})")
        self.A = A
        self.m, self.n = A.shape
        K = self.K

        if self.m >= self.n:
            ATA = self.A.T @ self.A
            eigvals, V = np.linalg.eigh(ATA)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx][:K]
            V = V[:, idx][:, :K]
            S = np.sqrt(np.clip(eigvals, 0, None))
            U = self.A @ V / S
        else:
            AAT = self.A @ self.A.T
            eigvals, U = np.linalg.eigh(AAT)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx][:K]
            U = U[:, idx][:, :K]
            S = np.sqrt(np.clip(eigvals, 0, None))
            V = self.A.T @ U / S
            V = V.T

        return U, S, V

    def svd_randomized(self, A):
        """
        Randomized SVD approximation using only NumPy.
        """
        n_iter = 2
        k = 10
        n, m = A.shape
        print(f"[INFO] Randomized SVD (k={k}, n_iter={n_iter})")

        Omega = np.random.randn(n, k)
        Y = A @ Omega

        for _ in range(n_iter):
            Y = A @ (A.T @ Y)

        Q, _ = np.linalg.qr(Y)
        B = Q.T @ A
        U_hat, S, VT = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat
        return U, S, VT

    # --- Plotting methods ---

    def plot_singular_values(self, idxs=[0], title='', figsize=(12, 8), filename=None):
        post.plot_singular_values(
            self.singular_values,
            title=title,
            figsize=figsize,
            path=self._results_dir,
            filename=filename,
            rank=self.rank
        )

    def plot_1D_modes(self, idxs=[0], title='', figsize=(12, 8), filename=None):
        post.plot_1D_modes(
            self.modes,
            idxs=idxs,
            title=title,
            figsize=figsize,
            path=self._results_dir,
            filename=filename,
            rank=self.rank
        )

    def plot_2D_modes(self, num_rows, num_cols, idxs=[0], title='', figsize=(12, 8), filename=None):
        post.plot_2D_modes(
            self.modes,
            num_rows,
            num_cols,
            self._nprocs,
            idxs=idxs,
            title=title,
            figsize=figsize,
            path=self._results_dir,
            filename=filename,
            rank=self.rank
        )
