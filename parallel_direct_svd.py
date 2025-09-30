#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from mpi4py import MPI

from base_parallel import SVD_Base


class Dsvd(SVD_Base):
    """
    Distributed SVD algorithms on top of mpi4py.

    This class provides multiple SVD variants designed for tall-and-skinny (m >> n)
    matrices that are row-partitioned across MPI ranks:

      - SVD_TSQR   : TSQR (Tall-and-Skinny QR) + final SVD on the stacked R.
      - RSVD_TSQR  : Randomized range finder + TSQR + final SVD.
      - SVD_APMOS  : APMOS-style right-vector construction then global SVD.
      - SVD_EVD    : Snapshot method via eigen-decomposition of A^T A.

    Notes
    -----
    * We assume each rank holds a local block A_i with the same number of columns.
    * All collective operations use self.comm (provided by SVD_Base).
    * Optional helper functions can be injected at init:
        - low_rank_svd_fn(rfinal, k) -> (U, s, Vt)
        - generate_right_vectors_fn(A, k) -> (V_local, s_local)
    """

    def __init__(
        self,
        K: int,
        ff,
        low_rank: bool = False,
        results_dir: str = "results",
        low_rank_svd_fn=None,
        generate_right_vectors_fn=None,
    ):
        """
        Parameters
        ----------
        K : int
            Target rank (number of singular values/vectors to return).
        ff : Any
            Kept for compatibility with SVD_Base signature.
        low_rank : bool, optional
            If True, use the injected low-rank SVD routine where applicable.
        results_dir : str, optional
            Directory used by the base class for outputs.
        low_rank_svd_fn : callable or None
            Custom function for truncated SVD on a small core matrix (R_final).
        generate_right_vectors_fn : callable or None
            Custom function for APMOS to build right vectors locally.
        """
        super().__init__(K, ff, low_rank, results_dir)
        self._low_rank_svd_fn = low_rank_svd_fn
        self._gen_right_vecs_fn = generate_right_vectors_fn

    # -------------------------------------------------------------------------
    # TSQR-based SVD
    # -------------------------------------------------------------------------
    def SVD_TSQR(self, A: np.ndarray):
        """
        Compute SVD via TSQR:
          1) Local QR: A_i = Q_i R_i
          2) Gather all R_i to rank 0 and stack -> R_stack
          3) QR(R_stack) = Q_g R_final
          4) SVD(R_final) = U_new Σ V^T
          5) U_i = Q_i @ U_new

        Returns
        -------
        U_local : np.ndarray, shape (m_i, K or n)
        s       : np.ndarray, shape (K or n,)
        Vt      : np.ndarray, shape (K or n, n)
        """
        # Local QR
        Q_i, R_i = np.linalg.qr(A, mode="reduced")
        r_rows = R_i.shape[0]  # equals n (number of columns of A)

        # Gather local R blocks on root
        R_list = self.comm.gather(R_i, root=0)

        if self.rank == 0:
            # Stack all R blocks by rows
            R_stack = np.concatenate(R_list, axis=0)

            # Second-stage QR
            Q_g, R_final = np.linalg.qr(R_stack, mode="reduced")

            # Optional sign flip for deterministic sign convention
            Q_g = -Q_g
            R_final = -R_final

            # Slice Q_g per rank (each slice has r_rows rows)
            Q_slice_0 = Q_g[:r_rows]
            Q_slices = [Q_slice_0] + [
                Q_g[i * r_rows : (i + 1) * r_rows] for i in range(1, self.nprocs)
            ]

            # Local projection on root
            Qlocal = Q_i @ Q_slices[0]

            # Send the appropriate slice to each non-root rank
            for r in range(1, self.nprocs):
                self.comm.send(Q_slices[r], dest=r, tag=100 + r)

            # Final SVD on the small core
            if self._low_rank and self._low_rank_svd_fn is not None:
                U_new, s, Vt = self._low_rank_svd_fn(R_final, self._K)
            else:
                U_full, s_full, Vt_full = np.linalg.svd(R_final, full_matrices=False)
                U_new = U_full[:, : self._K] if self._K else U_full
                s = s_full[: self._K] if self._K else s_full
                Vt = Vt_full[: self._K] if self._K else Vt_full
        else:
            Q_g_slice = self.comm.recv(source=0, tag=100 + self.rank)
            Qlocal = Q_i @ Q_g_slice
            U_new = s = Vt = None

        # Broadcast SVD factors to all ranks
        U_new = self.comm.bcast(U_new, root=0)
        s = self.comm.bcast(s, root=0)
        Vt = self.comm.bcast(Vt, root=0)

        # Build local left singular vectors
        U_local = Qlocal @ U_new
        return U_local, s, Vt

    # -------------------------------------------------------------------------
    # Randomized SVD + TSQR
    # -------------------------------------------------------------------------
    def RSVD_TSQR(self, A: np.ndarray, n_iter: int = 1, seed: int = 42):
        """
        Randomized SVD using a power iteration, followed by TSQR on the sketch.

        Steps
        -----
        1) Draw random test matrix Ω (n x K).
        2) Form Y = A Ω and (optionally) apply n_iter power iterations:
           Y ← A (A^T Y) to improve subspace capture.
        3) Local QR on Y_i; gather and do the TSQR reduction.
        4) SVD on the reduced core; map back to U_i.

        Parameters
        ----------
        A : np.ndarray
            Local block (m_i x n), with common n across ranks.
        n_iter : int
            Number of power iterations (>= 0).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        U_local, s, Vt
        """
        m_i, n = A.shape
        rng = np.random.default_rng(seed)
        Omega = rng.normal(size=(n, self._K))

        # Randomized range finder with optional power iterations
        Y = A @ Omega
        for _ in range(max(0, n_iter)):
            Y = A @ (A.T @ Y)

        # Local QR on the sketch
        Q_i, R_i = np.linalg.qr(Y, mode="reduced")
        r_rows = R_i.shape[0]  # should be K

        # Gather R_i to root
        R_list = self.comm.gather(R_i, root=0)

        if self.rank == 0:
            R_stack = np.concatenate(R_list, axis=0)
            Q_g, R_final = np.linalg.qr(R_stack, mode="reduced")

            # Optional sign flip for determinism
            Q_g = -Q_g
            R_final = -R_final

            # Slice Q_g by rank and project local subspace
            Q_slice_0 = Q_g[:r_rows]
            Q_slices = [Q_slice_0] + [
                Q_g[i * r_rows : (i + 1) * r_rows] for i in range(1, self.nprocs)
            ]
            Qlocal = Q_i @ Q_slices[0]

            for r in range(1, self.nprocs):
                self.comm.send(Q_slices[r], dest=r, tag=200 + r)

            # Core SVD (truncated if K is set)
            if self._low_rank and self._low_rank_svd_fn is not None:
                U_new, s, Vt = self._low_rank_svd_fn(R_final, self._K)
            else:
                U_full, s_full, Vt_full = np.linalg.svd(R_final, full_matrices=False)
                U_new = U_full[:, : self._K] if self._K else U_full
                s = s_full[: self._K] if self._K else s_full
                Vt = Vt_full[: self._K] if self._K else Vt_full
        else:
            Q_g_slice = self.comm.recv(source=0, tag=200 + self.rank)
            Qlocal = Q_i @ Q_g_slice
            U_new = s = Vt = None

        U_new = self.comm.bcast(U_new, root=0)
        s = self.comm.bcast(s, root=0)
        Vt = self.comm.bcast(Vt, root=0)

        U_local = Qlocal @ U_new
        return U_local, s, Vt

    # -------------------------------------------------------------------------
    # APMOS-style SVD
    # -------------------------------------------------------------------------
    def SVD_APMOS(self, A: np.ndarray):
        """
        APMOS variant: build (local) right vectors, reduce globally, then SVD.

        Requires
        --------
        An injected callable `generate_right_vectors_fn(A, K)` returning:
           V_local : (n x K)
           s_local : (K,)
        """
        if self._gen_right_vecs_fn is None:
            raise RuntimeError("generate_right_vectors_fn must be provided for SVD_APMOS.")

        V_local, s_local = self._gen_right_vecs_fn(A, self._K)  # (n x K), (K,)
        # Weight local contribution
        W_local = V_local * s_local[None, :]

        # Gather all W_local on root (as a list of (n x K))
        W_list = self.comm.gather(W_local, root=0)

        if self.rank == 0:
            W_stack = np.concatenate(W_list, axis=1)  # (n x (K * nprocs))

            if self._low_rank and self._low_rank_svd_fn is not None:
                X, s, _ = self._low_rank_svd_fn(W_stack, self._K)
            else:
                X_full, s_full, _ = np.linalg.svd(W_stack, full_matrices=False)
                X = X_full[:, : self._K]
                s = s_full[: self._K]
        else:
            X = None
            s = None

        X = self.comm.bcast(X, root=0)
        s = self.comm.bcast(s, root=0)

        # Build local left singular vectors: phi_i = A_i X Σ^{-1}
        # Compute column-wise: phi[:, j] = (1/s[j]) * A @ X[:, j]
        Phi_cols = [(A @ X[:, j : j + 1]) * (1.0 / s[j]) for j in range(self._K)]
        U_local = np.concatenate(Phi_cols, axis=1)
        return U_local, s

    # -------------------------------------------------------------------------
    # Snapshot SVD (EVD of A^T A)
    # -------------------------------------------------------------------------
    def SVD_EVD(self, A: np.ndarray):
        """
        Snapshot SVD via eigen-decomposition of G = A^T A.

        Steps
        -----
        1) Each rank forms G_i = A_i^T A_i (n x n).
        2) Allreduce sum to get G.
        3) EVD on root: G = V Λ V^T (Λ >= 0).
        4) Σ = sqrt(Λ), take top-K.
        5) U_i = A_i V_K Σ^{-1}.

        Returns
        -------
        U_local : (m_i x K)
        s       : (K,)
        V       : (n x K)
        """
        A = A.astype(np.float64, copy=False)
        _, n = A.shape

        G_local = (A.T @ A).astype(np.float64, copy=False)
        G = np.zeros((n, n), dtype=np.float64)
        self.comm.Allreduce(G_local, G, op=MPI.SUM)

        if self.rank == 0:
            evals, evecs = np.linalg.eigh(G)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            s = np.sqrt(np.maximum(evals[: self._K], 0.0))
            V = evecs[:, : self._K]
        else:
            V = None
            s = None

        V = self.comm.bcast(V, root=0)
        s = self.comm.bcast(s, root=0)

        # Avoid division with broadcasting surprises
        inv_s = np.where(s > 0, 1.0 / s, 0.0)
        U_local = (A @ V) * inv_s[None, :]
        return U_local, s, V

