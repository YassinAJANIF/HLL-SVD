#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple HLL-SVD driver (GPU + MPI) for streaming SVD computation.

- First pass: initialize + incorporate_data  -> computes spatial modes (U) and singular values (S)
- Gather modes across ranks
- Second pass: initialize_vt + compute_vt    -> computes temporal modes (Vt)
- Saves U.npy, S.npy, Vt.npy into results/

Run (example):
    mpirun -np 4 python run_hll_svd.py
"""

import os
import numpy as np
from mpi4py import MPI
import cupy as cp

# If the package is local, ensure it can be imported.
# Adjust/remove this if parallel_svd is already on your PYTHONPATH.
try:
    from parallel_svd.stream_svd_gpus import HLL_SVD
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "parallel_svd")))
    from stream_svd_gpus import HLL_SVD  # type: ignore


# ----------------------------
# User settings
# ----------------------------
DATA_DIR    = "data/data_parallel"  # expects files: points_rank_{rank}_batch_{i}.npy
RESULTS_DIR = "results"
K           = 10                    # target rank
FORGET_FF   = 1.0                    # forget factor
BATCH_IDS   = [0, 1, 2, 3, 4]       # batches to process (edit as needed)


def to_numpy(x):
    """Return a NumPy array (move from GPU if x is a CuPy array)."""
    try:
        return cp.asnumpy(x)
    except Exception:
        return np.asarray(x)


def batch_path(rank: int, i: int) -> str:
    """Build the expected path for a given rank and batch index."""
    return os.path.join(DATA_DIR, f"points_rank_{rank}_batch_{i}.npy")


def main():
    # Create the solver (exposes MPI rank through hll.rank)
    hll = HLL_SVD(K=K, ff=FORGET_FF)
    rank = hll.rank

    # Ensure output directory exists (created by rank 0)
    if rank == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------
    # First pass: U and S (streaming)
    # ----------------------------
    t1 = MPI.Wtime()

    # Load first batch and initialize
    p0 = batch_path(rank, BATCH_IDS[0])
    if not os.path.exists(p0):
        if rank == 0:
            print(f"[ERROR] Missing batch file: {p0}")
        return
    data = np.load(p0, mmap_mode="r")  # mmap to minimize RAM usage
    hll.initialize(data)
    del data

    # Incorporate remaining batches one-by-one (keep only one in RAM)
    for i in BATCH_IDS[1:]:
        p = batch_path(rank, i)
        if not os.path.exists(p):
            if rank == 0:
                print(f"[WARNING] Skipping missing batch: {p}")
            continue
        data = np.load(p, mmap_mode="r")
        hll.incorporate_data(data)   # incorporate new data
        del data                     # free RAM before next batch

    t2 = MPI.Wtime()

    # Gather modes across ranks before computing Vt
    hll._gather_modes()

    # ----------------------------
    # Second pass: Vt (streaming)
    # ----------------------------
    t3 = MPI.Wtime()

    # Re-load first batch and initialize_vt
    data = np.load(p0, mmap_mode="r")
    hll.initialize_vt(data)
    del data

    # Process remaining batches for Vt
    for i in BATCH_IDS[1:]:
        p = batch_path(rank, i)
        if not os.path.exists(p):
            continue
        data = np.load(p, mmap_mode="r")
        hll.compute_vt(data)
        del data

    t4 = MPI.Wtime()

    # ----------------------------
    # Save results (rank 0)
    # ----------------------------
    if rank == 0:
        print(f"[TIME] Spatial pass (U,S): {t2 - t1:.6f} s")
        print(f"[TIME] Temporal pass (Vt): {t4 - t3:.6f} s")

        U  = to_numpy(hll._modes)            # spatial modes
        S  = to_numpy(hll._singular_values)  # singular values
        Vt = to_numpy(hll.Vt)                # temporal modes

        np.save(os.path.join(RESULTS_DIR, "U.npy"),  U)
        np.save(os.path.join(RESULTS_DIR, "S.npy"),  S)   # Save the singular values
        np.save(os.path.join(RESULTS_DIR, "Vt.npy"), Vt)

        print(f"[SAVE] U  -> {RESULTS_DIR}/U.npy,  shape={U.shape}")
        print(f"[SAVE] S  -> {RESULTS_DIR}/S.npy,  shape={S.shape}")
        print(f"[SAVE] Vt -> {RESULTS_DIR}/Vt.npy, shape={Vt.shape}")


if __name__ == "__main__":
    main()

