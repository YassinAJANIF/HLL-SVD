#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a random matrix, then split it:
- into `num_batches` column-wise batches
- and for each batch, into `num_ranks` row-wise chunks ("ranks").
"""

from pathlib import Path
import numpy as np

# -----------------------------
# User parameters (edit here)
# -----------------------------
OUT_DIR      = Path(".")           # output directory (e.g., Path("data/data_parallel"))
NUM_BATCHES  = 3                   # number of column batches
NUM_RANKS    =4                    # number of row chunks per batch
N_ROWS       = 60_000_000
N_COLS       = 100
DTYPE        = np.float32
SEED         = 42                  # set to None for non-deterministic data

# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional reproducibility
    rng = np.random.default_rng(SEED) if SEED is not None else np.random.default_rng()

    # 1) Create the matrix (shape: N_ROWS x N_COLS) and save it
    matrix = rng.random((N_ROWS, N_COLS), dtype=DTYPE)
    np.save(OUT_DIR / "matrix.npy", matrix)

    # 2) Compute columns per batch (last batch may get the remainder)
    cols_per_batch = N_COLS // NUM_BATCHES

    for batch in range(NUM_BATCHES):
        # Column range for this batch
        start_col = batch * cols_per_batch
        end_col = (batch + 1) * cols_per_batch if batch != NUM_BATCHES - 1 else N_COLS

        # Extract column slice (view, no copy until needed)
        batch_data = matrix[:, start_col:end_col]

        # 3) Compute rows per rank (last rank may get the remainder)
        rows_per_rank = batch_data.shape[0] // NUM_RANKS

        for rank in range(NUM_RANKS):
            # Row range for this rank
            start_row = rank * rows_per_rank
            end_row = (rank + 1) * rows_per_rank if rank != NUM_RANKS - 1 else batch_data.shape[0]

            # Extract the rank slice and save
            rank_data = batch_data[start_row:end_row, :]
            out_path = OUT_DIR / f"points_rank_{rank}_batch_{batch}.npy"
            np.save(out_path, rank_data)

    print(
        f"Done. Split into {NUM_BATCHES} batches (columns) and {NUM_RANKS} ranks (rows) "
        f"and saved under: {OUT_DIR.resolve()}"
    )

if __name__ == "__main__":
    main()

