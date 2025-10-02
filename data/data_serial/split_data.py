#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serial utility to create a skinny random matrix and split it into column-wise batches.

- Generates a (rows x cols) float32 matrix.
- Saves the full matrix as `matrix.npy`.
- Splits columns into batches of size `batch_size`.
- Saves each batch as `Batch_{i}_data.npy`.

Notes:
- The last batch receives any leftover columns (if cols is not a multiple of batch_size).
- Set `seed=None` for non-deterministic data (different values at each run).
"""

from pathlib import Path
import numpy as np

# -----------------------------
# User parameters (edit here)
# -----------------------------
ROWS       = 60_000          # number of rows
COLS       = 30              # number of columns (skinny matrix)
BATCH_SIZE = 10              # columns per batch
DTYPE      = np.float32      # storage dtype
SEED       = 42              # set to None for non-deterministic data
OUTPUT_DIR = None            # None -> same folder as this script, or set e.g. "data/data_serial"


def create_and_split_matrix(rows: int,
                            cols: int,
                            batch_size: int,
                            output_dir: str | Path | None = None,
                            dtype=np.float32,
                            seed: int | None = 42) -> None:
    """
    Create a skinny random matrix (float32), split it into column batches, and save to .npy files.

    Parameters
    ----------
    rows : int
        Number of rows in the original matrix.
    cols : int
        Number of columns in the original matrix.
    batch_size : int
        Number of columns per batch.
    output_dir : str | Path | None
        Directory where .npy files are written. If None, uses the directory of this script.
    dtype : numpy dtype
        Data type used for storage (default: float32).
    seed : int | None
        RNG seed for reproducibility. Set to None for non-deterministic data.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("`rows` and `cols` must be positive integers.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")

    # Resolve output directory
    if output_dir is None:
        out_dir = Path(__file__).resolve().parent
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # RNG (reproducible if seed is an integer)
    rng = np.random.default_rng(seed)

    # 1) Create the matrix and save it
    matrix = rng.random((rows, cols), dtype=dtype)
    matrix_path = out_dir / "matrix.npy"
    np.save(matrix_path, matrix)

    # 2) Compute number of batches (ceil division)
    num_batches = (cols + batch_size - 1) // batch_size

    # 3) Split by columns and save each batch
    for i in range(num_batches):
        start_col = i * batch_size
        end_col = min(start_col + batch_size, cols)

        batch = matrix[:, start_col:end_col]                 # view/slice
        batch_path = out_dir / f"Batch_{i}_data.npy"
        np.save(batch_path, batch.astype(dtype, copy=False))  # ensure dtype without extra copy

        print(f"[SAVE] Batch {i}: cols[{start_col}:{end_col}) -> {batch_path.name} "
              f"(shape={batch.shape})")

    print(f"[DONE] Original matrix saved to {matrix_path.name} "
          f"(shape=({rows}, {cols}), dtype={dtype}))")
    print(f"[DONE] {num_batches} batches written to {out_dir}")


if __name__ == "__main__":
    create_and_split_matrix(
        rows=ROWS,
        cols=COLS,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
        dtype=DTYPE,
        seed=SEED,
    )

