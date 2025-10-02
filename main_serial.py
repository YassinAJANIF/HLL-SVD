#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Serial version of the Levy–Lindenbaum approach, using a streaming workflow (one batch at a time)
Steps:
1) Load the initial batch and initialize the SVD.
2) Load the next batch and incorporate it into the current SVD.
3) Print total elapsed time.

Notes:
- Paths are resolved relative to this file.
- Keep it simple: no extra dependencies, no complex path logic.
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------
# Make sure we can import the local package `serial_svd` from this folder
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR)

from serial_svd.serial_stream import Serial_SVD  # your serial SVD implementation

# ---------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------
DATA_DIR = os.path.join(THIS_DIR, "data", "data_serial")

# Initialize the serial SVD solver
# K  = target rank
# ff = forget factor
SerSVD = Serial_SVD(K=10, ff=1.0)

# ---------------------------------------------------------------------
# Load data batches (serial)
# ---------------------------------------------------------------------

initial_path = os.path.join(DATA_DIR, "Batch_0_data.npy")
new_path     = os.path.join(DATA_DIR, "Batch_1_data.npy")


# Load the initial and next batches
initial_data_ser = np.load(initial_path)
new_data_ser     = np.load(new_path)

# ---------------------------------------------------------------------
# Run the serial SVD in a streaming fashion
# ---------------------------------------------------------------------
start = time.time()

# First pass: initialize with the first batch
SerSVD.initialize(initial_data_ser)

# Incorporate new data (second batch)
SerSVD.incorporate_data(new_data_ser)
# SerSVD.incorporate_data(newer_data_ser)  # <- example if you have more batches

elapsed = time.time() - start
print(f"Elapsed time (SERIAL): {elapsed:.6f} s")

