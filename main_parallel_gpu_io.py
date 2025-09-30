#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from mpi4py import MPI
import h5py

# --- MPI ---
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

# --- Paramètres I/O ---
DATA_DIR     = "/home/yaajanif/New_library/Data/Data_parallel_io"
NBATCH       = 20                 # ou None pour tout prendre
RESULTS_DIR  = "results"

# ---------- utilitaires HDF5 ----------
def _row_slice(n_rows: int, size: int, r: int):
    """Retourne [start, stop) pour une répartition équilibrée des lignes."""
    base = n_rows // size
    rem  = n_rows %  size
    start = r * base + min(r, rem)
    stop  = start + base + (1 if r < rem else 0)
    return start, stop

def _find_single_dataset_path(h5file):
    """Retourne le chemin de l'unique dataset du fichier. Erreur si 0 ou >1."""
    paths = []
    def _cb(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append("/" + name)  # h5py fournit des noms avec '/' comme séparateur
    h5file.visititems(_cb)

    if len(paths) == 1:
        return paths[0]
    if len(paths) == 0:
        raise KeyError("Aucun dataset trouvé dans ce fichier HDF5.")
    raise KeyError(
        "Plus d'un dataset trouvé :\n  - " + "\n  - ".join(paths) +
        "\nSpécifiez le dataset explicitement ou simplifiez vos fichiers."
    )

def load_h5_local_rows(path: str, comm, rank, nprocs):
    """Charge uniquement le bloc de lignes local de l'unique dataset HDF5 en mode MPI."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rank {rank}: fichier introuvable: {path}")

    with h5py.File(path, "r", driver="mpio", comm=comm) as f:
        ds_path = _find_single_dataset_path(f)
        dset = f[ds_path]
        if dset.ndim < 2:
            raise ValueError(f"Dataset '{ds_path}' n'est pas 2D (shape: {dset.shape}).")

        n_rows = dset.shape[0]
        i0, i1 = _row_slice(n_rows, nprocs, rank)
        if rank == 0:
            print(f"[INFO] Lecture dataset '{ds_path}' dans {os.path.basename(path)}")
        return dset[i0:i1, :]  # numpy array CPU

# ---------- main ----------
def main():
    # Liste des fichiers Batch_*.h5 (triés)
    all_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith("Batch_") and f.endswith(".h5")
    )
    if NBATCH is not None:
        all_files = all_files[:NBATCH]
    batch_paths = [os.path.join(DATA_DIR, f) for f in all_files]

    # Import paresseux (ta classe ROM)
    from parallel_svd.stream_svd_gpus import HLL_SVD
    parallel = HLL_SVD(K=50, ff=1.0)

    # ===== Pass 1 : Streaming pour U, S =====
    t1 = MPI.Wtime()
    for i, path in enumerate(batch_paths):
        data = load_h5_local_rows(path, comm, rank, nprocs)
        if i == 0:
            parallel.initialize(data)
        else:
            parallel.incorporate_data(data)
    t2 = MPI.Wtime()

    # Optionnel : rassemblement/globalisation des modes si nécessaire
    parallel._gather_modes()

    # ===== Pass 2 : Streaming pour Vt =====
    for i, path in enumerate(batch_paths):
        data = load_h5_local_rows(path, comm, rank, nprocs)
        if i == 0:
            parallel.initialize_vt(data)
        else:
            parallel.compute_vt(data)

    # ===== Récup & sauvegarde (rank 0) =====
    if rank == 0:
        print("Temps (Pass 1):", t2 - t1, "s")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        U  = parallel._modes
        S  = parallel._singular_values
        Vt = parallel.Vt
        print("U:", type(U), getattr(U, "shape", None))
        print("S:", type(S), getattr(S, "shape", None))
        print("Vt:", type(Vt), getattr(Vt, "shape", None))
        np.save(os.path.join(RESULTS_DIR, "U.npy"),  U)
        np.save(os.path.join(RESULTS_DIR, "S.npy"),  S)
        np.save(os.path.join(RESULTS_DIR, "Vt.npy"), Vt)

if __name__ == "__main__":
    main()

