#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from mpi4py import MPI
import h5py
import posixpath as pp

# -------------------- Paramètres --------------------
OUT_DIR         = "/home/yaajanif/New_library/Data/Data_parallel_io"  # = DATA_DIR de ton lecteur
NBATCH          = 20
NROWS           = 4000           # ndof (lignes)
NCOLS           = 25             # colonnes (snapshots par batch)
BASE_SEED       = 1234           # reproductible; changé par batch
DATASET_PATH    = "Prsr_d"       # dataset principal (ex: "Prsr_d" ou "Solution/Prsr_d")
VARY_DATASET    = False          # True => crée des variantes de nom/chemin pour tester ton resolver
USE_CHUNKING    = True           # True => dset chunké (sans compression en MPI)
CHUNK_ROWS      = 512            # taille de chunk en lignes

# -------------------- MPI ---------------------------
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

# Détecte si h5py supporte MPI; si oui et nprocs>1 on écrira en parallèle
H5_MPI = bool(getattr(h5py.get_config(), "mpi", False)) and nprocs > 1

# -------------------- Utils HDF5 --------------------
def _row_slice(n_rows: int, size: int, r: int):
    """Retourne [start, stop) pour une répartition équilibrée des lignes."""
    base = n_rows // size
    rem  = n_rows %  size
    start = r * base + min(r, rem)
    stop  = start + base + (1 if r < rem else 0)
    return start, stop

def _ensure_parent_groups(h5file, dataset_path: str):
    """Crée les groupes parents si DATASET_PATH contient des sous-groupes (ex: 'Solution/Prsr_d')."""
    path = dataset_path.lstrip("/")
    if "/" not in path:
        return  # top-level
    parent = pp.dirname(path)
    # Crée récursivement les groupes manquants
    cur = h5file
    for part in parent.split("/"):
        if part not in cur:
            cur = cur.create_group(part)
        else:
            cur = cur[part]

def _dataset_path_for_batch(i: int) -> str:
    """Optionnel: varie le chemin du dataset pour tester ton resolver."""
    if not VARY_DATASET:
        return DATASET_PATH
    variants = [
        "Prsr_d",
        "prsr_d",
        "Solution/Prsr_d",
        "Data/Pressure",
    ]
    return variants[i % len(variants)]

# -------------------- Génération --------------------
def make_random_block(shape, seed):
    """Génère un bloc (lignes locales × NCOLS) pseudo-aléatoire reproductible."""
    rng = np.random.default_rng(seed)
    # Exemple: bruit gaussien + légère tendance par colonne pour différencier les batches
    block = rng.normal(loc=0.0, scale=1.0, size=shape).astype("f4")
    # petite structure pour réalisme (décroissance par colonne)
    cols = np.linspace(1.0, 0.5, shape[1], dtype="f4")
    block *= cols
    return block

def write_one_file(path, nrows, ncols, batch_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Driver HDF5
    if H5_MPI:
        f = h5py.File(path, "w", driver="mpio", comm=comm)
    else:
        # écriture série (seul rank 0 écrit effectivement)
        if rank == 0:
            f = h5py.File(path, "w")
        else:
            return  # autres ranks ne font rien en mode série

    with f:
        dspath = _dataset_path_for_batch(batch_id)
        _ensure_parent_groups(f, dspath)

        # Définition du dataset
        dkwargs = dict(shape=(nrows, ncols), dtype="f4")
        # Chunking ok; éviter la compression en écriture mpio (souvent non supportée)
        if USE_CHUNKING:
            dkwargs["chunks"] = (min(CHUNK_ROWS, nrows), ncols)

        dset = f.create_dataset(dspath, **dkwargs)

        # Attributs optionnels (métadonnées légères)
        dset.attrs["FileType"] = "Solution"
        dset.attrs["Variables"] = "Prsr_d"
        dset.attrs["BatchID"] = batch_id

        if H5_MPI:
            # Ecriture parallèle: chaque rang écrit sa tranche de lignes
            i0, i1 = _row_slice(nrows, nprocs, rank)
            local_rows = i1 - i0
            data_local = make_random_block((local_rows, ncols), BASE_SEED + batch_id * 1000 + rank)
            dset[i0:i1, :] = data_local
            # barrière implicite à la fermeture
        else:
            # Ecriture série: rank 0 écrit tout
            if rank == 0:
                full = make_random_block((nrows, ncols), BASE_SEED + batch_id * 1000)
                dset[:, :] = full

def main():
    if rank == 0:
        print(f"[INFO] Writing {NBATCH} HDF5 batches to: {OUT_DIR}")
        print(f"[INFO] MPI mode: {'parallel' if H5_MPI else 'serial'} | ranks={nprocs}")

    for i in range(NBATCH):
        filename = f"Batch_{i}.h5"
        path = os.path.join(OUT_DIR, filename)
        write_one_file(path, NROWS, NCOLS, i)

    if rank == 0:
        print("[OK] Done.")

if __name__ == "__main__":
    main()

