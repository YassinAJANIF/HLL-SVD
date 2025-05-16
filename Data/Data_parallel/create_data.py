import numpy as np

# Paramètres : nombre de batchs et nombre de ranks
num_batches =2 # Par exemple, diviser en 3 batchs
num_ranks = 2#Par exemple, 4 ranksi

# Créer une matrice aléatoire de taille 600000x100
matrix = np.random.rand(600_000,50).astype(np.float32)  #          (30000000,300)#la version ancienne  (6000000,100), meilleur config


# Sauvegarder la matrice complète dans un fichier
np.save('matrix.npy', matrix)

# Calculer le nombre de colonnes par batch
cols_per_batch = matrix.shape[1] // num_batches

# Diviser la matrice en batchs selon les colonnes
for batch in range(num_batches):
    # Calculer les indices des colonnes pour chaque batch
    start_col = batch * cols_per_batch
    end_col = (batch + 1) * cols_per_batch if batch != num_batches - 1 else matrix.shape[1]

    # Extraire le batch correspondant
    batch_data = matrix[:, start_col:end_col]

    # Calculer le nombre de lignes par rank pour ce batch
    rows_per_rank = batch_data.shape[0] // num_ranks

    # Diviser le batch en plusieurs ranks selon les lignes
    for rank in range(num_ranks):
        # Calculer les indices des lignes pour chaque rank
        start_row = rank * rows_per_rank
        end_row = (rank + 1) * rows_per_rank if rank != num_ranks - 1 else batch_data.shape[0]

        # Extraire les données de ce rank
        rank_data = batch_data[start_row:end_row, :]

        # Sauvegarder les données de ce rank et de ce batch dans un fichier distinct
        np.save(f'points_rank_{rank}_batch_{batch}.npy', rank_data)

print(f"Les matrices ont été divisées en {num_batches} batchs et {num_ranks} ranks, et sauvegardées avec succès.")
                       
