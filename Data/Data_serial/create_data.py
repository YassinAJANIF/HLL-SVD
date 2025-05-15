import numpy as np
import os

# Paras
rows=6000000   # Nombre de lignes
cols=30      # Nombre de colonnes (skinny matrix)
batch_size=10 # Taille de chaque batch en termes de colonnes

def create_and_split_matrix(rows, cols, batch_size, output_dir=None):
    """
    Cree une matrice "skinny" aleatoire en 32 bits, la divise en lots selon les colonnes, et sauvegarde les fichiers.

    :param rows: Nombre de lignes dans la matrice originale
    :param cols: Nombre de colonnes dans la matrice originale
    :param batch_size: Nombre de colonnes dans chaque lot (batch)
    :param output_dir: Repertoire ou enregistrer les fichiers .npy (par defaut, repertoire courant)
    """
    # Si le repertoire de sortie n'est pas fourni, utiliser le repertoire contenant le script
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))  # Repertoire courant

    # Creer le repertoire de sortie s'il n'existe pas
#    os.makedirs(os.path.join(output_dir, "Batch"), exist_ok=True)

    # Creer une matrice aleatoire en float32
    matrix = np.random.rand(rows, cols).astype(np.float32)                ###########format des nombres

    # Sauvegarder la matrice originale
    matrix_path = os.path.join(output_dir, "matrix.npy")
    np.save(matrix_path, matrix)

    # Diviser la matrice en lots selon les colonnes
    num_batches = cols // batch_size + (1 if cols % batch_size != 0 else 0)

    for i in range(num_batches):
        start_col = i * batch_size
        end_col = min(start_col + batch_size, cols)

        batch = matrix[:, start_col:end_col]
        batch_filename = os.path.join(output_dir, f"tmp/Batch_{i}_data.npy")
        np.save(batch_filename, batch.astype(np.float32))  # Conversion en float32 (redondante ici, mais sure)
        print(f"Batch {i} sauvegarde dans {batch_filename}")

    print(f"Matrice originale sauvegardee dans {matrix_path}")


# Appeler la fonction
create_and_split_matrix(rows, cols, batch_size)


