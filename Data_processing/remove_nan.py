import pandas as pd
import os

################ DEFINITION OF PATHS

import_path = "/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/avec NA/non concat/equilibres"
export_path = "/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/sans NA/non concat/equilibres"


############### FONCTION

# Problème avec les colonnes Unnamed
def remove_unnamed_column(directory):
    # Pour chaque fichier dans le dossier
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Lecture du fichier CSV
            df = pd.read_csv(os.path.join(directory, filename))
            
            # Vérification et suppression de la colonne Unnamed si elle existe
            if 'Unnamed: 0' in df.columns:
                df.drop(columns='Unnamed: 0', inplace=True)
            
            # Enregistrement du fichier modifié dans le même dossier
            df.to_csv(os.path.join(directory, filename), index=False)
            
remove_unnamed_column(import_path)


def process_files(import_path,export_path):
    # Pour chaque fichier dans le dossier
    for filename in os.listdir(import_path):
        if filename.endswith(".csv"):
            # Lecture des fichiers X et Y
            df_x = pd.read_csv(os.path.join(import_path, filename))
            df_y = pd.read_csv(os.path.join(import_path, filename.replace("X", "Y")))
            df_y = df_y.rename(columns={'x': 'JourF'}) #Si la colonne s'appelle x on la renomme JourF
            
            # Fusion des dataframes X et Y sur l'indice de ligne
            df = pd.merge(df_x, df_y, left_index=True, right_index=True)
            
            # Suppression des lignes contenant des valeurs manquantes dans les données Y
            df = df.dropna(subset=['JourF'])
            
            # Définition de N le nombre de colonnes de X
            N = len(df_x.columns)
            
            # Sélection des colonnes correspondantes pour X et Y
            df_X = df.iloc[:, 0:N]   # Remplacez N par le numéro de la dernière colonne de X
            df_Y = df.iloc[:, N:N+1] # Remplacez N par le numéro de la première colonne de Y
            
            # Enregistrement des dataframes X et Y en CSV
            df_X.to_csv(os.path.join(export_path, filename.replace("X", "X_out")), index=False)
            df_Y.to_csv(os.path.join(export_path, filename.replace("X", "Y_out")), index=False)
            
process_files(import_path, export_path)
