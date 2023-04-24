import numpy as np
import pandas as pd

#Here is the test for the combine_tree_RGB_days
#tree_mosaic = combine_trees_RGB_days([[0,1,2,3],[4,5,6,7],[8,9,10,11]],df)

def combine_trees_RGB_days(tab_indexes,df):
    n_lig=len(tab_indexes)
    n_col=len(tab_indexes[0])
    D,X,Y,C=np.shape(df['array'][0])
    tab_out=np.zeros(([D,X*n_col,Y*n_lig,C]))
    for i in range(n_lig):
        for j in range(n_col):
            tab_out[:, j*X:(j+1)*X, i*Y:(i+1)*Y, :]=df['array'][tab_indexes[i][j]]
    return tab_out


def generate_tab_index(i, j):
    tab_index = []
    start = 0
    for row in range(i):
        end = start + j
        tab_index.append(list(range(start, end)))
        start = end
    return tab_index


def filter_data_to_export(data_rgb, data_info):
    """
    Retourne un nouveau dataframe contenant pour chaque arbre, un array pour tous ses mois dans l'ordre temporel 
    ainsi que le numéro de l'arbre et son jour de floraison

    Args:
    data_rgb : array 5D avec les valeurs RGB pour chaque arbre pour tous les mois
    data_info : dataframe avec les informations sur chaque arbre
    
    Returns:
    Un nouveau dataframe avec pour chaque arbre un array de tous ses mois dans l'ordre temporel,
    ainsi que son numéro et son jour de floraison
    """
    arbres = data_info['id'].tolist()
    jours_floraison = data_info['jourF'].tolist()
    jour_floraison_centre = data_info['jourF_centre'].tolist()
    genotypes = data_info['genot'].tolist()
    std = data_info['std'].tolist()
    mean = data_info['mean'].tolist()
    
    # Créer un dataframe vide pour stocker les données filtrées
    filtered_data = pd.DataFrame(columns=['array', 'id','genot','jourF','mean','jourF_centre','std'])
    
    # Parcourir chaque arbre et extraire ses données
    ordre_mois = [0, 3, 2, 1]  # Ordre temporel : juin, septembre, octobre, novembre
    for i, arbre in enumerate(arbres):
        mois_arbre = []
        for mois in ordre_mois:
            mois_arbre.append(data_rgb[mois,arbre,:,:,:])
        filtered_data = filtered_data.append({'array': np.array(mois_arbre),
                                              'id': arbre,
                                              'genot': genotypes[i],
                                              'jourF': jours_floraison[i],
                                              'mean': mean[i],
                                              'jourF_centre': jour_floraison_centre[i],
                                              'std': std[i]},
                                             ignore_index=True)
    return filtered_data
