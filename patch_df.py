import os
import numpy as np
import matplotlib.pyplot as plt
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
    genotypes = data_info['genot'].tolist()
    std = data_info['std'].tolist()
    
    # Créer un dataframe vide pour stocker les données filtrées
    filtered_data = pd.DataFrame(columns=['array', 'id', 'jourF','genot','std'])
    
    # Parcourir chaque arbre et extraire ses données
    ordre_mois = [0, 3, 2, 1]  # Ordre temporel : juin, septembre, octobre, novembre
    for i, arbre in enumerate(arbres):
        mois_arbre = []
        for mois in ordre_mois:
            mois_arbre.append(data_rgb[mois,i,:,:,:])
        filtered_data = filtered_data.append({'array': np.array(mois_arbre),
                                              'id': arbre,
                                              'jourF': jours_floraison[i],
                                              'genot': genotypes[i],
                                              'std': std[i]},
                                             ignore_index=True)
    return filtered_data


# 0 = juin, 1 = novembre, 2 = octobre, 3 = septembre ==> 0,3,2,1
def time_plot(data_to_plot, num_arbre, data_info):
    jour_f = data_info.loc[num_arbre, 'jourF']  # obtenir le jour de floraison de l'arbre
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(data_to_plot[0, num_arbre])
    axs[0].set_title(f"Juin - Arbre {num_arbre} - Jour F: {jour_f}")
    axs[0].axis('off')
    axs[1].imshow(data_to_plot[3, num_arbre])
    axs[1].set_title(f"Septembre - Arbre {num_arbre} - Jour F: {jour_f}")
    axs[1].axis('off')
    axs[2].imshow(data_to_plot[2, num_arbre])
    axs[2].set_title(f"Octobre - Arbre {num_arbre} - Jour F: {jour_f}")
    axs[2].axis('off')
    axs[3].imshow(data_to_plot[1, num_arbre])
    axs[3].set_title(f"Novembre - Arbre {num_arbre} - Jour F: {jour_f}")
    axs[3].axis('off')
    plt.show()
  

def patch_to_array(data, capteur, path):
    """
    For a cameras defined ("ms" or "rgb"), load each patch in a list and save the month corresponding.

    Parameters
    ----------
    capteur : str
        define the "ms" or "rgb" patch to load. 
        
    data : pd.DataFrame
        dataframe which contains the name for each tree to use.
        
    path : str
        path where patchs are. 

    Returns
    -------
    """
    
    os.chdir(os.path.join(path, capteur))
    
    split  = []
    
    for mois in os.listdir():
        
        if capteur in mois:
            
            os.chdir(mois)
            img = []

            for ligne in range(data.shape[0]):
                
                line = data.loc[ligne]
                
                if capteur == "rgb": 
                    image_rgb = np.load(f"{line.id_tree}.npy")
                    img.append(image_rgb.copy())
                    del image_rgb
                
                elif capteur == "ms": 
                    image_ms = np.load(f"{line.id_tree}.npy")
                    img.append(image_ms.copy())
                    del image_ms
                
                else : 
                    print("capteur inconnue")
                    break
             
            split.append(img.copy())
            del img
            
            print(f"{mois} done")
            print("-----------")
            
            os.chdir("..")
    
    return np.array(split)

