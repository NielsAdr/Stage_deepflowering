import os
import numpy as np

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
