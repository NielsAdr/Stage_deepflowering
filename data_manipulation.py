########### IMPORTATION DES PACKAGES

import os 
import pandas as pd
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy.stats import linregress
import patch_to_array

########### IMPORTATION DES DONNÉES DE FLORAISON

path_patch = "/media/u108-s786/Donnees/FruitFlowDrone/data/Patch"
os.chdir("/media/u108-s786/Donnees/FruitFlowDrone/data")

# On importe les données de floraison de 2022
data_2022 = pd.read_csv("trees_cluster.csv")
data_2022 = data_2022.rename(columns={data_2022.columns[0]: 'id'})
data_2022 = data_2022.rename(columns={'arbre': 'pos'})

# On importe les données de floraison de 2021
data_2021 = pd.read_excel("/media/u108-s786/Donnees/NIRS data D1-D7/Donnees Corecoll 2021 pour NIRS.xlsx")
data_2021 = data_2021.dropna()

# On renomme la colonne jour F de data_2021 en jourF
data_2021 = data_2021.rename(columns={'jour F': 'jourF'})
data_2021 = data_2021.rename(columns={'arbre': 'pos'})

# On importe le dataframe de correspondance des années
corresp_2021_2022 = pd.read_csv("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/comparatif_annees/correspondance.csv")

########### IMPORTATION DES DONNÉES NIRS
os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv & rds PLS rstudio/csv non conca/sans preprocess/")

D1 = pd.read_csv("D1.csv")
D2 = pd.read_csv("D2.csv")
D3 = pd.read_csv("D3.csv")
D4 = pd.read_csv("D4.csv")

########### TRAITEMENT DES DONNÉES NIRS

# On crée une clé commune pour toutes les dates et on y ajoute jourF
D1_selected = D1[['pos', 'rang']]
D_all = pd.merge(D1_selected, D2[['pos', 'rang']], on=['pos', 'rang'], how='inner')
D_all = pd.merge(D_all, D3[['pos', 'rang']], on=['pos', 'rang'], how='inner')
D_all = pd.merge(D_all, D4[['pos', 'rang']], on=['pos', 'rang'], how='inner')
D_all = pd.merge(D_all, corresp_2021_2022, on=['pos','rang'], how = 'inner')
D_all = D_all.loc[:,['pos','rang']]

# On met data 2022 au même nombre d'arbres
data_2022 = pd.merge(data_2022, D_all, on=['pos','rang'], how = 'inner')
data_2021 = pd.merge(data_2021, D_all, on=['pos','rang'], how = 'inner')

# On équilibre les données de façon à avoir toujours les mêmes arbres pour chaque date grâce à la clé commune
D1 = pd.merge(D_all, D1, on=['pos', 'rang'], how='inner')
D2 = pd.merge(D_all, D2, on=['pos', 'rang'], how='inner')
D3 = pd.merge(D_all, D3, on=['pos', 'rang'], how='inner')
D4 = pd.merge(D_all, D4, on=['pos', 'rang'], how='inner')

Y = D1.loc[:,'jourF']
X1 = D1.iloc[:,5:2157]
X2 = D2.iloc[:,5:2157]
X3 = D3.iloc[:,5:2157]
X4 = D4.iloc[:,5:2157]

X_all = pd.DataFrame(np.concatenate((X1,X2,X3,X4),axis=1))

# On sauvegarde les X et Y non concat
#os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022/non concat")
#X1.to_csv('X1.csv', index=False)
#X2.to_csv('X2.csv', index=False)
#X3.to_csv('X3.csv', index=False)
#X4.to_csv('X4.csv', index=False)
#Y.to_csv('Y.csv', index=False)

# On sauvegarde les X et Y concat
#os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022/concat")
#X_all.to_csv('X.csv', index=False)
#Y.to_csv('Y.csv', index=False)


# On sauvegarde les dataframe 'raw'
#os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022")
#D1.to_csv('D1.csv', index=False)
#D2.to_csv('D2.csv', index=False)
#D3.to_csv('D3.csv', index=False)
#D4.to_csv('D4.csv', index=False)

########### MERGE DES DONNÉES

# Fusion des dataframes sur les clés communes 'arbre' et 'rang'
#merged_data = pd.merge(data_2021[['arbre', 'rang', 'jourF']],
                       #data_2022[['arbre', 'rang', 'jourF']],
                       #on=['arbre', 'rang'],
                       #suffixes=('_2021', '_2022'))

#merged_data.to_csv('/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/corresp_2021_2022',index = False)

########### REMPLACER LA DATE DE FLORAISON PAR LA DATE DE FLORAION MOYENNE DU GENOTYPE

# Calculer la date de floraison moyenne pour chaque génotype
genot_means = data_2022.groupby('genot')['jourF'].mean()

# Appliquer la moyenne à toutes les entrées de chaque groupe
data_2022['mean'] = data_2022.groupby('genot')['jourF'].transform(lambda x: genot_means[x.name])

# On enregistre cette colonne comme un Y_mean
Y_mean = data_2022.loc[:,'mean']
#os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022/non concat")
#Y_mean.to_csv('Y_mean.csv', index = False)

########### REMPLACER LA DATE DE FLORAISON PAR LE DELTA FLORAISON /R AU GENOTYPE

# On crée une colonne de l'écart de floraison de chaque arbre par rapport à son géno
data_2022['deltajourF'] = data_2022.loc[:,'jourF']-data_2022.loc[:,'mean']

# On enregistre cette colonne comme un Y_delta
Y_delta = data_2022.loc[:,'deltajourF']

#os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022/non concat")
# Y_delta.to_csv('Y_delta.csv', index = False)

########### ON DETERMINE LE RANG DE FLORAISON PAR RAPPORT AU GENO

def add_ranks(dataframe):
    # Créer une copie du dataframe
    df = dataframe.copy()

    # Grouper les arbres par génotype
    grouped = df.groupby('genot')

    # Pour chaque groupe de génotype, donner à chaque arbre le rang correspondant à son ordre de floraison dans le groupe
    for genot, group in grouped:
        # Utiliser la fonction rank pour donner le même rang à tous les arbres ayant la même date de floraison
        df.loc[group.index, 'rang'] = group['jourF'].rank(method='dense')

    return df

data_2022_with_ranks = add_ranks(data_2022)


###################################################################################
####### TRANSFORMATION DES IMAGES POUR APPLIQUER LE FILTRE DEM ET RETIRER LE GAZON
###################################################################################

def Lambert_to_npy_coordinates(orthomosaic:gdal.Dataset, metadata:pd.DataFrame):
    """
    Transform lambert-93 coordinates to numpy coordinates.

    Parameters
    ----------
    orthomosaic : osgeo.gdal.Dataset
        a gdal dataset which contains orthomosaic
    
    metadata : pd.DataFrame
        a dataset which easting and northing coordinates

    Returns
    -------
    Add x and y numpy coordinates to metadata dataframe.
    """
    
    geot = orthomosaic.GetGeoTransform() 
    number_tree = metadata.shape[0]
    
    metadata = metadata.assign(pixel_x=[(metadata["easting"].loc[i] - geot[0])/geot[1] for i in range(number_tree)],
                              pixel_y=[(metadata["northing"].loc[i] - geot[3])/geot[5] for i in range(number_tree)])
     
    return metadata   
    
    
    
def raster_to_img(orthomosaic:gdal.Dataset):  
    """
    transform an orthomosaic to numpy array.

    Parameters
    ----------
    orthomosaic : osgeo.gdal.Dataset
        a gdal dataset which contains orthomosaic

    Returns
    -------
    orthomosaic in numpy array of dimension (resolution x,resolution y, channels).
    """
    
    raster = [orthomosaic.GetRasterBand(i+1).ReadAsArray() for i in range(orthomosaic.RasterCount)]
    return np.dstack(raster)


metadata = data_2022[["northing", "easting", "id_tree","pos","rang"]]
DEM_PATH = "/media/u108-s786/Donnees/FruitFlowDrone/data/agisoft/DEM.tif"
PATH_450 = "/media/u108-s786/Donnees/FruitFlowDrone/data/agisoft/orthomosaic_master_570_450_12mm.tif"
ds_dem = gdal.Open(DEM_PATH)
ds_450 = gdal.Open(PATH_450)

img_dem = raster_to_img(ds_dem)[:,:,0]
img_450 = raster_to_img(ds_450)[:,:,0]

pad_width = ((0, img_450.shape[0]-img_dem.shape[0]), (0, img_450.shape[1]-img_dem.shape[1]))
img_dem_pad = np.pad(img_dem, pad_width, mode='constant', constant_values = -32767)

metadata_dem = Lambert_to_npy_coordinates(ds_dem,metadata)
metadata_450 = Lambert_to_npy_coordinates(ds_450,metadata)

