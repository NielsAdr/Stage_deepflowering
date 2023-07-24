########### IMPORTATION DES PACKAGES

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split


########### IMPORTATION DES DONNÉES DE FLORAISON

path_patch = "/media/u108-s786/Donnees/Stage_Niels/Drone/data/Patch"
os.chdir("/media/u108-s786/Donnees/Stage_Niels/Drone/data")

# On importe les données de floraison de 2022
data_2022 = pd.read_csv("trees_cluster.csv")
data_2022 = data_2022.rename(columns={data_2022.columns[0]: 'id'})
data_2022 = data_2022.rename(columns={'arbre': 'pos'})

# On importe les données de floraison de 2021
data_2021 = pd.read_excel("/media/u108-s786/Donnees/Stage_Niels/NIRS/Donnees Corecoll 2021 pour NIRS.xlsx")
data_2021 = data_2021.dropna()

# On renomme la colonne jour F de data_2021 en jourF
data_2021 = data_2021.rename(columns={'jour F': 'jourF'})
data_2021 = data_2021.rename(columns={'arbre': 'pos'})

# On importe le dataframe de correspondance des années
corresp_2021_2022 = pd.read_csv("/media/u108-s786/Donnees/Stage Niels/NIRS/csv_x_y_cnn/comparatif_annees/correspondance.csv")

########### IMPORTATION DES DONNÉES NIRS
os.chdir("/media/u108-s786/Donnees/Stage_Niels/NIRS/csv & rds PLS rstudio/csv non conca/sans preprocess/")

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
D_all = pd.merge(D_all, data_2022, on=['pos','rang'], how = 'inner')
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

########### IMPORTATION DES DONNÉES NIRS TERRAIN

D1_bis = pd.read_csv("/media/u108-s786/Donnees/Stage Niels/NIRS/fruitflow-juillet2021.csv")

D1_bis[['rang', 'pos']] = D1_bis['Id'].str.split('-', n=2, expand=True).iloc[:, :2]
D1_bis = D1_bis.drop(columns=['Id', 'Température', 'Notes', 'Horodatage', 'Numéro de série instrument'])

########### TRAITEMENT DES DONNÉES NIRS TERRAIN

data_2022['rang'] = data_2022['rang'].astype(str)
data_2022['pos'] = data_2022['pos'].astype(str)

D_corresp = pd.merge(D1_bis, data_2022[['pos', 'rang','jourF']], on=['pos', 'rang'], how='inner')
# Si NDVI NIR et R ajoutés
#data_2022['R'] = juin_average_R
#data_2022['NIR'] = juin_average_NIR
#data_2022['NDVI'] = juin_average_NDVI
#D_corresp = pd.merge(D1_bis, data_2022[['pos', 'rang','jourF','NDVI','NIR','R']], on=['pos', 'rang'], how='inner')


########## On regroupe par la moyenne dans un premier temps pour essayer
D1_bis_means = D_corresp.groupby(['rang', 'pos']).mean()
X,y = D1_bis_means.iloc[:, :125],D1_bis_means.loc[:, 'jourF']


########## On laisse toutes les data dans un second temps et on sépare en train et test set 

# On détermine le nombre d'individus uniques
individus_uniques = D_corresp[['rang', 'pos']].drop_duplicates()

# On sépare en 2 groupes train et test
train_pos, test_pos = train_test_split(individus_uniques, test_size=0.2, random_state=42)

# Fusionner les données avec les valeurs de rang et pos pour chaque ensemble
train_data = pd.merge(train_pos, D_corresp, on=['rang', 'pos'], how='inner')
test_data = pd.merge(test_pos, D_corresp, on=['rang', 'pos'], how='inner')

# Définir y_train/test et X_train/test

y_train = train_data['jourF']
y_test = test_data['jourF']
X_train = train_data.drop(columns=['jourF','pos','rang'])
X_test = test_data.drop(columns=['jourF','pos','rang'])



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


########### TESTING

###########  ON VIENT RECOUPER NOS GEOTIFF POUR N'AVOIR QUE CE QU'IL FAUT

import tifffile as tiff
import numpy as np

path_ortho = "/media/u108-s786/Donnees/Stage Niels/Orthomosaiques/"
path_dem = "/media/u108-s786/Donnees/Stage Niels/DEM/"
camera = "ms"
mois = "septembre"

ortho = tiff.imread(path_ortho + camera + "/" + mois + "/" + "20210922_ms.tif")
dem = tiff.imread(path_dem + "rgb" + "/" + mois + "/" + "DEM-2021-06-25-MS.tif")

def first_values(array):
    if array.ndim == 3:
        array = array[:, :, 0]
    row_index = np.argmax(np.any(array > 0, axis=1))
    print("La première ligne avec une valeur est : ", row_index)
    col_index = np.argmax(np.any(array > 0, axis=0))
    print("La première colonne avec une valeur est : ", col_index)
    return (row_index,col_index)

        
def last_values(array):
    if array.ndim == 3:
        array = array[:, :, -1]
    row_index = np.argmax(np.any(array > 0, axis=1)[::-1])
    col_index = np.argmax(np.any(array > 0, axis=0)[::-1])
    row_index = array.shape[0] - row_index - 1
    print("La dernière ligne avec une valeur est : ", row_index)
    col_index = array.shape[1] - col_index - 1
    print("La dernière colonne avec une valeur est : ", col_index)
    return (row_index,col_index)
        
first_values_ortho = first_values(ortho)
last_values_ortho = last_values(ortho)
first_values_dem = first_values(dem)
last_values_dem = last_values(dem)

def decoupage_tiff(array,first_values,last_values):
    lignes = first_values[0],last_values[0]
    colonnes = first_values[1],last_values[1]
    if array.ndim == 3:
        new_array = array[lignes[0]:lignes[1],colonnes[0]:colonnes[1],:]
    else:
        new_array = array[lignes[0]:lignes[1],colonnes[0]:colonnes[1]]
    return(new_array)



new_ortho = decoupage_tiff(ortho,first_values_ortho,last_values_ortho)
new_dem = decoupage_tiff(dem,first_values_dem,last_values_dem)

tiff.imsave(path_ortho + camera + "/" + mois + "/" + "ortho_coupee.tif", new_ortho)
tiff.imsave(path_dem + camera + "/" + mois + "/" + "DEM_coupe.tif", new_dem)

###########  ON PREND LES GEOTIFF RECOUPÉS ET ON LES FAIT SE CORRESPONDRE

import tifffile as tiff
import SimpleITK as sitk
import numpy as np
from osgeo import gdal
from pyproj import Proj, transform

path_ortho = "/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/"
path_dem = "/media/u108-s786/Donnees/Stage_Niels/DEM/"
camera = "ms"
mois = "juin"

ortho = tiff.imread(path_ortho + camera + "/" + mois + "/" + "ms.tif")
dem = tiff.imread(path_dem + "rgb" + "/" + mois + "/" + "DEM.tif")


####
# # Sauvegarde RGB
# red = ortho[:,:,3]
# green = ortho[:,:,1]
# blue = ortho[:,:,0]

# tiff.imsave(path_ortho + camera + "/" + mois + "/" + "red.tif",red)
# tiff.imsave(path_ortho + camera + "/" + mois + "/" + "green.tif",green)
# tiff.imsave(path_ortho + camera + "/" + mois + "/" + "blue.tif",blue)

# ####


ds_ortho = gdal.Open(path_ortho + camera + "/" + mois + "/" + "ms.tif")
ds_dem = gdal.Open(path_dem + "rgb" + "/" + mois + "/" + "DEM.tif")

def convert_geotiff_wgs_to_lambert(orthomosaic:gdal.Dataset,image_shape):
    """
    Convert the geotiff coordinates of a WGS84 dataset to the Lambert 93 projection system.
    
    Args:
    - orthomosaic (gdal.Dataset): the WGS84 dataset to convert.
    - image_shape (tuple): the shape of the image.
    
    Returns:
    - new_gt (tuple): the new geotransform of the dataset with coordinates converted to Lambert 93 projection system.
    """
    
    # Extract GeoTransform object
    gt = orthomosaic.GetGeoTransform()
    
    # Define input and output projections
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:2154')
    
    # Extract the coordinates of the upper left and lower right corners of the image
    x1,y1 = gt[0],gt[3]
    x1bis,y1bis = gt[0] +(gt[1]*image_shape[1]),gt[3] +(gt[5]*image_shape[0])
    
    # Convert upper left and lower right corners from WGS 84 to Lambert 93
    x2,y2 = transform(inProj,outProj,x1,y1)
    x2,y2 = x2,y2
    x2bis,y2bis =  transform(inProj,outProj,x1bis,y1bis)
    
    # Calculate the resolution in Lambert 93
    res_x,res_y = abs(x2bis - x2)/image_shape[1],abs(y2bis - y2)/image_shape[0]
    
    # Create and return a new GeoTransform object with Lambert 93 coordinates and resolution
    new_gt = x2,res_x,0,y2,0,-res_y
    return new_gt

gt_ortho = ds_ortho.GetGeoTransform()
gt_dem = convert_geotiff_wgs_to_lambert(ds_dem,dem.shape)



def pixel_to_lambert_position(x,y,gt):
    """
    This function give the position in lambert of the selected point
    
    Args :
        x : The x value of the point
        y : The y value of the point
        gt : The 6 values tuple of the full orthomosaic (GetGeoTransform)
    
    Returns:
        The position of this point in lambert-93
    """
    x_rel = gt[0] + x * gt[1]
    y_rel = gt[3] + y * gt[5]
    return (x_rel,y_rel)


    
#def my_function(data_initial,gt_initial,gt_destination,shape_destination)
#Le but de cette fonction va être de passer via une matrice de transformation afin que l'image intiale soit dans le format de l'image finale
#Il me faut définir où se situe le point d'origine de mon espace de destination dans mon espace d'origine
# Ensuite, il faut remplir ma matrice vide par une pondération des pixels de ratio_x*x,ratio_y*y de mon espace initial

ortho_shape = ortho.shape
del ds_dem,ds_ortho

############ Recalage et redimensionnement via ratio de pixel size

image = sitk.GetImageFromArray(dem)
image_target = sitk.GetImageFromArray(ortho[:,:,5])
del ortho, dem

def rescale_image(image,image_target,gt_image,gt_target):
    
    # On calcule quelle est la différence de position des deux points initiaux des orthomosaïques
    deplacement_x,deplacement_y = (gt_image[0]-gt_target[0])/gt_image[1],(gt_image[3]-gt_target[3])/gt_image[5]
    
    # On paramètre la transformation de recalage
    translation_transform = sitk.TranslationTransform(2, [np.abs(deplacement_x), np.abs(deplacement_y)])
    
    # On recale l'image
    image_transformed = sitk.Resample(image, (image.GetWidth(), image.GetHeight()), translation_transform, sitk.sitkLinear)
    
    # Calculer le ratio entre les tailles des pixels
    ratio_x = (gt_target[1]/gt_image[1])
    ratio_y = (gt_target[5]/gt_image[5])
    
    # Créer la transformation d'échelle pour ajuster la taille des pixels
    scaling_transform = sitk.ScaleTransform(2)
    scaling_transform.SetScale([ratio_x, ratio_y])
    
    # On rescale l'image avec la taille de l'image target et les ratios de pixel
    image_source_rescaled = sitk.Resample(image_transformed, image_target.GetSize(), scaling_transform, sitk.sitkLinear)
   
    # On fait un array à partir de ça
    new_image = sitk.GetArrayFromImage(image_source_rescaled)
    return new_image

new_image = rescale_image(image,image_target,gt_dem,gt_ortho)
#del image

############## Recalage
import cv2

fixed_image = sitk.GetArrayFromImage(image_target)
moving_image = new_image

del image_target, image, new_image

# On définit des points correspondants
fixed_points = np.array([(6192,8555),(5326,2878),(18114,13772),(15240,16068),(4143,3968),(4131,4559),(10517,7748),(6747,4489),(7343,4556),(6078,5236),(6220,5349),(5643,2883),(15510,16033),(15239,16066)])  # Coordonnées des points dans l'image de référence
moving_points = np.array([(6229,8609),(5311,2861),(18085,13735),(15270,16107),(4153,3977),(4156,4579),(10498,7732),(6738,4470),(7324,4541),(6079,5244),(6224,5349),(5620,2860),(15562,16052),(15265,16107)])  # Coordonnées des points dans l'image flottante


### Affine

# Calculer la transformation affine
affine_transform = cv2.estimateAffinePartial2D(moving_points, fixed_points)[0]

# Appliquer la transformation affine à l'image à aligner
aligned_image = cv2.warpAffine(moving_image, affine_transform, (fixed_image.shape[1], fixed_image.shape[0]))

# Déformation toujours présente, image encore moins bien recalée et décalée dans des sens différents


### Homographique

# Estimer la transformation homographique
homography, _ = cv2.findHomography(moving_points, fixed_points)

# Appliquer la transformation homographique
aligned_image = cv2.warpPerspective(moving_image, homography, (fixed_image.shape[1], fixed_image.shape[0]))

# Déformation toujours présente, décalage dans le même sens sur toute l'image


### Polynomiale

# Estimation de la transformation polynomiale
polynomial_transform = cv2.estimateAffine2D(moving_points, fixed_points)[0]

# Appliquer la transformation polynomiale à l'image flottante
aligned_image = cv2.warpAffine(moving_image, polynomial_transform, (fixed_image.shape[1], fixed_image.shape[0]))

# Belle correspondance sur la fin, moins au début, je vais vérifier les correspondances au début du verger et réessayer


############## Visualisation

import matplotlib.pyplot as plt

# On regarde le décalage sur un grand patch au début du verger
plt.imshow(ortho[5000:7000,2500:4500,5],vmin = 12)
plt.imshow(aligned_image[5000:7000,2500:4500],vmin = 8)

# On regarde le décalage sur un grand patch à la fin du verger
plt.imshow(fixed_image[14500:16500,14000:16000],vmin = 8)
plt.imshow(aligned_image[14500:16500,14000:16000],vmin = 8)


# On regarde le décalage sur un petit patch au début du verger
plt.imshow(fixed_image[5200:5400,3300:3500],vmin = 0)
plt.imshow(aligned_image[5200:5400,3300:3500],vmin = 8)

# On regarde le décalage sur un petit patch à la fin du verger
plt.imshow(fixed_image[15800:16000,16000:16200],vmin = 9)
plt.imshow(aligned_image[15800:16000,16000:16200],vmin = 8)

# On regarde le décalage sur un petit patch à la fin du verger
plt.imshow(fixed_image[15600:15800,15800:16000],vmin = 9)
plt.imshow(aligned_image[15600:15800,15800:16000],vmin = 8)



############# Ok on va essayer la méthode de Fred
# Consiste à prendre le DEM, l'ortho, fais la variance glissante sur tout le dem, et utiliser les gt pour
# Trouver la correspondance de chaque pixel dans l'ortho, dans le DEM, et de le multiplier.

from scipy.ndimage import generic_filter
import tqdm
import torch
from torch.autograd import Variable

threshold = 0.0001

### Calcul via CPU

def variance_sliding_window_cpu(image, window_size):
    def variance(data):
        return data.var()

    filtered_dem = []

    # Diviser l'image en n chunks de taille égale
    num_chunks = 10
    chunk_size_rows = image.shape[0] // num_chunks
    chunk_size_cols = image.shape[1] // num_chunks

    for i in range(num_chunks):
        chunk_row = []
        print("Étape",i,"sur",num_chunks)
        for j in tqdm.trange(num_chunks):
            chunk = image[i * chunk_size_rows:(i + 1) * chunk_size_rows,
                          j * chunk_size_cols:(j + 1) * chunk_size_cols]
            # On applique la variance glissante
            filtered_chunk = generic_filter(chunk, variance, size=window_size)
            # On ajoute le chunk à la ligne en cours
            chunk_row.append(filtered_chunk)
            
        # On concatène les chunks de la ligne en cours 
        filtered_dem.append(np.concatenate(chunk_row, axis=1))
        
    # On concatène les lignes pour reconstituer l'image
    filtered_dem = np.concatenate(filtered_dem, axis=0)
    
    return filtered_dem


# On fait la variance glissante sur le DEM
filtered_dem = variance_sliding_window_cpu(dem, 11)

# On met à 0 sous un seuil, et à 1 le reste
filtered_dem[filtered_dem < threshold] = 0
filtered_dem[filtered_dem >= threshold] = 1

filtered_dem = np.int16(filtered_dem)


# le but maintenant va être de prendre la correspondace avec l'image et de multiplier soit par le plus proche
# Voisin, soit par une approximation bilinéaire, le pb étant comment faire entre 0 et 1 ?

def relative_pixel_position(x_initial,y_initial,gt_initial,gt_destination):
    """
    This function give the position of the pixel given in the destination image
    
    Args :
        x_initial : The x value of the point in the initial image
        y_initial : The y value of the point in the initial image
        gt_initial : The 6 values tuple of the full intiial orthomosaic (GetGeoTransform)
        gt_destination : The 6 values tuple of the full orthomosaic of destination (GetGeoTransform)
    
    Returns:
        The position of this point in the destination image in pixels
    """
    x_rel = gt_initial[0] + (x_initial * gt_initial[1])
    y_rel = gt_initial[3] - (-y_initial * gt_initial[5])
    x_dest = (x_rel - gt_destination[0])/(gt_destination[1])
    y_dest = (y_rel - gt_destination[3])/(gt_destination[5])
    return (x_dest,y_dest)


def reshape_dem_to_ortho_size(ortho, filtered_dem, gt_ortho, gt_dem):
    new_ortho = np.zeros(ortho.shape[:2])
    for i in tqdm.trange(ortho.shape[0]):
        for j in range(ortho.shape[1]):
            x,y = relative_pixel_position(i, j, gt_ortho, gt_dem)
            x,y = round(x),round(y)
            if x >= 0 and y >= 0 and x< filtered_dem.shape[0] and y < filtered_dem.shape[1] :
                new_ortho[i,j] =  filtered_dem[x,y]
    return new_ortho


dem_ortho_sized = reshape_dem_to_ortho_size(ortho, filtered_dem, gt_ortho, gt_dem)
dem_ortho_sizde = np.int16(dem_ortho_sized)

tiff.imsave('/media/u108-s786/Donnees/Stage Niels/DEM/mask/septembre/DEM_ortho_size.tiff',dem_ortho_sized)


#######################################################################
######### On a recalé le DEM sur l'ortho via Fijiyama, multiplions l'un par l'autre
#######################################################################

######### On va faire du Voronoï sur le DEM, mais a vant ça il faut labeliser les arbres

 
import tifffile as tiff
import SimpleITK as sitk
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
from pyproj import Proj, transform

path_dem = "/media/u108-s786/Donnees/Stage_Niels/DEM/fijiyama/"
path_ortho = "/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/"
camera = "ms"
mois = "novembre"

dem = tiff.imread(path_dem + mois + "/" + "Exported_data/" +"Transformed image.tif")
ds = gdal.Open(path_ortho + camera + "/" + mois + "/" + camera + ".tif")

label_tree = np.zeros_like(dem).astype(np.uint16)

metadata = pd.read_csv('/media/u108-s786/Donnees/Stage_Niels/Drone/data/trees_cluster_without_clear.csv')[["northing", "easting"]]


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
    
    #Si besoin de convertir en wgs84
    #metadata = convert_lambert_to_wgs84(metadata)
    
    metadata = metadata.assign(pixel_x=[(metadata["northing"].loc[i] - geot[3])/geot[5] for i in range(number_tree)],
                               pixel_y=[(metadata["easting"].loc[i] - geot[0])/geot[1] for i in range(number_tree)]
                              )
     
    return metadata   

# On ajoute les correspondances en pixels
metadata = Lambert_to_npy_coordinates(ds, metadata)

# On labelise chaque centroïde d'abre
for tree in range(len(metadata)):
    x,y = round(metadata.iloc[tree,2]),round(metadata.iloc[tree,3])
    label_tree[x,y] = 1
    
saving_path = '/media/u108-s786/Donnees/Stage_Niels/DEM/labels_centroides/'

tiff.imsave(saving_path + mois + "/" + "labels.tif",label_tree)


#######################################################################
####### ON FAIT LE VORONOI SUR IMAGEJ ET ON REVIENT ICI, TUTO SUR DRIVE
#######################################################################

path_dem = "/media/u108-s786/Donnees/Stage_Niels/DEM/fijiyama/"
path_voronoi = '/media/u108-s786/Donnees/Stage_Niels/DEM/labels_centroides/'

camera = "ms"
mois = "octobre"

dem = tiff.imread(path_dem + mois + "/" + "Exported_data/" +"Transformed image.tif")
voronoi = tiff.imread(path_voronoi + mois + "/" + "Voronoi_labels.tif")


# On remet le dem en mask 0 et 1
dem[dem < 1] = 0
dem[dem >= 1] = 1


new_dem = dem*voronoi

saving_path = '/media/u108-s786/Donnees/Stage_Niels/DEM/final_dem/'
tiff.imsave(saving_path + mois + ".tif",new_dem)

####### On tire les features de l'ortho via le dem

from skimage.measure import regionprops_table

path_ortho = "/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/"
path_dem = '/media/u108-s786/Donnees/Stage_Niels/DEM/final_dem/'

camera = "ms"
mois = "juin"

ds = gdal.Open(path_ortho + camera + "/" + mois + "/" + camera + ".tif")

dem = tiff.imread(path_dem +"/" + mois +".tif")
ortho = tiff.imread(path_ortho + camera + "/" + mois + "/" + "ms.tif")

## On ajoute un 7e canal qui sera le NDVI (pour avoir la moyenne du NDVI d'un arbre plutot que le NDVI de la moyenne)

# On prend le canal NIR et RED
NIR = ortho[:,:,5].astype(np.float32)
RED = ortho[:,:,3].astype(np.float32)

# Créer un masque pour exclure les pixels ayant une valeur de zéro dans NIR et RED
mask = np.logical_or(NIR == 0, RED == 0)

# Calculer le NDVI en utilisant le masque pour exclure les pixels avec des zéros
NDVI = np.where(mask, 0, (NIR - RED) / (NIR + RED))
                
new_orthomosaic = np.concatenate((ortho, NDVI[:, :, np.newaxis]), axis=2)


metadata = pd.read_csv('/media/u108-s786/Donnees/Stage_Niels/Drone/data/trees_cluster.csv')[["northing", "easting",'arbre','rang','jourF']]
metadata = metadata.rename(columns={'arbre': 'pos'})


metadata = Lambert_to_npy_coordinates(ds, metadata)

# La fonction d'écart-type que l'on va ajouter dans les extra_properties
# arguments must be in the specified order, matching regionprops
def image_stdv(region, intensities):
    return np.std(intensities[region], ddof=1)

# La fonction de médiane que l'on va ajouter dans les extra_properties
def image_median(region, intensities):
    return np.median(intensities[region])

# Mediane, moyenne, écart-type et area serait mieux d'après romain
props = regionprops_table(label_image = dem, intensity_image = new_orthomosaic, properties=('centroid',
                                                 'area',
                                                 'intensity_mean',
                                                 ),
                          extra_properties=([image_stdv,
                                            image_median])
                                            )

props = pd.DataFrame(props)


def trouver_centroide_proche(centroide_exact, props):
    centroides_approximatifs = props.iloc[:,:2]
    features_approximatifs = props.iloc[:,2:]
    distances = []
    for centroide_approx in centroides_approximatifs.values:
        distance = np.linalg.norm(np.round(centroide_approx) - np.round(centroide_exact))
        distances.append(distance)
    
    indice_plus_proche = np.argmin(distances)
    features_plus_proche = features_approximatifs.iloc[indice_plus_proche]
    
    return features_plus_proche

def affectation_centroides(metadata,props):
    centroides = metadata[['pixel_x', 'pixel_y']]
    features = []
    for i, centroide in enumerate(centroides.values):
        feature = trouver_centroide_proche(centroide, props)
        feature['rang'] = metadata.loc[i, 'rang']
        feature['pos'] = metadata.loc[i, 'pos']
        feature['jourF'] = metadata.loc[i, 'jourF']
        features.append(feature)
        
    return pd.DataFrame(features, columns=features[0].index)

features_tree = affectation_centroides(metadata, props)

saving_path = '/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/'
features_tree.to_csv(saving_path + camera + "/" + mois + '/features.csv', index=False)

#######################################################################
####### ON VA UTILISER LES CSV DES 4 DATES AFIN DE FORMER UN NOUVEAU CSV POUR DE LA PREDICTION
#######################################################################

# Dans ce csv on veut :
# L'aire relative pour chaque mois par rapport à juin (1 en juin)
# Le ndvi moyen de chaque date et le ndvi relatif par rapport à juin pour chaque date (1 en juin)

data_path = '/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/ms/'

data_juin =  pd.read_csv(data_path + "juin/features.csv")
data_septembre =  pd.read_csv(data_path + "septembre/features.csv")
data_octobre =  pd.read_csv(data_path + "octobre/features.csv")
data_novembre =  pd.read_csv(data_path + "novembre/features.csv")

# Renommer la colonne "intensity_mean-6" dans chaque DataFrame
data_juin = data_juin.rename(columns={"intensity_mean-6": "NDVI"})
data_septembre = data_septembre.rename(columns={"intensity_mean-6": "NDVI"})
data_octobre = data_octobre.rename(columns={"intensity_mean-6": "NDVI"})
data_novembre = data_novembre.rename(columns={"intensity_mean-6": "NDVI"})

# # Fusion des DataFrames sur les clés communes
# merged_data = pd.merge(data_juin, data_septembre, on=['rang', 'pos'], suffixes=('_juin', '_septembre'))
# merged_data = pd.merge(merged_data, data_octobre, on=['rang', 'pos'], suffixes=('', '_octobre'))
# merged_data = pd.merge(merged_data, data_novembre, on=['rang', 'pos'], suffixes=('', '_novembre'))

# Supprimer les colonnes spécifiées
# merged_data = merged_data.drop(['jourF_septembre', 'jourF_juin', 'jourF_novembre'], axis=1)

# Fusion des DataFrames
merged_data = data_juin[['rang', 'pos','jourF']].copy()

# Calcul de l'aire relative
merged_data['relative_area_juin'] = data_juin['area'] / data_juin['area']
merged_data['relative_area_septembre'] = data_septembre['area'] / data_juin['area']
merged_data['relative_area_octobre'] = data_octobre['area'] / data_juin['area']
merged_data['relative_area_novembre'] = data_novembre['area'] / data_juin['area']

# Calcul du NDVI relatif
merged_data['relative_NDVI_juin'] = data_juin['NDVI'] / data_juin['NDVI']
merged_data['relative_NDVI_septembre'] = data_septembre['NDVI'] / data_juin['NDVI']
merged_data['relative_NDVI_octobre'] = data_octobre['NDVI'] / data_juin['NDVI']
merged_data['relative_NDVI_novembre'] = data_novembre['NDVI'] / data_juin['NDVI']

# Calcul du delta NDVI
merged_data['delta_NDVI_juin'] = data_juin['NDVI'] - data_juin['NDVI']
merged_data['delta_NDVI_septembre'] = data_septembre['NDVI'] -  data_juin['NDVI'] 
merged_data['delta_NDVI_octobre'] = data_octobre['NDVI'] - data_juin['NDVI'] 
merged_data['delta_NDVI_novembre'] = data_novembre['NDVI'] - data_juin['NDVI'] 

# On sauvegarde ce dataframe

saving_path = '/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/ms/'
merged_data.to_csv(saving_path + 'features_relatives.csv', index = False)


########### CONCATÉNATION DE TOUS LES CSV D'ARBRES EN 1 CSV AVEC 'RANG', 'POS' ET JOURF

# Définir le chemin du dossier contenant les fichiers CSV
path = "/media/u108-s786/Donnees/Stage_Niels/Drone/data/temporal_patchs/Target/"
saving_path = "/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/"

# Créer une liste pour stocker les données de chaque fichier
donnees = []

# Parcourir tous les fichiers du dossier
for fichier in os.listdir(path):
    if fichier.endswith(".csv"):
        # Lire le fichier CSV
        chemin_fichier = os.path.join(path, fichier)
        df = pd.read_csv(chemin_fichier)
        
        # Extraire le rang et la position du nom du fichier
        rang = int(fichier.split("_")[1])
        position = int(fichier.split("_")[3].split(".")[0])
        
        # Ajouter le rang et la position comme colonnes au DataFrame
        df["rang"] = rang
        df["pos"] = position
        
        # Ajouter les données au DataFrame
        donnees.append(df)
        
# Concaténer tous les DataFrames en un seul
donnees_concat = pd.concat(donnees, ignore_index=True)

#### On ajoute la date de floraison :
path_metadata = "/media/u108-s786/Donnees/Stage_Niels/Drone/data/"

# On importe les données de floraison de 2022
metadata = pd.read_csv(path_metadata + "trees_cluster.csv")
metadata = metadata.rename(columns={metadata.columns[0]: 'id'})
metadata = metadata.rename(columns={'arbre': 'pos'})

# On ajoute le mean et le delta jourF
genot_means = metadata.groupby('genot')['jourF'].mean()
metadata['mean'] = metadata.groupby('genot')['jourF'].transform(lambda x: genot_means[x.name])

# On crée une colonne de l'écart de floraison de chaque arbre par rapport à son géno
metadata['delta'] = metadata.loc[:,'jourF']-metadata.loc[:,'mean']

# On ne prend que la clé 'rang' 'pos', le jourF, le mean et le delta jourF
y = metadata.loc[:,['jourF','rang','pos','mean','delta']]

# On rajoute la date de floraison en concaténant
donnees_concat = pd.merge(donnees_concat, y, on = ['rang','pos'])

# Enregistrer le DataFrame concaténé dans un nouveau fichier CSV
donnees_concat.to_csv(saving_path+"manual_index_surface_ndvi_over_time.csv", index=False)


#### Plot jourF/delta/mean en fonction de la surface en novembre

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv("/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/manual_index_surface_ndvi_over_time.csv")

#### Régression linéaire

# Tracer les points
plt.scatter(df['M1_surf'], df['jourF'])

# Récupérer les données de vos colonnes
x = df['M1_surf']
y = df['jourF']

# Calculer la pente (m) et l'ordonnée à l'origine (c) avec la régression linéaire
m, c = np.polyfit(x, y, 1)

# Calculer les prédictions
y_pred = m * x + c

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y, y_pred)

# Tracer la droite de régression
plt.plot(x, y_pred, color='red', label='Droite de régression')

# Ajouter le coefficient de détermination (R²) dans la légende
plt.legend(['Droite de régression (R²={:.2f})'.format(r2)])

# Ajouter des étiquettes et un titre
plt.xlabel('% de feuilles en septembre')
plt.ylabel('jourF')
plt.title('jourF en fonction du % feuilles en septembre')

# Afficher le graphique
plt.show()

#### On tente la régression polylinéaire
from sklearn.linear_model import LinearRegression

# Récupérer les données de vos colonnes
X = np.column_stack((df['M1_surf'], df['M2_surf'], df['M3_surf'], df['M1_NDVI'], df['M2_NDVI'], df['M3_NDVI']))
y = df['jourF']

# Création de l'estimateur de régression linéaire multiple
regression = LinearRegression()

# Entraînement du modèle de régression
regression.fit(X, y)

y_pred = regression.predict(X)  # Prédiction sur les données d'entraînement
r2 = r2_score(y, y_pred)  # Calcul du coefficient de détermination R2

# Tracer les points observés par rapport aux valeurs prédites
plt.scatter(y, y_pred)

# Tracer une ligne diagonale pour représenter la droite x=y
plt.plot([min(y), max(y)], [min(y), max(y)], color='gray', linestyle='--', label='x=y')

# Ajouter le coefficient de détermination (R²) dans le titre du graphique
plt.title('Régression polylinéaire (R²={:.2f})'.format(r2))

# Ajouter des étiquettes et une légende
plt.xlabel('Valeurs observées')
plt.ylabel('Valeurs prédites')
plt.legend()

# Afficher le graphique
plt.show()


###### VISUALISATION DE NOS FEATURES RELATIVES
import seaborn as sns
from sklearn.linear_model import LinearRegression

# features auto
df = pd.read_csv("/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/ms/features_relatives.csv") # features auto

# features manuelles
df = pd.read_csv("/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/manual_index_surface_ndvi_over_time.csv")

# Densité aire relative

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=df['relative_area_septembre'],
            color='crimson', label='septembre', fill=True, ax=ax).set(title='Relative area of differents months')
sns.kdeplot(data=df['relative_area_octobre'],
            color='limegreen', label='octobre', fill=True, ax=ax)
sns.kdeplot(data=df['relative_area_novembre'],
            color='gold', label='novembre', fill=True, ax=ax)
ax.legend()
#ax.set_xlim(0, 2.5) # auto
ax.set_xlim(0, 1)# manuel
ax.set(xlabel='Relative area')
plt.tight_layout()
plt.show()

# Densité NDVI relatif

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=df['relative_NDVI_septembre'],
            color='crimson', label='septembre', fill=True, ax=ax).set(title='Relative NDVI of differents months')
sns.kdeplot(data=df['relative_NDVI_octobre'],
            color='limegreen', label='octobre', fill=True, ax=ax)
sns.kdeplot(data=df['relative_NDVI_novembre'],
            color='gold', label='novembre', fill=True, ax=ax)
ax.legend()
ax.set_xlim(0, 2.5) # auto
#ax.set_xlim(0, 1)# manuel
ax.set(xlabel='Relative NDVI')
plt.tight_layout()
plt.show()

# Densité delta NDVI

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=df['delta_NDVI_septembre'],
            color='crimson', label='septembre', fill=True, ax=ax).set(title='Delta NDVI of differents months')
sns.kdeplot(data=df['delta_NDVI_octobre'],
            color='limegreen', label='octobre', fill=True, ax=ax)
sns.kdeplot(data=df['delta_NDVI_novembre'],
            color='gold', label='novembre', fill=True, ax=ax)
ax.legend()
ax.set_xlim(-0.3, 0.4)
ax.set(xlabel='Delta NDVI')
plt.tight_layout()
plt.show()

# jourF en fonction de M3_surf

# Tracer le graphique
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="M3_surf", y="jourF", marker='o', s = 100)
plt.xlabel('Surface au moins de novembre')
plt.ylabel('Date')
plt.title('Relation entre jourF et la surface au mois de novembre')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculer le coefficient de détermination R^2
X = df["M3_surf"].values.reshape(-1, 1)  # Conversion de la colonne jourF en un tableau 2D
y = df["jourF"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("R^2 (Coefficient de détermination) :", r2)
# Il y a 2 outliers, un en aire, un en NDVI, je vais les retirer pour revisualiser, ce sont le 184 et 270

df.drop([184,270], axis=0, inplace=True)
