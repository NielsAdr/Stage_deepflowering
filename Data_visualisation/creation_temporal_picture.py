########### IMPORTATION DES PACKAGES

import os 
import warnings
import datetime
import tqdm
import numpy as np
import pandas as pd
import tifffile as tiff
from Data_visualisation import sorting_exportation
from patch_to_array import patch_to_array

########### SELECTION DES PARAMETRES

type_data='rgb' # Choisir si on importe les données rgb ou ms
tri = 'mean' # Choisir par quel paramètre les arbres seront triés
head = 4 # Choisir le nombre d'arbres qui sont au minimum du paramètre choisi précédement (tri)
tail = 4 # Choisir le nombre d'arbres qui sont au maximum du paramètre choisi précédement (tri)

########### IMPORTATION DES DONNÉES

path_patch = "/media/u108-s786/Donnees/Stage_Niels/Drone/data/Patch"
os.chdir("/media/u108-s786/Donnees/Stage_Niels/Drone/data")

data = pd.read_csv("trees_cluster.csv")
data = data.rename(columns={data.columns[0]: 'id'})
data = data.rename(columns={'arbre': 'pos'})

data_rgb = patch_to_array(data,type_data,path_patch)

# 0 = juin, 1 = novembre, 2 = octobre, 3 = septembre ==> 0,3,2,1

warnings.filterwarnings('ignore', category=FutureWarning) # On évite les warnings futurs de pandas

########### TRI & TRAITEMENT DES DONNÉES

### Afin d'avoir une visualisation avec 4 images on ne va garder que les génoypes avec 4 individus
# Compter le nombre d'occurrences de chaque individu par génotype
counts = data.groupby('genot')['id_tree'].count().reset_index(name='count')

# Filtrer les individus qui ont un compte égal à 4
filtered_df = data[data['genot'].isin(counts.loc[counts['count'] == 4, 'genot'])]


## Je calcule la moyenne et l'écart type des individus pour chaque génotype
# Moyenne
mean_genot = filtered_df[["genot","jourF"]].groupby("genot").agg(["mean"]).reset_index().dropna()
mean_genot.columns = ["genot", "mean"]

# Écart-type
std_genot = filtered_df[["genot","jourF"]].groupby("genot").agg(["std"]).reset_index().dropna()
std_genot.columns = ["genot", "std"]

## Je centre la date de floraison de chaque individu par rapport à la moyenne de son génotyoe
merge1 = filtered_df.merge(mean_genot[["genot", "mean"]], how="left",on="genot")
merge1["jourF_centre"] = merge1["jourF"] - merge1["mean"]

## Je calcule l'écart type des dates de floraison des individu pour chaque génotype et je l'ajoute au dataset
std_OnCentered_genot = merge1[["genot","jourF_centre"]].groupby("genot").agg("std").reset_index().dropna()
std_OnCentered_genot.columns = ['genot', 'std']

merge2 = merge1.merge(std_OnCentered_genot[["genot", "std"]], how="left",on="genot")
merge2 = merge2.dropna().sort_values(by="std")


########### SÉLECTION DES PARAMETRES DE L'IMAGE ENREGISTRÉE

merge2 = merge2.sort_values(by=[tri, 'genot'])
min_genot = merge2.head(head*4)
max_genot = merge2.tail(tail*4)

minmax_genot = pd.concat([min_genot,max_genot])
nombre_geno = int(len(minmax_genot)/4)

### L'objectif dorénavant va être de cumuler les 4 images dans différents array 4D (2D image, RGB, mois)

df_info_array = filter_data_to_export(data_rgb, minmax_genot)
df_info_array = df_info_array.sort_values(by=['std','jourF','genot'])

# On utilise la fonction combine_trees_RGB_days afin d'avoir nos images prêtes à l'export
tab_index = generate_tab_index(nombre_geno,4) #Le premier indice est le nombre de génotypes, le 2e est le nombre d'arbres dans le génotype
combine = combine_trees_RGB_days(tab_index,df_info_array) #On combine le tout en un array pour pouvoir l'exporter en tiff

########### ENREGISTREMENT DE L'IMAGE

# On rédige nos métadata et on transpose pour le MS pour ImageJ
# combine_2=np.transpose(combine,(0,3,1,2))
# metadata = {'axes': 'TCYX','t': 4,  'c': 6, 'y': 800, 'x': 1000}

# On crée un dossier pour stocker les images si il n'existe pas
if not os.path.exists('/media/u108-s786/Donnees/images_visualisation_tiff'):
    os.mkdir('/media/u108-s786/Donnees/images_visualisation_tiff')

#On définit le savepath
os.chdir('/media/u108-s786/Donnees/images_visualisation_tiff/')

# On prend l'heure actuelle afin que nos images ne s'écrasent pas
now = datetime.datetime.now()
heure = now.strftime("%Y-%m-%d_%H-%M-%S")

# On sauvegarde notre image
tiff.imwrite('concatenation_'+type_data+'_deltaJF_'+tri+'_'+str(nombre_geno)+f'_arbres_{heure}',combine)

########### CRÉATION D'IMAGES TEMPORELLES VERTICALES NDVI ET RGB POUR CHAQUE ARBRE, ON ENREGISTRE PAR RANG_N°_POS_N°.TIFF

data_rgb = patch_to_array(data,"rgb",path_patch)
data_ms = patch_to_array(data,"ms",path_patch)

# Index des mois dans l'ordre souhaité : 'juin', 'septembre', 'octobre', 'novembre'
order = [0, 3, 2, 1]

# Réorganisation du tableau data_rgb
data_rgb_ordered = data_rgb[order]
data_ms_ordered = data_ms[order]

rgb_path =  "/media/u108-s786/Donnees/Stage_Niels/Drone/data/temporal_patchs/rgb/"
ndvi_path = "/media/u108-s786/Donnees/Stage_Niels/Drone/data/temporal_patchs/ndvi/"

dates = data_ms.shape[0]


for i in tqdm.trange(data_ms.shape[1]):
    rgb = np.zeros(([300*dates,300,3])).astype(np.uint8)
    ndvi = np.zeros(([200*dates,200])).astype(np.float16)
    for j in range (dates):
        ## On attribue l'image RGB
        rgb[j*300:(j+1)*300,:,:] = data_rgb_ordered[j,i,:,:,:]
        
        ## On prend les canaux NIR et RED en float pour éviter les erreurs
        NIR = data_ms_ordered[j,i,:,:,5].astype(np.float16)
        RED = data_ms_ordered[j,i,:,:,3].astype(np.float16)
        
        ## On attribue l'image NDVI
        ndvi[j*200:(j+1)*200,:] = (NIR-RED)/(NIR+RED)
    tiff.imwrite(rgb_path+"rgb_rang_"+str(data.loc[i,'rang'])+"_pos_"+str(data.loc[i,'pos'])+".tiff",rgb)
    tiff.imwrite(ndvi_path+"ndvi_rang_"+str(data.loc[i,'rang'])+"_pos_"+str(data.loc[i,'pos'])+".tiff",ndvi)
    