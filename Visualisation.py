import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff

os.chdir("/home/u108-s786/github/Stage")

from patch_df import patch_to_array, time_plot, filter_data_to_export, combine_trees_RGB_days, generate_tab_index

path_patch = "/media/u108-s786/Donnees/FruitFlowDrone/data/Patch"
os.chdir("/media/u108-s786/Donnees/FruitFlowDrone/data")

data = pd.read_csv("trees_cluster.csv")
data = data.rename(columns={data.columns[0]: 'id'})


data_rgb = patch_to_array(data,"rgb",path_patch)

# 0 = juin, 1 = novembre, 2 = octobre, 3 = septembre ==> 0,3,2,1
# plt.imshow(data_rgb[date, arbre])

### Afin d'avoir une visualisation avec 4 images on ne va garder que les génoypes avec 4 individus

# Compter le nombre d'occurrences de chaque individu par génotype
counts = data.groupby('genot')['id_tree'].count().reset_index(name='count')

# Filtrer les individus qui ont un compte égal à 4
filtered_df = data[data['genot'].isin(counts.loc[counts['count'] == 4, 'genot'])]

# Afficher le résultat
print(filtered_df)


## Visualisation de la distribution écart-type des jours de floraison des génotypes
std_genot = filtered_df[["genot","jourF"]].groupby("genot").agg(["std"]).reset_index().dropna()
std_genot.columns = ["genot", "std"]

plt.figure(figsize=(12,8))
sns.histplot(std_genot["std"], kde=True)
plt.xlabel("écart-type des jours de floraison (stade F)", fontsize=20)
plt.ylabel("count", fontsize=20)
plt.title("Distribution écart-type des jours de floraison des génotypes", fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



# Je calcule la moyenne et l'écart type des individus pour chaque génotype
mean_genot = filtered_df[["genot","jourF"]].groupby("genot").agg(["mean"]).reset_index().dropna()
mean_genot.columns = ["genot", "mean"]

# Je centre la date de floraison de chaque individu par rapport à la moyenne de son génotyoe
merge1 = filtered_df.merge(mean_genot[["genot", "mean"]], how="left",on="genot")
merge1["jourF_centre"] = merge1["jourF"] - merge1["mean"]

#J'affiche
merge1[["jourF_centre", "jourF", "mean"]]



#Je calcule l'écart type des dates de floraison des individu pour chaque génotype et je l'ajoute au dataset
std_OnCentered_genot = merge1[["genot","jourF_centre"]].groupby("genot").agg("std").reset_index().dropna()
std_OnCentered_genot.columns = ['genot', 'std']

merge2 = merge1.merge(std_OnCentered_genot[["genot", "std"]], how="left",on="genot")
merge2 = merge2.dropna().sort_values(by="std")

plt.figure(figsize=(25,15))
sns.boxplot(x="jourF_centre", y="clone", data=merge2, hue="genot", showfliers = False)
plt.title("Visualisation des écart-types des jours de floraison pour chaque génotype", fontsize=25)
plt.xlabel("jour de floraison centré", fontsize=20)
plt.ylabel("génotype", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(color='w')
plt.legend([])
plt.show()


### Sélection de 4 génotpetifffile.imwrite('mon_array.tiff', my_array)s avec 2 écarts-types petits et 2 grands
merge2 = merge2.sort_values(by=['std', 'genot'])
max_std_genot = merge2.tail(16)
min_std_genot = merge2.head(4)

minmax_std_genot = pd.concat([min_std_genot,max_std_genot])

## On plot les 4 arbres de chaque génotype pour observer les différences d'évolution

#for arbre in (minmax_std_genot.iloc[:,0]):
    #time_plot(data_rgb,arbre,data)
    
    
### L'objectif dorénavant va être de cumuler les 4 images dans différents array 4D (2D image, RGB, mois)

df_info_array = filter_data_to_export(data_rgb, minmax_std_genot)
df_info_array = df_info_array.sort_values(by=['std', 'jourF'])

# On utilise la fonction combine_trees_RGB_days afin d'avoir nos images prêtes à l'export
tab_index = generate_tab_index(5,4) #Le premier indice est le nombre de génotypes, le 2e est le nombre d'arbres dans le génotype
combine = combine_trees_RGB_days(tab_index,df_info_array) #On combine le tout en un array pour pouvoir l'exporter en tiff

# On crée un dossier pour stocker les images si il n'existe pas
if not os.path.exists('/media/u108-s786/Donnees/donnes_test_import_3d'):
    os.mkdir('/media/u108-s786/Donnees/donnes_test_import_3d')

#On définit le savepath
os.chdir('/media/u108-s786/Donnees/donnes_test_import_3d/')

# On sauvegarde notre image
tiff.imwrite('concatenation',combine)

# Parcourir chaque arbre et enregistrer l'image correspondante
#for i in range(test.shape[0]):
    #arbre = test.loc[i]
    #nom_image = f"Arbre_{arbre['id']}_genot_{arbre['genot']}_jourF_{arbre['jourF']}.tiff"
    #tiff.imwrite(os.path.join('test', nom_image), arbre['array'])