########### IMPORTATION DES PACKAGES

import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patch_to_array

########### IMPORTATION DES DONNÉES

path_patch = "/media/u108-s786/Donnees/FruitFlowDrone/data/Patch"
os.chdir("/media/u108-s786/Donnees/FruitFlowDrone/data")

data = pd.read_csv("trees_cluster.csv")
data = data.rename(columns={data.columns[0]: 'id'})

type_data='ms'
data_rgb = patch_to_array(data,type_data,path_patch)

########### FONCTION DE VISUALISATION TEMPORELLE D'UN ARBRE 

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
  
########### TRI DES DONNÉES

### Afin d'avoir une visualisation avec 4 images on ne va garder que les génoypes avec 4 individus

# Compter le nombre d'occurrences de chaque individu par génotype
counts = data.groupby('genot')['id_tree'].count().reset_index(name='count')

# Filtrer les individus qui ont un compte égal à 4
filtered_df = data[data['genot'].isin(counts.loc[counts['count'] == 4, 'genot'])]

########### TRAITEMENT DES DONNÉES

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

########### VISUALISATION

## Visualisation de la distribution écart-type des jours de floraison des génotypes
plt.figure(figsize=(12,8))
sns.histplot(std_genot["std"], kde=True)
plt.xlabel("écart-type des jours de floraison (stade F)", fontsize=20)
plt.ylabel("count", fontsize=20)
plt.title("Distribution écart-type des jours de floraison des génotypes", fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


## Visualisation des écart-types des jours de floraison pour chaque génotype"
plt.figure(figsize=(25,15))
sns.boxplot(x="jourF_centre", y="clone", data=merge2, hue="genot", showfliers = False)
plt.title("Visualisation des écart-types des jours de floraison pour chaque génotype", fontsize=25)
plt.xlabel("jour de floraison centré", fontsize=20)
plt.ylabel("génotype", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(color='w')
plt.legend([])
plt.show()

# Visualisation de la répartition du jour de floraison en fonction de l'année

corresp = pd.read_csv('/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/corresp_2021_2022')

# Sélectionner les colonnes qui contiennent les dates de floraison des deux années
floraison_2021 = corresp['jourF_2021']
floraison_2022 = corresp['jourF_2022']
print("En 2021, la date moyenne de floraison est : ",floraison_2021.mean(),"et l'écart-type est : ",floraison_2021.std())
print("En 2022, la date moyenne de floraison est : ",floraison_2022.mean(),"et l'écart-type est : ",floraison_2022.std())

# Créer un histogramme côte à côte pour chaque année
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].hist(floraison_2021, bins=20, color='blue', alpha=0.5)
ax[1].hist(floraison_2022, bins=20, color='red', alpha=0.5)

# Ajouter des titres et des légendes
ax[0].set_title('Dates de floraison en 2021')
ax[1].set_title('Dates de floraison en 2022')
ax[0].set_xlabel('Jours')
ax[1].set_xlabel('Jours')
ax[0].set_ylabel('Nombre d\'arbres')
ax[1].set_ylabel('Nombre d\'arbres')
plt.show()

# En boxplot
sns.boxplot(data=[floraison_2021, floraison_2022])
plt.xticks(ticks=[0, 1], labels=['2021', '2022'])
plt.xlabel('Année')
plt.ylabel('Jour de floraison')
plt.show()

# Test statistique, savoir si l'année est significative

from scipy.stats import f_oneway

resultat_anova = f_oneway(floraison_2021, floraison_2022)

print("Statistique F :", resultat_anova.statistic)
print("p-value :", resultat_anova.pvalue)