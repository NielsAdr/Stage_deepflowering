from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir("/home/u108-s786/github/Stage")
from utils import adj_asd

os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/sans NA/concat/X_Y")

# Importer les données 
X, y = pd.read_csv("X_out_concat.csv"),pd.read_csv("Y_out_concat.csv")

# On renomme les colonnes du spectre en partant de 350
# X = X.rename(columns=lambda x: x.replace(x, str(int(x)+350))) # Si les colonnes commencent à 0 (plus rapide)

for i, col in enumerate(X.columns):
    X.rename(columns={col: str(i+350)}, inplace=True)

#X_adj_cut = adj_asd(X,[650,1450],5)

# Définir le nombre de composantes à utiliser pour la PLS et la SEED
n_components = 33
seed = 42

##################################

def compute_pls_rmse(X, y, n_components_range=(1,30)):
    rmse = []
    bestcomp, RMSEopt = 1, 100
    
    for i in range(n_components_range[0], n_components_range[1]):
        # Initialiser l'objet de régression PLS
        pls = PLSRegression(n_components=i)

        # Effectuer une cross-validation à 20-fold avec prédiction
        y_pred = cross_val_predict(pls, X, y, cv=20)

        # Calculer le RMSE sur les prédictions
        rmse_i = np.sqrt(mean_squared_error(y, y_pred))
        rmse.append(rmse_i)
        print("nombre de composantes = " + str(i))
        print("RMSE = " + str(rmse_i))

        if rmse_i < RMSEopt:
            RMSEopt = rmse_i
            bestcomp = i

    print("Le meilleur nombre de composantes est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
    plt.plot(range(n_components_range[0], n_components_range[1]),rmse)
    return bestcomp

n_components = compute_pls_rmse(X,y)


def pls_seed_rmse(X, y, n_components, test_size, num_splits, seed = 42):
    rmse_scores = []
    for i in range(num_splits):
        # On sépare les données en train et test avec une seed différente à chaque itération
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed+i)

        # Initialiser l'objet de régression PLS
        pls = PLSRegression(n_components=n_components)

        # Entraîner le modèle PLS sur l'ensemble d'entraînement complet
        pls.fit(X_train, y_train)

        # Prédire les valeurs sur l'ensemble de test
        y_pred_test = pls.predict(X_test)

        # Calculer le RMSE sur l'ensemble de test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Ajouter le RMSE à la liste des scores
        rmse_scores.append(rmse_test)
        
    # Calculer la moyenne des RMSE
    mean_rmse = np.mean(rmse_scores)
    
    print('RMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))


pls_seed_rmse(X, y, n_components, 0.2, 10)

###############################
########### PLS UNIQUE
###############################

# On sépare les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Nombre de composantes optimales pour D1 = 33

# Initialiser l'objet de régression PLS
pls = PLSRegression(n_components=n_components)

# Entraîner le modèle PLS sur l'ensemble d'entraînement complet
pls.fit(X_train, y_train)

# Prédire les valeurs sur l'ensemble de test
y_pred_test = pls.predict(X_test)
y_pred_train = pls.predict(X_train)

# Calculer le coefficient de détermination R² sur l'ensemble d'entraînement et l'ensemble de test
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Afficher les résultats
print('Coefficient de détermination R² sur l\'ensemble d\'entraînement: {:.3f}'.format(r2_train))
print('RMSE sur l\'ensemble d\'entraînement: {:.3f}'.format(rmse_train))
print('Coefficient de détermination R² sur l\'ensemble de test: {:.3f}'.format(r2_test))
print('RMSE sur l\'ensemble de test: {:.3f}'.format(rmse_test))

# Tracer les valeurs prédites et les valeurs réelles pour l'ensemble de test
plt.scatter(y_test, y_pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('JourF réel')
plt.ylabel('JourF prédit')
plt.title('Prédictions du modèle PLSR')
plt.show()
