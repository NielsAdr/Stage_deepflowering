import os
import utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from pinard import preprocessing as pp
from tqdm import tqdm




os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/comparatif_annees") # Pour les spectres


# Importer les données 
X1,X2,X3,X4, y = pd.read_csv("X1.csv"), pd.read_csv("X2.csv"), pd.read_csv("X3.csv"), pd.read_csv("X4.csv"), pd.read_csv("Y_2022.csv")

X = pd.concat([X1, X2, X3, X4],axis=1)
X.columns = [f"{i+350}" for i in range(X.shape[1])]

# Définir le nombre de composantes à utiliser pour la PLS et la SEED
#n_components = 33
seed = 42

##################################

def compute_pls_rmse(X, y, n_components_range=(1,21), seed = 42):
    rmse = []
    bestcomp, RMSEopt = 1, 100
    
    for i in tqdm(range(n_components_range[0], n_components_range[1])):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        # Initialiser l'objet de régression PLS
        pls = PLSRegression(n_components=i)

        # Effectuer une cross-validation à 20-fold avec prédiction
        y_pred = cross_val_predict(pls, X_train, y_train, cv=20)

        # Calculer le RMSE sur les prédictions
        rmse_i = np.sqrt(mean_squared_error(y_train, y_pred))
        rmse.append(rmse_i)
        print("\nnombre de composantes = " + str(i))
        print("RMSE = " + str(rmse_i))

        if rmse_i < RMSEopt:
            RMSEopt = rmse_i
            bestcomp = i

    print("Le meilleur nombre de composantes est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
    plt.plot(range(n_components_range[0], n_components_range[1]),rmse)
    return bestcomp



def pls_seed_rmse(X, y, n_components, test_size, num_splits, seed = 42):
    rmse_scores = []
    for i in tqdm(range(num_splits)):
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
    
    print('\nRMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))
    
    # Calculer la variance capturé
    variance_capt = round((1-(mean_rmse**2)/np.var(y))*100,ndigits = 2)
    
    if isinstance(variance_capt, pd.core.series.Series):
        variance_capt = variance_capt.values[0]

    print("\nLe pourcentage de variance capturée est de : {:.2f} %".format(variance_capt))


n_components = compute_pls_rmse(X,y)

pls_seed_rmse(X1, NDVI_juin, n_components, 0.2, 20)



###############################
########### ON TESTE PINARD
###############################

os.chdir("/media/u108-s786/Donnees/Stage_Niels/NIRS/csv_x_y_cnn/2021/sans NA/non concat/equilibres") #2021
os.chdir("/media/u108-s786/Donnees/Stage_Niels/NIRS/csv_x_y_cnn/2022/non concat") #2022

# On charge les données
X1,X2,X3,X4, y = pd.read_csv("X1.csv"), pd.read_csv("X2.csv"), pd.read_csv("X3.csv"), pd.read_csv("X4.csv"), pd.read_csv("Y.csv")
#X1,X2,X3,X4 = utils.adj_asd(X1,5),utils.adj_asd(X2,5),utils.adj_asd(X3,5),utils.adj_asd(X4,5)


X = pd.concat([X1, X2, X3, X4],axis=1)
X.columns = [f"{i+350}" for i in range(X.shape[1])]

preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("haar", pp.Wavelet("haar")),
        ("detrend", pp.Detrend()),
    ]

def compute_pinard(X, y, max_components=20, patience=5, seed=42):
    rmse = []
    bestcomp, RMSEopt = 1, 100
    count = 0
    
    for i in tqdm(range(1, max_components+1)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
        # Initialiser le pipeline
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()), # scaling the data
            ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
            ('PLS',  PLSRegression(n_components=i)) # regressor
        ])
        
        # Estimator including y values scaling
        estimator = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
        
        # # Compute metrics on the test set
        Y_preds = cross_val_predict(estimator, X_train, y_train, cv=20)
            
        # Calculer le RMSE sur les prédictions
        rmse_i = np.sqrt(mean_squared_error(y_train, Y_preds))
        rmse.append(rmse_i)
        print("\nnombre de composantes = " + str(i))
        print("RMSE = " + str(rmse_i))
        
        if rmse_i < RMSEopt:
            RMSEopt = rmse_i
            bestcomp = i
            count = 0
        else:
            count += 1
            if count >= patience:
                print("Le nombre de composantes optimal est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
                plt.plot(range(1, i+1), rmse)
                return bestcomp

    print("Le nombre de composantes optimal est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
    plt.plot(range(1, max_components+1), rmse)
    return bestcomp



def plot_prediction(y,y_pred):
    fig, ax = plt.subplots()
    
    # Tracer la ligne d'identité
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', lw=2, label = 'x=y')
        
    # Tracer la tendance affine
    a, b = np.polyfit(y_pred.ravel(), y, 1)
    x = np.linspace(y.min(), y.max())
    ax.plot(x, a*x + b, 'r-', label = 'Tendance affine')
    
    # Calculer le R2
    r = np.corrcoef(y_pred.ravel(),y.values.ravel())[0,1]
    r2 = r**2
    a,b = float(a),float(b)
    ax.text(0.04, 0.8, f'R2 = {r2:.3f}', transform=ax.transAxes, fontsize=14)
    ax.text(0.04, 0.75, f'a= {a:.3f}', transform=ax.transAxes, fontsize=12)
    ax.text(0.04, 0.70, f"b= {b:.3f}", transform=ax.transAxes, fontsize=12)
    
    # Mettre des labels sur les axes
    ax.set_xlabel('Valeur prédite', fontsize=14)
    ax.set_ylabel('Valeur réelle', fontsize=14)
    ax.tick_params(labelsize=12)
    
    ax.scatter(y_pred, y, c='b', s=50, alpha=0.5)
    
    # Ajouter une légende
    ax.legend(fontsize=12)
    
    # Afficher un graphique carré
    fig.set_size_inches(6, 6)

    plt.show()



def pinard_seed_rmse(X, y, n_components, test_size, num_splits, seed = 42):
    rmse_scores = []
    y_pred_tests = []

    for i in tqdm(range(num_splits)):
        # On sépare les données en train et test avec une seed différente à chaque itération
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed+i)
        
        # Initialiser le pipeline
        pipeline = Pipeline([
           ('scaler', MinMaxScaler()), # scaling the data
           ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
           ('PLS',  PLSRegression(n_components=n_components)) # regressor
       ])
       
        # Estimator including y values scaling
        estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

        # Entraîner le modèle PLS sur l'ensemble d'entraînement complet
        estimator.fit(X_train, y_train)

        # Prédire les valeurs sur l'ensemble de test
        y_pred_test = estimator.predict(X_test)
        y_pred_tests.append(y_pred_test)

        # Calculer le RMSE sur l'ensemble de test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Ajouter le RMSE à la liste des scores
        rmse_scores.append(rmse_test)

    # Calculer la moyenne des RMSE
    mean_rmse = np.mean(rmse_scores)
    
    print('\nRMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))
    
    # Calculer la variance capturé
    variance_capt = round((1-(mean_rmse**2)/np.var(y))*100,ndigits = 2)
    
    if isinstance(variance_capt, pd.core.series.Series):
        variance_capt = variance_capt.values[0]

    print("\nLe pourcentage de variance capturée est de : {:.2f} %".format(variance_capt))

    # Calculer la moyenne des prédictions pour chaque observation
    #mean_y_pred_test = np.mean(y_pred_tests, axis=0)
    
    # Plot des prédictions
    plot_prediction(y_test,y_pred_test)
    return(mean_rmse)

# Afin de prédire si les rangs sont bien prédits, remplacer y par les rangs de floraison intra géno
# y = data_2022['rang']

n_components = compute_pinard(X_bis, y, patience= 5)

n_components = compute_pinard(X, NDVI_octobre, patience= 5)

pinard_seed_rmse(X, y, n_components, 0.2, 20)





##################### ON FAIT DU ML SUR LES FEATURES D'IMAGES

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error

os.chdir('/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/ms/') # Pour les images segmentées automatiquement
os.chdir("/media/u108-s786/Donnees/Stage_Niels/Orthomosaiques/") # Pour les images segmentées manuellement

# Importer les données image
# Pour les images segmentées manuellement
X = pd.read_csv('manual_index_surface_ndvi_over_time.csv', usecols=lambda column: column not in ['rang', 'pos', 'jourF','mean','delta'])
y = pd.read_csv('manual_index_surface_ndvi_over_time.csv')['jourF'] #choisir mean, jourF ou delta selon ce que l'on veut

# Pour les images segmentées automatiquement
X,y = pd.read_csv('features_concatenees.csv', usecols=lambda column: column not in ['rang', 'pos', 'jourF']),  pd.read_csv('features_concatenees.csv')['jourF']

y = y.astype(int)
###### Fonction de plot de prédiction

def plot_prediction(y,y_pred):
    fig, ax = plt.subplots()
    
    # Tracer la ligne d'identité
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', lw=2, label = 'x=y')
        
    # Tracer la tendance affine
    a, b = np.polyfit(y_pred.ravel(), y, 1)
    x = np.linspace(y.min(), y.max())
    ax.plot(x, a*x + b, 'r-', label = 'Tendance affine')
    
    # Calculer le R2
    r = np.corrcoef(y_pred.ravel(),y.values.ravel())[0,1]
    r2 = r**2
    a,b = float(a),float(b)
    ax.text(0.04, 0.8, f'R2 = {r2:.3f}', transform=ax.transAxes, fontsize=14)
    ax.text(0.04, 0.75, f'a= {a:.3f}', transform=ax.transAxes, fontsize=12)
    ax.text(0.04, 0.70, f"b= {b:.3f}", transform=ax.transAxes, fontsize=12)
    
    # Mettre des labels sur les axes
    ax.set_xlabel('Valeur prédite', fontsize=14)
    ax.set_ylabel('Valeur réelle', fontsize=14)
    ax.tick_params(labelsize=12)
    
    ax.scatter(y_pred, y, c='b', s=50, alpha=0.5)
    
    # Ajouter une légende
    ax.legend(fontsize=12)
    
    # Afficher un graphique carré
    fig.set_size_inches(6, 6)

    plt.show()

###### UN Random Forest

def RF_seed(X,y,test_size,seed = 42):
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Listes des valeurs à tester pour n_estimators et max_depth
    n_estimators_list = [50, 100, 150]
    max_depth_list = [5, 10, 15]
    
    # Boucle pour tester différentes combinaisons d'hyperparamètres
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            # Configuration du modèle Random Forest
            random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    
            # Entraînement du modèle
            random_forest.fit(X_train, y_train)
    
            # Validation croisée sur l'ensemble d'entraînement
            cross_val_scores = cross_val_score(random_forest, X_test, y_test, cv=20, scoring='neg_mean_squared_error')
            test_rmse = np.sqrt(-cross_val_scores.mean())
            variance_captured = (1 - (test_rmse ** 2 / np.var(y)))*100
    
            # Affichage des résultats pour la combinaison d'hyperparamètres actuelle
            print("n_estimators:", n_estimators)
            print("max_depth:", max_depth)
            print("Test RMSE:", test_rmse)
            print("Test Variance Captured: {:.2f} %".format(variance_captured))
            print("------------------------")
    
RF_seed(X,y,0.2)

###### UN XGBoost

def XGBRegressor_seed(X, y, test_size, num_splits, seed = 42):
    rmse_scores = []
    y_pred_tests = []

    for i in tqdm(range(num_splits)):
        # On sépare les données en train et test avec une seed différente à chaque itération
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed+i)
        
        # Créer le modèle XGBoost
        model = xgb.XGBRegressor()

        # Entraîner le modèle sur les données d'entraînement
        model.fit(X_train, y_train)

        # Prédire les valeurs sur l'ensemble de test
        y_pred_test = model.predict(X_test)
        y_pred_tests.append(y_pred_test)

        # Calculer le RMSE sur l'ensemble de test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Ajouter le RMSE à la liste des scores
        rmse_scores.append(rmse_test)

    # Calculer la moyenne des RMSE
    mean_rmse = np.mean(rmse_scores)
    
    print('\nRMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))
    
    # Calculer la variance capturé
    variance_capt = round((1-(mean_rmse**2)/np.var(y))*100,ndigits = 2)
    
    if isinstance(variance_capt, pd.core.series.Series):
        variance_capt = variance_capt.values[0]

    print("\nLe pourcentage de variance capturée est de : {:.2f} %".format(variance_capt))

    # Calculer la moyenne des prédictions pour chaque observation
    #mean_y_pred_test = np.mean(y_pred_tests, axis=0)
    
    # Plot des prédictions
    plot_prediction(y_test,y_pred_test)
    return(mean_rmse)

XGBRegressor_seed(X,y,0.2,20)


# Test weka imageJ pour remplacer DEM et watershed via imageJ








##################### TESTS PLS POUR UN DATASET AVEC N REPETITIONS PAR INDIVIDU, AVEC POS ET RANG

preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("haar", pp.Wavelet("haar")),
        ("detrend", pp.Detrend()),
    ]

def compute_pinard_repetitions(dataframe, max_components=20, patience=10, seed=42):
    rmse = []
    bestcomp, RMSEopt = 1, 100
    count = 0
    # On détermine le nombre d'individus uniques
    individus_uniques = dataframe[['rang', 'pos']].drop_duplicates()
    
    for i in tqdm(range(1, max_components+1)):
        # On sépare en 2 groupes train et test
        train_pos, test_pos = train_test_split(individus_uniques, test_size=0.2, random_state=seed)
        
        # Fusionner les données avec les valeurs de rang et pos pour chaque ensemble
        train_data = pd.merge(train_pos, dataframe, on=['rang', 'pos'], how='inner')
        
        # Définir y_train et X_train
        y_train = train_data['jourF']
        X_train = train_data.drop(columns=['jourF','pos','rang'])
    
        # Initialiser le pipeline
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()), # scaling the data
            ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
            ('PLS',  PLSRegression(n_components=i)) # regressor
        ])
        
        # Estimator including y values scaling
        estimator = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
        
        # # Compute metrics on the test set
        Y_preds = cross_val_predict(estimator, X_train, y_train, cv=20)
            
        # Calculer le RMSE sur les prédictions
        rmse_i = np.sqrt(mean_squared_error(y_train, Y_preds))
        rmse.append(rmse_i)
        print("\nnombre de composantes = " + str(i))
        print("RMSE = " + str(rmse_i))
        
        if rmse_i < RMSEopt:
            RMSEopt = rmse_i
            bestcomp = i
            count = 0
        else:
            count += 1
            if count >= patience:
                print("Le nombre de composantes optimal est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
                plt.plot(range(1, i+1), rmse)
                return bestcomp

    print("Le nombre de composantes optimal est : " + str(bestcomp) + " avec un RMSE de " + str(RMSEopt))
    plt.plot(range(1, max_components+1), rmse)
    return bestcomp


def pinard_seed_rmse_repetitions(dataframe, n_components, test_size, num_splits, seed = 42):
    rmse_scores = []
    # On détermine le nombre d'individus uniques
    individus_uniques = dataframe[['rang', 'pos']].drop_duplicates()
    
    for i in tqdm(range(num_splits)):
        # On sépare en 2 groupes train et test
        train_pos, test_pos = train_test_split(individus_uniques, test_size=0.2, random_state=seed)
        
        
        # Fusionner les données avec les valeurs de rang et pos pour chaque ensemble
        train_data = pd.merge(train_pos, dataframe, on=['rang', 'pos'], how='inner')
        test_data = pd.merge(test_pos, dataframe, on=['rang', 'pos'], how='inner')
        
        # Définir y_train/test et X_train/test
        y_train = train_data['jourF']
        y_test = test_data['jourF']
        X_train = train_data.drop(columns=['jourF','pos','rang'])
        X_test = test_data.drop(columns=['jourF','pos','rang'])
        
        # Définir y pour le std
        y = pd.concat([y_train,y_test], axis=0)
        
        # Initialiser le pipeline
        pipeline = Pipeline([
           ('scaler', MinMaxScaler()), # scaling the data
           ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
           ('PLS',  PLSRegression(n_components=n_components)) # regressor
       ])
       
        # Estimator including y values scaling
        estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

        # Entraîner le modèle PLS sur l'ensemble d'entraînement complet
        estimator.fit(X_train, y_train)

        # Prédire les valeurs sur l'ensemble de test
        y_pred_test = estimator.predict(X_test)

        # Calculer le RMSE sur l'ensemble de test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Ajouter le RMSE à la liste des scores
        rmse_scores.append(rmse_test)
    
    # Calculer la moyenne des RMSE
    mean_rmse = np.mean(rmse_scores)
    
    print('\nRMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))
    
    # Calculer la variance capturé
    variance_capt = round((1-(mean_rmse**2)/np.var(y))*100,ndigits = 2)
    
    print("\nLe pourcentage de variance capturée est de : {:.2f} %".format(variance_capt[0]))

    # Plot des prédictions
    plot_prediction(y_test,y_pred_test)
    return(mean_rmse)


n_components = compute_pinard_repetitions(D_corresp)

pinard_seed_rmse_repetitions(D_corresp, n_components, 0.2, 20)





##################### TEST DE L'IMPORTANCE DES FEATURES

# Ajouter le fait qu'il calcule d'abord le RMSE sur test normal, et qu'il retire cette valeur plutot que le rmse min, et on divise par cette valeur aussi (x100 pour un % de différence /r à la base)
def feature_importance(X, y , n_components, n_transposition, len_split, seed = 42):
    rmse_scores = []
    
    rmse_base = pinard_seed_rmse(X, y, n_components, 0.2, 20)
    print("Le RMSE de base de ce dataframe est de : ",rmse_base)
    
    n_splits = int(X.shape[1]/len_split)
    
    # Initialiser le pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()), # scaling the data
        ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
        ('PLS',  PLSRegression(n_components=n_components)) # regressor
    ])
     
    for i in tqdm(range (n_splits)):
        print("\nSplit numéro ",(i+1)," sur ", (n_splits))
        print("De ",350+i*len_split,"nm à ",350+(i+1)*len_split,"nm.")
        rmse = []
        for j in range (n_transposition):
            # On fait une copie du dataframe originel à chaque itération
            new_X = X.copy()
            
            # On ajoute 1 à la seed
            seed += 1
            # On définit une seed de transposition, qui change à chaque itération de j
            np.random.seed(seed)
            
            # On sélectionne le segment à transposer
            segment = X.iloc[:,i*len_split:(i+1)*len_split]
            
            # Choix aléatoire d'un autre segment à transposer
            other_segment_index = np.random.choice([k for k in range(n_splits) if k != i])
            other_segment = X.iloc[:,other_segment_index *len_split:(other_segment_index +1)*len_split]
            
            # On intervertit/transpose les 2 éléments
            new_X.iloc[:,i*len_split:(i+1)*len_split] = other_segment
            new_X.iloc[:,other_segment_index *len_split:(other_segment_index +1)*len_split] = segment
            
            # On sépare les données en train et test avec une seed pour que ce soit toujours le même split
            X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=seed)
            
            # Estimator including y values scaling
            estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

            # Entraîner le modèle PLS sur l'ensemble d'entraînement complet
            estimator.fit(X_train, y_train)

            # Prédire les valeurs sur l'ensemble de test
            y_pred_test = estimator.predict(X_test)

            # Calculer le RMSE sur l'ensemble de test
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Ajouter le RMSE à la liste de RMSE du segment i
            rmse.append(rmse_test)
        
        # On prend la moyenne des RMSE et on l'ajoute à la liste de scores de RMSE
        rmse_scores.append(np.array(rmse).mean())
        
        # On print le RMSE obtenu sur ce split
        print("\nRMSE : ",rmse_scores[i])
    
    # On retire la valeur minimale de rmse_scores à toutes les valeurs de rmse_scores afin d'avoir l'importance des features
    rmse_scores = ((np.array(rmse_scores)-rmse_base)/rmse_base)*100
    
    # Créer un histogramme
    plt.hist(np.arange(len_split, n_splits * len_split + len_split, len_split), bins=n_splits, weights=rmse_scores, range=(350, 350+(n_splits*len_split)), edgecolor='black')
    plt.title("RMSE par tranche de " + str(len_split))
    plt.xlabel("Tranche de longueur d'onde")
    plt.ylabel("Feature importance")
    plt.show()
    
    return(rmse_scores)


n_components = compute_pinard(X, y)

importance = feature_importance(X, y, n_components, n_transposition=10, len_split=2151)