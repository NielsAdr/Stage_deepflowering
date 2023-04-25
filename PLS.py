import os
import utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from pinard import preprocessing as pp
from tqdm import tqdm




os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/comparatif_annees")

# Importer les données 
X1,X2,X3,X4, y = pd.read_csv("X1.csv"), pd.read_csv("X2.csv"), pd.read_csv("X3.csv"), pd.read_csv("X4.csv"), pd.read_csv("Y_2022.csv")

X = pd.concat([X1, X2, X3, X4],axis=1)
X.columns = [f"{i+350}" for i in range(X.shape[1])]

# Définir le nombre de composantes à utiliser pour la PLS et la SEED
#n_components = 33
seed = 42

##################################

def compute_pls_rmse(X, y, n_components_range=(1,30), seed = 42):
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

n_components = compute_pls_rmse(X,y)

pls_seed_rmse(X, y, n_components, 0.2, 20)



###############################
########### ON TESTE PINARD
###############################

os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022/non concat")


# On charge les données
X1,X2,X3,X4, y = pd.read_csv("X1.csv"), pd.read_csv("X2.csv"), pd.read_csv("X3.csv"), pd.read_csv("X4.csv"), pd.read_csv("Y.csv")
X1,X2,X3,X4 = utils.adj_asd(X1,5),utils.adj_asd(X2,5),utils.adj_asd(X3,5),utils.adj_asd(X4,5)


X = pd.concat([X1, X2, X3, X4],axis=1)
X.columns = [f"{i+350}" for i in range(X.shape[1])]

preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("haar", pp.Wavelet("haar")),
        ("detrend", pp.Detrend()),
    ]

def compute_pinard(X, y, max_components=20, patience=3, seed=42):
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
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    
    # Mettre des labels sur les axes
    ax.set_xlabel('Floraison prédite', fontsize=14)
    ax.set_ylabel('Floraison réelle', fontsize=14)
    ax.tick_params(labelsize=12)
    
    ax.scatter(y_pred, y, c='b', s=50, alpha=0.5)
    
    plt.show()



def pinard_seed_rmse(X, y, n_components, test_size, num_splits, seed = 42):
    rmse_scores = []

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

        # Calculer le RMSE sur l'ensemble de test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Ajouter le RMSE à la liste des scores
        rmse_scores.append(rmse_test)
    
    # Calculer la moyenne des RMSE
    mean_rmse = np.mean(rmse_scores)
    
    print('\nRMSE sur l\'ensemble de test: {:.3f}'.format(mean_rmse))

    # Plot des prédictions
    plot_prediction(y_test,y_pred_test)
    return(mean_rmse)

# Afin de prédire si les rangs sont bien prédits, remplacer y par les rangs de floraison intra géno
# y = data_2022['rang']

n_components = compute_pinard(X, y)

pinard_seed_rmse(X, y, n_components, 0.2, 20)




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