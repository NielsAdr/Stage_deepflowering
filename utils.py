import numpy as np

def adj_asd(X, iadj, ws=5):
    """
    Elimine les sauts dus au passage d'un détecteur à un autre.
    On extrapole l'ajustement linéaire fait sur les [ws] points avant le saut, sur le point suivant.
    La différence entre l'ancien et le nouveau "point suivant" est soustraite à tous les points après le "point suivant".

    Args:
    - X (pandas.DataFrame): le dataframe à corriger.
    - iadj (list): la liste des indices de changements de détecteurs.
    - ws (int): le nombre de points utilisés pour ajuster la droite.

    Returns:
    - X (pandas.DataFrame): le dataframe corrigé.

    Exemple:
    - X_adj_cut = adj_asd(X, [650, 1450], 5)
    """
    for i in range(len(iadj)):
        x = np.arange(iadj[i] - ws + 1, iadj[i] + 1)
        Y = X.iloc[:, x]
        my = np.mean(Y, axis=1)

        sx = np.var(x)
        mx = np.mean(x)
        b = np.cov(x, Y)[0, 1:] / sx  # Ajustement linéaire.
        b0 = my - b * mx  # Ajustement linéaire.
        dif = X.iloc[:, iadj[i] + 1] - (b0 + b * (iadj[i] + 1))
        loremp = np.arange(iadj[i] + 1, X.shape[1])
        Xi = X.copy()
        Xi.iloc[:, loremp] = Xi.iloc[:, loremp].to_numpy() - np.kron(np.ones((1, Xi.shape[1] - iadj[i])), dif.to_numpy().reshape(-1, 1))  # On utilise kronecker pour remplacer repmat

    Xo = Xi
    return Xo