import numpy as np

def adj_asd(Xi, ws=10, iadj=[651,1451]):
    # Elimine les sauts dus au passage d'un dtecteur a un autre.
    # On extrapole l'ajustement linaire fait sur les [ws] points avant le saut, sur le point suivant
    # La diffrence entre l'ancien et le nouveau "point suivant" est soustraite a tous les points apres le "point suivant"
    # les sauts sont situés aux colonnes[651,1451]
    # ws=5
    # browser()
    Xo = Xi.copy()
    for i in range(len(iadj)):
        x = np.arange(iadj[i]-ws+1, iadj[i]+1)
        Y = Xo.iloc[:, x]
        my = np.mean(Y, axis=1)
        sx = np.var(x)
        mx = np.mean(x)
        b = np.cov(x, Y)[0, 1:] / sx  # Ajustement linaire.
        b0 = my - b * mx  # Ajustement linaire.
        dif = Xi.iloc[:, iadj[i]+1] - (b0 + b * (iadj[i]+1))
        loremp = np.arange(iadj[i], Xi.shape[1])
        Xo.iloc[:, loremp] = Xo.iloc[:, loremp] - np.kron(np.ones((1, Xi.shape[1]-iadj[i])), dif.values.reshape((-1, 1)))
        # On utilise np.kron pour remplacer np.tile
    return Xo


#new_X1 = adj_asd(X1,[651,1451], ws = 10)

#plt.plot(X1.T)

#plt.plot(new_X1.T)
