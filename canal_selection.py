import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import gaussian


def gaussian_spectrum(spectrum, wavelength, fwhm =10):
    """
    Apply a gaussian filter to a spectrum centered on a given wavelength and with a given FWHM.

    Args:
        spectrum (dataframe): the input spectrum with each column a spectrum index, each row a spectrum.
        wavelength (float): the central wavelength.
        fwhm (float): the FWHM of the gaussian filter.

    Returns:
        filtered_spectrum (numpy.ndarray): the filtered spectrum.

    """
    
    # Compute the standard deviation of the gaussian filter
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Create the gaussian filter window
    weights = gaussian(wavelength*2, std, sym=True)
    if len(weights) < len(spectrum.iloc[0,:]):
                          weights = np.concatenate((weights,np.zeros(len(spectrum.iloc[0,:])-len(weights))))
    else:
        weights = weights[0:len(spectrum.iloc[0,:])]

    # Apply the gaussian filter
    filtered_spectrum = weights*spectrum

    # Filter the values below a certain threshold
    threshold = 0.00001
    filtered_spectrum[filtered_spectrum < threshold] = 0

    # Return the filtered spectrum
    return filtered_spectrum, weights


def select_canals(spectrum, canal_names, start_wavelength = 350):
    """
    Takes as input one or more channels as well as the NIRS spectrum, and outputs this information as 
    if the multispectral camera had captured this information.
        
    Args:
    - spectrum: the NIRS dataframe to be processed.
    - canal_names : the dictionary of channels to return.
    
    Returns:
    - X: the NIRS spectrum dataframe with only ecteur de poids,the selected channel information.
    
    Example:
    - select_channel(X,{R,NIR})
    """
    
    # On introduit des variables locales afin de ne pas modifier les variables globales
    canals = canal_names.copy()
    spect = spectrum.copy()
    
    # add 350 empty columns at the beginning to keep the wavelength information
    new_cols = pd.DataFrame(np.zeros((len(spectrum.iloc[:,]), start_wavelength)))
    spect = pd.concat([new_cols,spect],axis=1)
         
    # On crée les arrays de sortie ainsi que les correspondances de couleur
    spectres = np.zeros((len(spect.iloc[:,0]),len(spect.iloc[0,:]),len(canals)))
    weights = np.zeros((len(spect.iloc[0,:]),len(canals)))
    couleurs  = {"B": 450,"G": 530,"GE":570,"RGB":(380,780),"R":675,"RE":730,"NIR":850}
    
    ### On vérifie si il y a la présence de RGB afin de le traiter différement
    if "RGB" in canals:
        rgb = spect.copy()
        rgb.iloc[:,0:(couleurs["RGB"][0]-int(rgb.columns[0]))],rgb.iloc[:,(couleurs["RGB"][1]-int(rgb.columns[0])+1):] = 0,0
        spectres[:,:,0] = rgb
        weights = np.resize(weights,(len(weights[:,0]), (len(canals)-1)))

        canals.remove("RGB")
    
    # On parcourt les canaux d'entrée et on traite avec la fonction gaussian_spect
        for i,canal in enumerate(canals):
            if canal in couleurs:
                print(canal)
                spectres[:,:,(i+1)],weights[:,i] = gaussian_spectrum(spect, couleurs[canal]) #fwhm ms est de 10 nm
    
    else:
        for i,canal in enumerate(canals):
            if canal in couleurs:
                print(canal)
                spectres[:,:,(i)],weights[:,i] = gaussian_spectrum(spect, couleurs[canal]) #fwhm ms est de 10 nm
    
    return spectres,weights


def sum_weighted(spectrum,weights,canal_names,fwhm = 10):
    """
    Takes as input one or more channels as well as the NIRS spectrum filtered and the weights applied on it, 
    and outputs this information as if the multispectral camera had captured this information on one pixel.
        
    Args:
    - spectrum: the NIRS array which has been processed.
    - canal_names: the dictionary of that are required.
    - weights : the weights that has been applied for each canal on each spectrum of the array
    
    Returns:
    - new_df: the value of the sum((xi*wi)/wi) for each canal and each tree
    
    Example:
    - select_channel(X,{R,NIR})
    """
    # On introduit des variables locales afin de ne pas modifier les variables globales
    canals = canal_names.copy()
    
    # On définit la section sur laquelle on va travailler et l'array que l'on va retourner
    delta_3 = 1.5*fwhm
    new_df = pd.DataFrame(np.zeros((len(spectrum[:,0,:]),len(spectrum[0,0,:]))))
    
    ### On vérifie si il y a la présence de RGB afin de le traiter différement
    if "RGB" in canals:
        new_df = new_df.rename(columns={new_df.columns[0]: "RGB"})
        canals.remove("RGB")
        new_df.iloc[:,0] = np.sum(spectrum[:,:,0],axis=1)
        for i,canal in enumerate(canals):
            new_df = new_df.rename(columns={new_df.columns[i+1]: canal})
            # On prend l'indice maximal dans les poids pour centrer et on prend la somme +/- delta_3
            maxi = np.argmax(weights[:,i])+1 #On rajoute 1 à l'index, sinon la valeur n'est pas la bonne
            s = np.sum(spectrum[:,int(maxi-delta_3):int(maxi+delta_3),i+1],axis=1) # i+1 car RGB
            w = np.sum(weights[int(maxi-delta_3):int(maxi+delta_3),i],axis=0) # juste i car pas de poids RGB
            new_df.iloc[:,i+1] = s/w
    else:
        for i,canal in enumerate(canals):
            new_df = new_df.rename(columns={new_df.columns[i]: canal})
            # On prend l'indice maximal dans les poids pour centrer et on prend la somme +/- delta_3
            maxi = np.argmax(weights[:,i])+1 #On rajoute 1 à l'index, sinon la valeur n'est pas la bonne
            s = np.sum(spectrum[:,int(maxi-delta_3):int(maxi+delta_3),i],axis=1) # i+1 car RGB
            w = np.sum(weights[int(maxi-delta_3):int(maxi+delta_3),i],axis=0) # juste i car pas de poids RGB
            new_df.iloc[:,i] = s/w
    return new_df

################################
########## TEST DE SELECT_CANALS
################################

os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/sans NA/non concat/equilibres")
X = pd.read_csv("X_out_D1.csv")
canals = {"RGB","RE","R","NIR","B","G","GE"}

spectres_filtres, poids = select_canals(X, canals)
test = sum_weighted(spectres_filtres, poids, canals)

################################
########## PLOTS
################################

# Tout X
for i in range(len(X)):
    plt.plot(X.columns, X.iloc[i])
    
# La 1e dimension de spectres_filtres
spect_1 = pd.DataFrame(spectres_filtres[:,:,0])
for i in range(len(spect_1)):
    plt.plot(spect_1.columns, spect_1.iloc[i])

# La 2e dimension de spectres_filtres
spect_2 = pd.DataFrame(spectres_filtres[:,:,1])
for i in range(len(spect_2)):
    plt.plot(spect_2.columns, spect_2.iloc[i])

