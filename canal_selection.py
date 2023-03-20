import numpy as np
import pandas as pd
import os
from scipy.signal import gaussian


def gaussian_spectrum(spectrum, wavelength, fwhm):
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
    - channel_name: the dictionary of channels to return.
    
    Returns:
    - X: the NIRS spectrum dataframe with only the selected channel information.
    
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
                spectres[:,:,(i+1)],weights[:,i] = gaussian_spectrum(spect, couleurs[canal], 10) #fwhm ms est de 10 nm
    
    else:
        for i,canal in enumerate(canals):
            if canal in couleurs:
                print(canal)
                spectres[:,:,(i)],weights[:,i] = gaussian_spectrum(spect, couleurs[canal], 10) #fwhm ms est de 10 nm
    
    return spectres,weights

################################
########## TEST DE SELECT_CANALS
################################

os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/sans NA/concat/X_Y")
X = pd.read_csv("X_out_concat.csv")
canals = {"RGB","RE","R","NIR"}

spectres_filtres, poids = select_canals(X, canals)
