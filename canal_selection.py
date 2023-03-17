import numpy as np
import pandas as pd
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
    window = gaussian(wavelength*2, std, sym=True)
    window = window[350:]
    window = np.concatenate((window,np.zeros(len(spectrum.iloc[0,:])-len(window))))

    # Apply the gaussian filter
    filtered_spectrum = window*spectrum

    # Filter the values below a certain threshold
    threshold = 0.01
    filtered_spectrum[filtered_spectrum < threshold] = 0

    # Return the filtered spectrum
    return filtered_spectrum



def select_canal(NIRS,canal_name):
    """
    Prend en entrée un ou plusieurs canaux ainsi que le spectre NIRS, et ressort ces informations comme 
    si la caméra multispectrale avait capturé ces informations.
    
    Args:
    - NIRS : le dataframe de NIRS à traiter.
    - canal_name : le dictionnaire des canaux à retourner

    Returns:
    - X : le dataframe de spectre NIRS avec seulement les informations des canaux sélectionnés

    Exemple:
    - select_canal(X,{R,NIR})
    """
    RGB,R,RE,NIR = (380,780),675,730,850
    
