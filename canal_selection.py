import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.stats import linregress
import patch_to_array


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
    wavelength = wavelength + 60
    
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

########### CALCUL DE LA VALEUR MOYENNE PAR PATCH D'ARBRE

# Pour caculer la valeur moyenne des pixels
def calculate_average_pixel_values(data, month = None, canal = None):
    """
    This function allow ou to calculate the average pixel value (APV) of every canal and every month for every trees (return a 3D array),
    or either the APV of every canal for your month for every trees (return a 2D array),
    or either the APV of every months for your canal for every trees (return a 2D array),
    or either the APV of a canal of a month for every trees (return a 1D array).
    
    Parameters
    ----------
    data : array 5D (d,t,x,y,c) or 4D (d,t,x,y)/(t,x,y,c) or 3D (t,x,y) with d the months, t the trees, 
    x and y the dimensions of the patch_tree and c the canals
    
    month : if data got d as a dimension, let it as None, else put anything
    
    canal : if data got c as a dimension, le it as None, else put anything

    Returns
    -------
    results : Array 3D (d,t,c) or 2D (d,t)/(t,c) or 1D(t) depends on the entry

    """
    
    if month is not None:
        data = data[np.newaxis,:]
    
    if canal is not None:
        data = data[...,np.newaxis]

    # get dates
    num_dates = data.shape[0]
    
    # get number of trees
    num_trees = data.shape[1]
    
    # get number of channels
    num_channels = data.shape[4]
    
    # initialize results array
    results = np.zeros((num_dates, num_trees, num_channels))

    # loop over dates and trees
    for d in range(num_dates):
        for t in range(num_trees):
            # loop over channels
            for c in range(num_channels):
                # get pixel values for this channel
                pixel_values = data[d,t, 80:120, 80:120, c]

                # calculate average pixel value
                avg_pixel_value = np.mean(pixel_values)

                # store result in results array
                results[d,t,c] = avg_pixel_value
    if month is not None:
        if canal is not None:
            return results[0,:,0]
        return results[0,:,:]
    elif canal is not None:
        return results[:,:,0]
    else:
        return results


########### TROUVE LE MEILLEUR PIXEL PAR PATCH D'ARBRE

def find_best_pixel(data, NIRS, month = None, canal = None):
    """
    This function allow ou to find the best pixel value (BPV) of every canal and every month for every trees (return a 3D array),
    or either the BPV of every canal for your month for every trees (return a 2D array),
    or either the BPV of every months for your canal for every trees (return a 2D array),
    or either the BPV of a canal of a month for every trees (return a 1D array).
    
    Parameters
    ----------
    data : array 5D (d,t,x,y,c) or 4D (d,t,x,y)/(t,x,y,c) or 3D (t,x,y) with d the months, t the trees, 
    x and y the dimensions of the patch_tree and c the canals
    
    NIRS : a vector with t values of a canal
    
    month : if data got d as a dimension, let it as None, else put anything
    
    canal : if data got c as a dimension, le it as None, else put anything

    Returns
    -------
    results : Array 3D (d,t,c) or 2D (d,t)/(t,c) or 1D(t) depends on the entry

    """
    
    if month is not None:
        data = data[np.newaxis,:]
    
    if canal is not None:
        data = data[...,np.newaxis]

    # get dates
    num_dates = data.shape[0]
    
    # get number of trees
    num_trees = data.shape[1]
    
    # get number of channels
    num_channels = data.shape[4]
    
    # initialize results arrays
    results = np.zeros((num_dates, num_trees, num_channels))
    indices = np.zeros((num_dates, num_trees, num_channels, 2), dtype=int)


    # loop over dates and trees
    for d in range(num_dates):
        for t in range(num_trees):
            # loop over channels
            for c in range(num_channels):
                # get pixel values for this channel
                pixel_values = data[d,t,:,:, c]

                # Calculate the euclidian distance between the NIRS value and each pixel of the dataframe
                distances = pd.DataFrame(np.sqrt((pixel_values - NIRS.iloc[t])**2))

                # Find the indice of the pixel closest to the NIRS value
                best_pixel_index = distances.stack().idxmin()
                
                # Take the corresponding value
                best_pixel = pixel_values[best_pixel_index[0],best_pixel_index[1]]

                # store result in results array
                results[d,t,c] = best_pixel
                
                # Store index of best pixel in indices array
                indices[d, t, c,:] = [best_pixel_index[0], best_pixel_index[1]]
               
    if month is not None:
        if canal is not None:
            return results[0,:,0], indices[0,:,0,:]
        return results[0,:,:], indices[0,:,:,:]
    elif canal is not None:
        return results[:,:,0], indices[:,:,0,:]
    else:
        return results, indices



########### MAKE A CURSOR AROUND THE BEST PIXEL

def add_cursor(indexes, images):
    """
    Adds a cursor around the specified pixel indices in the image
    
    Args:
    - indexes: a (t, 2) numpy array containing the (x,y) indices of the pixels of interest
    - image: a 3D numpy array containing the image data
    
    Returns:
    - an updated 3D numpy array with a cursor around each specified pixel
    """
    img_cursor = images.copy()
    for i in range(len(indexes)):
        x, y = indexes[i]
        img_cursor[i, max(0, x - 3):min(images.shape[1], x + 4), y] = 1.5
        img_cursor[i, x, max(0, y - 3):min(images.shape[2], y + 4)] = 1.5
    return img_cursor


########### MAKE A CORRELATION MAP FOR A TREE

def compute_correlations(image, vector):
    # Create an empty array to store the correlation coefficients for each pixel
    image_correlations = np.empty((image.shape[0], image.shape[1]))

    # Loop over each pixel and compute the correlation coefficient
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_values = image[i, j, :]
            correlation_coefficient = np.corrcoef(pixel_values, vector)[0, 1]
            image_correlations[i, j] = correlation_coefficient**2

    # Convert the correlation coefficients to a Pandas DataFrame
    df = pd.DataFrame(image_correlations)

    return df

    
# Pour avoir les pixels moyens
#average_pixel_value = calculate_average_pixel_values(data_2022_ms)
#juin,septembre,octobre,novembre = pd.DataFrame(average_pixel_value[0,:,:]),pd.DataFrame(average_pixel_value[1,:,:]),pd.DataFrame(average_pixel_value[2,:,:]),pd.DataFrame(average_pixel_value[3,:,:])

## Les canaux sont dans un ordre croissant de longueur d'onde.
#juin.columns,septembre.columns,octobre.columns,novembre.columns = np.tile(["B","G","GE","R","RE","NIR"],(4,1))

########### NORMALISATION DU CHANNEL SOUHAITÉ OU CALCUL D'INDICES (NDVI/NDRE)

# Le but ici est de faire une fonction, et que selon le channel/indice souhaité, la fonction retourne
# Le channel normalisé à '99%' du max ou l'indice (qui du coup est normalisé) afin de comparer la corrélation
# Avec les 'pixels' NIRS.

def normalize_canal(array):
    """

    Parameters
    ----------
    array : Array 3D (t,x,y) d'un canal MS ou RGB, avec t le nombre d'arbres et x et y les pixels de chaque patch d'arbre

    Returns
    -------
    array_normalized : Array 3D (t,x,y) normalisé par la valeur à 99% (pour éviter les outliers)

    """
    percentile_99 = np.percentile(array, q=99,axis=None)
    array_normalized = array/percentile_99
    return array_normalized


def select_canal_or_index(array,selection = "NDVI"):
    """

    Parameters
    ----------
    array : array 4 Dimensions (t,x,y,c) d'un mois
    
    selection : set or str of the choosen canals to normalize or the indice to calculate (NDVI,NDRE,etc...)

    Returns
    -------
    new_array : 3D array (t,x,y) corresponding to the selection

    """
    
    canals = {"B": 0, "G": 1, "GE": 2, "R": 3, "RE": 4, "NIR":5}
    
    if isinstance(selection, str):
        if selection in canals :
            canal_selected = canals[selection]
            new_array = normalize_canal(array[:,:,:,canal_selected])
            return (new_array)
            
        elif selection == "NDVI":
            NIR,R = normalize_canal(array[:,:,:,canals["NIR"]]), normalize_canal(array[:,:,:,canals["R"]])
            new_array = np.zeros_like(array[:,:,:,0])
            new_array = (NIR-R)/(NIR+R)
            return (new_array)
        
        elif selection == "NDRE":
            NIR,RE = normalize_canal(array[:,:,:,canals["NIR"]]), normalize_canal(array[:,:,:,canals["RE"]])
            new_array = np.zeros_like(array[:,:,:,0])
            new_array = (NIR-RE)/(NIR+RE)
            return (new_array)
        
        else :
            raise TypeError("Your selection is either not a canal from the multispectral camera or an index provided by the code")
        
    elif isinstance(selection, set):
        new_array = np.zeros_like(array[:,:,:,0:len(selection)])
        new_array = new_array.astype('float64')
        for i,canal in enumerate(selection):
            canal_selected = canals[canal]
            new_array[:,:,:,i] = normalize_canal(array[:,:,:,canal_selected])
        return new_array
    
    else :
        raise TypeError("Your selection is either not a canal from the multispectral camera or an index provided by the code")


##############################################################################
########## IMPORTER LES DATA NIRS ET LES TRANSFORMER
##############################################################################


os.chdir("/media/u108-s786/Donnees/NIRS data D1-D7/csv_x_y_cnn/2022")
X = pd.read_csv("non concat/X1.csv")
canals = {"RE","R","NIR","B","G","GE"}

spectres_filtres, poids = select_canals(X, canals)
average_NIRS_value = sum_weighted(spectres_filtres, poids, canals)
average_NIRS_value = average_NIRS_value[["B","G","GE","R","RE","NIR"]]


##############################################################################
####### TRANSFORMATION DES IMAGES MS EN 1 "PIXEL" PAR CANAL PAR ARBRE PAR DATE
##############################################################################

########### IMPORTATION DES DONNÉES DE DRONE
path_patch = "/media/u108-s786/Donnees/FruitFlowDrone/data/Patch"
data_2022 = pd.read_csv("data_2022.csv")
type_data='ms'

data_2022_ms = patch_to_array(data_2022,type_data,path_patch)


juin = data_2022_ms[0,:,:,:,:]
septembre = data_2022_ms[1,:,:,:,:]
octobre = data_2022_ms[2,:,:,:,:]
novembre = data_2022_ms[3,:,:,:]

# On prend les canaux de juin
NIR_juin = select_canal_or_index(juin,"NIR")
R_juin = select_canal_or_index(juin,"R")
NDVI_juin = select_canal_or_index(juin,"NDVI")

# On prend le NDVI de septembre
NDVI_septembre = select_canal_or_index(septembre,"NDVI")

# On prend le NDVI d'octobre
NDVI_octobre = select_canal_or_index(octobre,"NDVI")

# On prend les canaux de novembre
NIR_novembre = select_canal_or_index(novembre,"NIR")
R_novembre = select_canal_or_index(novembre,"R")
NDVI_novembre = select_canal_or_index(novembre,"NDVI")


# On veut l'average NDVI pour les 4 dates
# Voir avec le NDVI de la moyenne plutôt
NDVI = np.stack((NDVI_juin,NDVI_septembre,NDVI_octobre,NDVI_novembre))
average_NDVI = np.transpose(calculate_average_pixel_values(NDVI,canal="NDVI"))



########### PLOT DE CORRÉLATION

###### Pour le NIR

## Average value
juin_average_NIR = calculate_average_pixel_values(NIR_juin,"juin","NIR")
NIR_NIRS = (average_NIRS_value.iloc[:,5])

# Calculer le r carré et la pente, etc ...
slope, intercept, r_value, p_value, std_err = linregress(NIR_NIRS, juin_average_NIR)

# Tracer le graphique de dispersion
plt.scatter(NIR_NIRS, juin_average_NIR)
plt.plot(NIR_NIRS,slope*NIR_NIRS+intercept, color='red')

# Ajouter des titres d'axes et un titre de graphique
plt.xlabel('NIRS')
plt.ylabel('Drone')
plt.title('Graphique de dispersion du canal NIR drone en fonction de NIRS - juin')

# Afficher le graphique
plt.show()
print("Le r2 est égal à : ",r_value**2)



## Best value
juin_best_NIR,juin_best_NIR_indexes = find_best_pixel(NIR_juin, average_NIRS_value['NIR'],"juin","NIR")
NIR_NIRS = (average_NIRS_value.iloc[:,5])

# Calculer le r carré et la pente, etc ...
slope, intercept, r_value, p_value, std_err = linregress(NIR_NIRS, juin_best_NIR)

# Tracer le graphique de dispersion
plt.scatter(NIR_NIRS, juin_best_NIR)
plt.plot(NIR_NIRS,slope*NIR_NIRS+intercept, color='red')

# Ajouter des titres d'axes et un titre de graphique
plt.xlabel('NIRS')
plt.ylabel('Drone')
plt.title('Graphique de dispersion du canal NIR drone en fonction de NIRS - juin')

# Afficher le graphique
plt.show()
print("Le r2 est égal à : ",r_value**2)

# On ajoute le curseur
NIR_juin_cursor = add_cursor(juin_best_NIR_indexes, NIR_juin)



###### Pour R == > r² = 0.0007



###### Pour le NDVI == > r² = 0.002
juin_average_NDVI = calculate_average_pixel_values(NDVI_juin,"juin","NDVI")

## Best value
juin_best_NIR = find_best_pixel(NIR_juin, average_NIRS_value['NIR'],"juin","NIR")
NIR_NIRS = (average_NIRS_value.iloc[:,5])

# Calculer le r carré et la pente, etc ...
slope, intercept, r_value, p_value, std_err = linregress(NIR_NIRS, juin_best_NIR)

# Tracer le graphique de dispersion
plt.scatter(NIR_NIRS, juin_best_NIR)
plt.plot(NIR_NIRS,slope*NIR_NIRS+intercept, color='red')

# Ajouter des titres d'axes et un titre de graphique
plt.xlabel('NIRS')
plt.ylabel('Drone')
plt.title('Graphique de dispersion du canal NIR drone en fonction de NIRS - juin')

# Afficher le graphique
plt.show()
print("Le r2 est égal à : ",r_value**2)



###### Plot la map de corrélation de tous les canaux MS avec tous les canaux NIRS pour chaque pixel

correl_map = compute_correlations(juin[2,:,:,:],average_NIRS_value.iloc[2,:])
correl_map_bis = compute_correlations(juin[2,:,:,:],average_NIRS_value.iloc[3,:])

plt.imshow(correl_map,vmin=0,vmax=1)

correl_diff = correl_map - correl_map_bis
plt.imshow(correl_diff,vmin=0,vmax=0.02)

# Vérification sur un pixel bien corrélé
slope, intercept, r_value, p_value, std_err = linregress(juin[0,199,3,:], average_NIRS_value.iloc[0,:])

plt.scatter(juin[0,199,3,:], average_NIRS_value.iloc[0,:])
plt.plot(juin[0,199,3,:],slope*juin[0,199,3,:]+intercept, color='red')

print("Le r2 est égal à : ",r_value**2)