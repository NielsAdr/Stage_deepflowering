from osgeo import gdal, osr, ogr
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import generic_filter



def orthomosaic_to_patch(orthomosaic:gdal.Dataset, metadata:pd.DataFrame, resolution_patch:int, file_directory):
    """
    Create patch from orthomosaic

    Parameters
    ----------
    orthomosaic : gdal.Dataset
        a gdal dataset which contains orthomosaic
    
    metadata : pd.DataFrame
        a dataset which easting and northing coordinates
        
    resolution_patch : int
        dimension (x and y) of patch. 
        
    Returns
    -------
    numpy array of length : (number_patch, resolution_patch, resolution_patch, number_channel)
    """
    
    metadata = Lambert_to_npy_coordinates(orthomosaic, metadata)
    img = tiff.imread(file_directory)
    
    patch_array = np.array([make_one_patch(metadata.loc[i], "pixel_x", "pixel_y", img, resolution_patch) for i in range(metadata.shape[0])])
    
    return patch_array




def convert_lambert_to_wgs84(df):
    # Définir le système de coordonnées de départ : Lambert 93
    source = osr.SpatialReference()
    source.ImportFromEPSG(2154)

    # Définir le système de coordonnées d'arrivée : WGS84
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    # Créer un transformateur de coordonnées
    transform = osr.CoordinateTransformation(source, target)

    # Convertir les coordonnées de chaque ligne du dataframe
    for index, row in df.iterrows():
        easting = row["easting"]
        northing = row["northing"]
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(easting, northing)
        point.Transform(transform)
        df.at[index, "latitude"] = point.GetY()
        df.at[index, "longitude"] = point.GetX()

    return df



def Lambert_to_npy_coordinates(orthomosaic:gdal.Dataset, metadata:pd.DataFrame):
    """
    Transform lambert-93 coordinates to numpy coordinates.

    Parameters
    ----------
    orthomosaic : osgeo.gdal.Dataset
        a gdal dataset which contains orthomosaic
    
    metadata : pd.DataFrame
        a dataset which easting and northing coordinates

    Returns
    -------
    Add x and y numpy coordinates to metadata dataframe.
    """
    
    geot = orthomosaic.GetGeoTransform()
    number_tree = metadata.shape[0]
    
    #Si besoin de convertir en wgs84
    #metadata = convert_lambert_to_wgs84(metadata)
    
    metadata = metadata.assign(pixel_x=[(metadata["easting"].loc[i] - geot[0])/geot[1] for i in range(number_tree)],
                              pixel_y=[(metadata["northing"].loc[i] - geot[3])/geot[5] for i in range(number_tree)])
     
    return metadata   




def raster_to_img(orthomosaic:gdal.Dataset):  
    """
    transform an orthomosaic to numpy array.

    Parameters
    ----------
    orthomosaic : osgeo.gdal.Dataset
        a gdal dataset which contains orthomosaic

    Returns
    -------
    orthomosaic in numpy array of dimension (resolution x,resolution y, channels).
    """
    
    raster = [orthomosaic.GetRasterBand(i+1).ReadAsArray() for i in range(orthomosaic.RasterCount)]
    return np.dstack(raster)



def make_one_patch(data:pd.DataFrame, col_x, col_y,image , length:int):  
    """
    Create a square patch from trees center coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        coordinates dataframe.
        
    col_x : 
        x numpy coordinates of tree center.
        
    col_y : int
        y numpy coordinates of tree center.
        
    length : int
        length of patch.


    Returns
    -------
    np.array
        return a patch of trees in dimension defined.
    """
    
    l = int(length/2)
    X, y = int(data[col_x]), int(data[col_y])
    
    return image[X-l:X+l, y-l:y+l]




def variance_sliding_window(image, window_size):
    def variance(data):
        return data.var()

    result = generic_filter(image, variance, size=window_size)
    return result



def filter_dem_on_image(image, dem):
    """
    Applique le filtre DEM sur chaque canal (couche) d'une image et met à 0 les valeurs
    en dehors du profil DEM.
    
    Args:
        image (numpy.ndarray): Image multispectrale ou RGB en format numpy de dimensions (200, 200, 3) ou (200, 200, 6).
        dem (numpy.ndarray): Profil DEM en format numpy de dimensions (200, 200).
    
    Returns:
        numpy.ndarray : Une nouvelle image numpy de même dimension que `image`, où les valeurs
        en dehors du profil DEM sont mises à 0 pour chaque canal.
    """
    
    # Créer une copie de l'image pour éviter de modifier l'originale
    filtered_image = image.copy()
    filtered_dem = dem.copy()
    
    var = variance_sliding_window(dem, (11,11))
    
    # Mettre toutes les valeurs du sol à 0 dans le profil DEM
    filtered_dem[var < 0.0001] = 0
    
    # Créer un masque pour retirer les valeurs du sol de l'image
    mask = np.zeros_like(filtered_image[:, :, 0])
    mask[filtered_dem > 0] = 1
    
    # Remplacer le canal de l'image avec le canal filtré
    filtered_image = np.multiply(filtered_image,mask[...,np.newaxis])
        
    return filtered_image