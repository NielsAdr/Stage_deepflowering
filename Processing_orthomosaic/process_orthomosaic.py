from osgeo import gdal
import numpy as np
import pandas as pd


def orthomosaic_to_patch(orthomosaic:gdal.Dataset, metadata:pd.DataFrame, resolution_patch:int):
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
    img = raster_to_img(orthomosaic)
    
    patch_array = np.array([make_one_patch(metadata.loc[i], "pixel_x", "pixel_y", orthomosaic, resolution_patch) for i in range(metadata.shape[0])])
    
    return patch_array


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


def make_one_patch(data:pd.DataFrame, col_x, col_y, length:int):  
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