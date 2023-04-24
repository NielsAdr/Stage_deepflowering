import os 

from Processing_orthomosaic import constants,process_orthomosaic
from osgeo import gdal
import numpy as np
import pandas as pd


#Loading metadata
metadata = pd.read_csv(constants.PATH_CSV)[["northing", "easting", "id_tree"]]

#For each cameras and each month, compute patch from orthomosaics
for camera in ["DEM"]:
    for month in ["juin", "septembre", "octobre", "novembre"]:
        
        end_path =  "/" + camera + "/" + month
        
        ds = gdal.Open(constants.PATH_FOLDER_ORTHOMOSAIC + end_path)
        patch = orthomosaic_to_patch(metadata, ds)

        del ds
        
        os.chdir(constants.PATH_PATCH + end_path)
        
        for tree in range(len(metadata)):
            np.save(f"{data.loc[tree].id_tree}", patch[][])

        del patch
