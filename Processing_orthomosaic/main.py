import os 

from Processing_orthomosaic import constants,process_orthomosaic
from osgeo import gdal
import numpy as np
import pandas as pd


#Loading metadata
metadata = pd.read_csv(constants.PATH_CSV)[["northing", "easting", "id_tree"]]

#For each cameras and each month, compute patch from orthomosaics
for camera in ["ms","rgb"]:
    if camera == "rgb":
        resolution = 300
    if camera == "ms":
        resolution = 200
    for month in ["juin", "septembre", "octobre", "novembre"]:
        end_path =  "/" + camera + "/" + month
        
        directory = constants.PATH_FOLDER_ORTHOMOSAIC + end_path
        
        for file in os.listdir(directory):
            ds = gdal.Open(os.path.join(directory, file))
            
            patch = process_orthomosaic.orthomosaic_to_patch(ds, metadata, resolution)
    
            del ds
            
            os.chdir(constants.PATH_PATCH + end_path)
            
            for tree in range(len(metadata)):
                np.save(f"{metadata.loc[tree].id_tree}.npy", patch)
    
            del patch
