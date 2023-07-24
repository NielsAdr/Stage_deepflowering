import os 

from Processing_orthomosaic import constants,process_orthomosaic
from osgeo import gdal
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

# For lauching patch alone
#ds = gdal.Open("/media/u108-s786/Donnees/Orthomosaiques/ms/juin/210625-ms-all.tif")
#resolution = 200
#file_directory = "/media/u108-s786/Donnees/DEM/rgb/juin/DEM_fitted_ms_juin.tif"

#Loading metadata
metadata = pd.read_csv(constants.PATH_CSV)[["northing", "easting", "id_tree"]]

#For each cameras and each month, compute patch from orthomosaics
for camera in ["rgb"]:
    if camera == "rgb":
        resolution = 300
    if camera == "ms":
        resolution = 200
        for month in ["juin_" + camera, "septembre_"+ camera, "octobre_"+ camera, "novembre_"+ camera]:        
            end_path =  "/" + camera + "/" + month
            
            directory = constants.PATH_FOLDER_ORTHOMOSAIC + end_path
        
        for file in os.listdir(directory):
            file_directory = os.path.join(directory, file)
            
            ds = gdal.Open(file_directory)
            
            patch = process_orthomosaic.orthomosaic_to_patch(ds, metadata, resolution, file_directory)
    
            del ds
            
            os.chdir(constants.PATH_PATCH + end_path)
            
            for tree in range(len(metadata)):
                np.save(f"{metadata.loc[tree].id_tree}.npy", patch[tree,:,:])
    
            del patch

# Now for each patch, take the npy of the photo, of them dem, and apply the filter, then register the new
# image as npy in a new folder

# for camera in ["ms","rgb"]:
#     for month in ["juin_" + camera, "septembre_"+ camera, "octobre_"+ camera, "novembre_"+ camera]:
#         maintenant = datetime.datetime.now()
#         heure = maintenant.strftime("%H:%M:%S")
#         print(heure)
#         saving_path =  constants.PATH_PATCH + "/Images_DEM" + "/" + camera + "/" + month
#         os.chdir(saving_path)
        
#         print("Répertoire de sauvegarde : ",saving_path)
        
#         dem_path =  constants.PATH_PATCH + "/DEM" + "/" + camera + "/" + month 
#         img_path =  constants.PATH_PATCH + "/Images" + "/" + camera + "/" + month 
#         for file in tqdm(os.listdir(img_path)):
#             dem = np.load(dem_path + "/" + file)
#             img = np.load(img_path + "/" + file)
            
#             new_img = process_orthomosaic.filter_dem_on_image(img,dem)
#             np.save(f"{file}", new_img)
            
# As the DEM filter is now already applied, we will make a pipeline to make patches on one filtered_orthomosaic

#Loading metadata
metadata = pd.read_csv(constants.PATH_CSV)[["northing", "easting", "id_tree"]]

# Constants
resolution = 200
month = "juin"
camera = "ms"
file = camera + "_" + month + ".tif"

# Paths
end_path =  "/" + camera + "/" + month
orthomosaic_directory = constants.PATH_FOLDER_ORTHOMOSAIC + '/mask'
ds_directory = constants.PATH_FOLDER_ORTHOMOSAIC + end_path

# Patching
file_directory = os.path.join(orthomosaic_directory, file)
    
# Put the corresponding orthomosaic wich have the good ds informations, we can also put the same name for this file and replace it by the file variable
ds = gdal.Open(ds_directory+'/210625-ms-all.tif')
    
patch = process_orthomosaic.orthomosaic_to_patch(ds, metadata, resolution, file_directory)
    
for tree in range(len(metadata)):
    np.save(f"{metadata.loc[tree].id_tree}.npy", patch[tree,:,:])