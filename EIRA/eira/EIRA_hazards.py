"""
This file is part of EIRA 
Copyright (C) 2024 GFDRR, World Bank
Developers: 
- Jose Antonio Leon Torres  
- Luc Marius Jacques Bonnafous
- Natalia Romero

Your should have received a copy of the GNU Geneal Public License along
with EIRA.
----------

DeFine Hazard classes and functions
"""

#Import libraries
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
import json

# eira dependencies
from eira import EIRA_files_handler
from eira import EIRA_GIS_tools
from eira.EIRA_utils import load_config


#HAZARD_DABASE_FOLDER = '/root/WB/EIRA_tool/EIRA/eira/input_data/DataBases'
HAZARD_DABASE_FOLDER = load_config()['inputs']['path_hazard_database']

HAZARD_TYPE = ['EARTHQUAKE', 'FLOOD', 'WIND', 'DROUGHT'] 




#EARTHQUAKES


def Load_Config():
    """
    This function download the required file for the analysis
    
    """
        

    return


class Class_Generic_Hazard():
    '''
    Generic Hazard Class
    '''

    #List of the hazard models availables
    #HAZARD_MODELS_AVAILABLE: list #list with the name of the models available
    

    # URL database 
    #url_sources: dict #dictionary :Key: name of the model. It have to be the same than in HAZARD_MODELS_AVAILABLES. 2) URL 

    #variable for the path where the hazard file will be saved after download it
    #folder_hazard_save: str 

    def __init__(self):
        '''
        Initialize an Generic Class Instance instance.
        '''
        #print ('Initialize an Generic Hazard Class instance')
        pass



    def print_hazard_models_available(self, Hazard_models_available: list):
        """
        Print the available hazard models in the EQ_HAZARD_MODELS_AVAILABLES list.
        """
        print("Available Hazard Models:")
        for model in Hazard_models_available:
            print(f"- {model}")    

    #@staticmethod
    def Download_1_EQ_hazard_file(self, key: str, url_souce: str, folder_hazard_save_path: str):
        """
        Download the database from online repository.
        """
        downloadfilename= key + '.tif'

        EIRA_files_handler.download_onedrive_file_2(url_souce,downloadfilename,folder_hazard_save_path)


    def extract_hazard_by_mask(self, Selected_EQ_model: str, shp_file_path: str, tif_file_path: str, outputfilepath: str = None):
        '''
        '''
        
        # File name
        tif_file_path= tif_file_path + '.tif'
        #tif_file_path = Class_Generic_Hazard.folder_hazard_save + Selected_EQ_model + '.tif'
        print(tif_file_path)
        # Extra the hazard within affecting the shapefile
        shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_raster_values_within_polygons_large_raster(tif_file_path, shp_file_path)
        #shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_focused_raster(tif_file_path, shp_file_path)

        #EIRA_GIS_tools.plot_extracted_rasters_with_shapefile(extracted_rasters, shapefile_with_values)

        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters(extracted_rasters,tif_file_path,'/root/WB/EIRA_tool/EIRA/eira/output_data/prueba_1.tif')
        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path,outputfilepath)
        
        if outputfilepath != None:
            joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path,outputfilepath)
            
        else:
            joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path)
    
        
        #print joint raster"
        #EIRA_GIS_tools.plot_mosaic_from_output(joined_rasters)
        EIRA_GIS_tools.plot_raster_in_memory(joined_rasters)
        
        print('Ok')

        return joined_rasters    


    def extract_hazard_by_mask_2(self, Selected_EQ_model: str, gdf_shape_file:str, tif_file_path: str, unit_hazard_model:  str = "values", outputfilepath: str = None):
        '''
        
        '''
        
        # File name
        tif_file_path= tif_file_path + '.tif'
        #tif_file_path = Class_Generic_Hazard.folder_hazard_save + Selected_EQ_model + '.tif'
        print(tif_file_path)
        # Extra the hazard within affecting the shapefile
        shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_raster_values_within_polygons_large_raster_3(tif_file_path, gdf_shape_file)
        #shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_focused_raster(tif_file_path, shp_file_path)

        #EIRA_GIS_tools.plot_extracted_rasters_with_shapefile(extracted_rasters, shapefile_with_values)

        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters(extracted_rasters,tif_file_path,'/root/WB/EIRA_tool/EIRA/eira/output_data/prueba_1.tif')
        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path,outputfilepath)
        
        if outputfilepath != None:
            joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path,outputfilepath)
            
        else:
            joined_rasters=EIRA_GIS_tools.join_extracted_rasters_3(extracted_rasters,tif_file_path)
    
        
        #print joint raster"
        #EIRA_GIS_tools.plot_mosaic_from_output(joined_rasters)
        EIRA_GIS_tools.plot_raster_in_memory(joined_rasters,Selected_EQ_model,unit_hazard_model)
        
        print('Ok')

        return joined_rasters    

    def Download_and_Prepare_hazard_model(self,Selected_Hazard_model: str, gdf_shape_referential, url_hazard_file: str,path_folder_to_save: str, 
                                          unit_hazard_model: str):
        """
        Function to download and prepare the hazard file to carry out risk analysis level 1 with EIRA
        
        Parameters
        ----------
        Selected_Hazard_model : (str)
            Name of the hazard model (as it appear in the EIRA database).
        gdf_shape_referential : (GeoDataFrame)
            The geodataframe to used as mask for the extraction of the raster file for the study region.
        url_hazard_file :  (str)
            url of the remote location of the hazard file in the EIRA database in OneDrive.
        path_folder_to_save (str)
            path of the foder in which the hazard file will be storaged/saved.
        unit_hazard_model : (str)
            units in which the intensity of the hazard file is presented.

        Return
        ------
        raster file in memory corresponing at the region limmited by the vector-gis file (shape)

        """

        # Prepare the selected hazard model 
        Class_Generic_Hazard.Download_1_EQ_hazard_file(self,Selected_Hazard_model,url_hazard_file,path_folder_to_save)

        # Extra the hazard within affecting region under study
        tif_file_path = path_folder_to_save + Selected_Hazard_model
        raster_hazard_region = Class_Generic_Hazard.extract_hazard_by_mask_2(self,Selected_Hazard_model,gdf_shape_referential, 
                                                                             tif_file_path,unit_hazard_model)
        
        return raster_hazard_region


class Floods(Class_Generic_Hazard):


    FLOOD_HAZARD_MODELS_AVAILABLES = {
        #"GIRI Flood Hazard 2 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138803&authkey=%21AJgh0W2BDmJpmiM",2,"cm"),
        #"GIRI Flood Hazard 5 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138837&authkey=%21AEa6S5YHKmFjUIo",5,"cm"),
        #"GIRI Flood Hazard 10 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138862&authkey=%21AEUrjSi9mfFkW%2DA",10,"cm"),
        "GIRI Flood Hazard 25 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138900&authkey=%21ADJOU860yHB41D4",25,"cm"),
        #"GIRI Flood Hazard 50 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138921&authkey=%21AHsmz%2DEVz6SS%5F6Q",50,"cm"),
        "GIRI Flood Hazard 100 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138930&authkey=%21AON59xvfKlZFFd0",100,"cm"),
        "GIRI Flood Hazard 200 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138933&authkey=%21AGOm4AbaZZoenXo",200,"cm"),
        #"GIRI Flood Hazard 500 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138940&authkey=%21AB7S2zskLjZ7FE0",500,"cm"),
        "GIRI Flood Hazard 1000 years - Existing Climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138949&authkey=%21AAMePZl358MKNW0",1000,"cm"),
        #"GIRI Flood Hazard 2 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138804&authkey=%21AIQyAa%5F6gUGmCAo",2,"cm"),
        #"GIRI Flood Hazard 5 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138838&authkey=%21AElsFG1Gr1tWAjk",5,"cm"),
        #"GIRI Flood Hazard 10 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138863&authkey=%21AHlw%5F1vyS162X9g",10,"cm"),
        "GIRI Flood Hazard 25 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138901&authkey=%21ACYzEIl8Qh%5F13q4",25,"cm"),
        #"GIRI Flood Hazard 50 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138928&authkey=%21ACja9Y83bTyNWpM",50,"cm"),
        "GIRI Flood Hazard 100 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138931&authkey=%21APzc3NKzWbmwT6k",100,"cm"),
        #"GIRI Flood Hazard 200 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138934&authkey=%21ANYi%2DnWMIzOFu3I",200,"cm"),
        #"GIRI Flood Hazard 500 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138941&authkey=%21AII2Ak7imiLqPvM",500,"cm"),
        "GIRI Flood Hazard 1000 years - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138947&authkey=%21AC0B76rLz0MZDmQ",1000,"cm"),
        #"GIRI Flood Hazard 2 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138811&authkey=%21AFUDHU9E8UMcBZI",2,"cm"),
        #"GIRI Flood Hazard 5 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138843&authkey=%21AA8SYZFhyqXEzyE",5,"cm"),
        #"GIRI Flood Hazard 10 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138864&authkey=%21AC1ahyZb35caZoM",10,"cm"),
        "GIRI Flood Hazard 25 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138902&authkey=%21AP5SPZb%5FYs0AaJo",25,"cm"),
        #"GIRI Flood Hazard 50 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138929&authkey=%21AAFnwps1IAsnrJY",50,"cm"),
        "GIRI Flood Hazard 100 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138932&authkey=%21ABBAe1sIv2WJprY",100,"cm"),
        #"GIRI Flood Hazard 200 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138935&authkey=%21AK0F9NKEF7HTfNE",200,"cm"),
        #"GIRI Flood Hazard 500 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138942&authkey=%21ANl7ZxLtAbh%2D4%2D0",500,"cm"),
        "GIRI Flood Hazard 1000 years - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21138948&authkey=%21AHGZMsEoS8F7kSw",1000,"cm"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_5year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137862&authkey=%21AO4nAnexPFfxlcs',5,"m"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_50year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137863&authkey=%21AFVV2pFLujn90JA',250,"m"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_250year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137868&authkey=%21AANpF67QUGq0YZo',50,"m"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_100year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137869&authkey=%21AN9vTCCvtRNjIvA',100,"m"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_25year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137870&authkey=%21AJn82kJejEWY0R4',25,"m"),
        "Aqueduct_Riverine_Baseline_historical_watch1980_rp_10year": ('https://onedrive.live.com/download?resid=23323221D505E66D%21137871&authkey=%21AHrz8NpAxtHJZ04',10,"m") 
    }


    Floods_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'Floods/'



    def __init__(self):
        '''Initialize a Flood hazard instance'''
        super().__init__() # Call the parent class constructor if needed
        print ('Initialize an Floods class')
        
        self.hazard_type: str = "Floods"





class Droughts(Class_Generic_Hazard):
    '''
    Drought class
    '''    


    DROUGHT_HAZARD_MODELS_AVAILABLES = {
        #"GIRI Drought hazard SMA 5-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139127&authkey=%21AExQdSet3u5z9cw",5,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 5-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139077&authkey=%21AIRfFJC2XZCV4%5F8",5,"Cumulated SD"),
        #"GIRI Drought hazard SSI 5-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139128&authkey=%21ACafTGBJvujiJUc",5,"Cumulated SD"),
        #"GIRI Drought hazard SMA 10-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139211&authkey=%21APodChAwB%2DGDhQw",10,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 10-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139295&authkey=%21APqNx7JAxL10M4E",10,"Cumulated SD"),
        #"GIRI Drought hazard SSI 10-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139253&authkey=%21AJyx9unBwR5fgIU",10,"Cumulated SD"),
        "GIRI Drought hazard SMA 25-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139204&authkey=%21APS7QvPaY8v9E9s",25,"Cumulated SD"),
        "GIRI Drought hazard SPI-6 25-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139288&authkey=%21ANMHE8S2sO2hfyI",25,"Cumulated SD"),
        "GIRI Drought hazard SSI 25-year return period - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139246&authkey=%21AGz7pMtb6l0xVPQ",25,"Cumulated SD"),
        #"GIRI Average duration of a drought event (SMA-1) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139379&authkey=%21AClIOnH5Yi6Okb4",0,"days"),
        "GIRI Average duration of a drought event (SPI-6) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139421&authkey=%21AJMhLvt78sCshRU",0,"days"),
        #"GIRI Average duration of a drought event (SSI-1) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139400&authkey=%21AEx%5Fgk3ZlhU4FlQ",0,"days"),
        #"GIRI Number of drought events in the analysed period (SMA-1) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139316&authkey=%21APgihUpUzpYDEd0",0,"# of events"),
        "GIRI Number of drought events in the analysed period (SPI-6) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139358&authkey=%21AFjd3%2DyGYpqp3ec",0,"# of events"),
        #"GIRI Number of drought events in the analysed period (SSI-1) - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139337&authkey=%21APNTyr1BYwWLhRE",0,"# of events"),
        #"GIRI Drought hazard SMA 5-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139155&authkey=%21AMio5eREWjGz%2DAM",5,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 5-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139102&authkey=%21AN7CIE%5F%5FkQBLu%2Dk",5,"Cumulated SD"),
        #"GIRI Drought hazard SSI 5-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139169&authkey=%21AFsu9klD3J2JCIs",5,"Cumulated SD"),
        #"GIRI Drought hazard SMA 10-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139197&authkey=%21AKkpiQTV39sxYU8",10,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 10-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139281&authkey=%21AL5bEQCuYkXpi88",10,"Cumulated SD"),
        #"GIRI Drought hazard SSI 10-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139239&authkey=%21AO7w6g8722OV8kQ",10,"Cumulated SD"),
        "GIRI Drought hazard SMA 25-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139190&authkey=%21ACeNm%5FyRlV%5FGnfQ",25,"Cumulated SD"),
        "GIRI Drought hazard SPI-6 25-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139274&authkey=%21AI7I7J7bi%5Fpu5VA",25,"Cumulated SD"),
        "GIRI Drought hazard SSI 25-year return period - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139232&authkey=%21AKEnZxW1I3Su35w",25,"Cumulated SD"),
        #"GIRI Average duration of a drought event (SMA-1) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139372&authkey=%21APYKEWwrgkdvSHE",0,"days"),
        "GIRI Average duration of a drought event (SPI-6) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139414&authkey=%21AArc%2DjqB0%5Fo6gOE",0,"days"),
        #"GIRI Average duration of a drought event (SSI-1) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139393&authkey=%21AAW4Drxpo%2DGN%5FYg",0,"days"),
        #"GIRI Number of drought events in the analysed period (SMA-1) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139309&authkey=%21AMWHGcC3%2D80P7MU",0,"# of events"),
        "GIRI Number of drought events in the analysed period (SPI-6) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139344&authkey=%21AAd5PeV9ExM76rQ",0,"# of events"),
        #"GIRI Number of drought events in the analysed period (SSI-1) - SSP1 Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139330&authkey=%21AOAxyJZlgmVOkf8",0,"# of events"),
        #"GIRI Drought hazard SMA 5-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139148&authkey=%21AD8wxQ35fts0cFI",5,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 5-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139141&authkey=%21AJQqKyHkk5Y01TQ",5,"Cumulated SD"),
        #"GIRI Drought hazard SSI 5-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139162&authkey=%21AJjYAlf7LTNNEq8",5,"Cumulated SD"),
        #"GIRI Drought hazard SMA 10-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139183&authkey=%21AByQ1egP%5FK4yla8",10,"Cumulated SD"),
        #"GIRI Drought hazard SPI-6 10-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139267&authkey=%21ACfewc08V2DKKrY",10,"Cumulated SD"),
        #"GIRI Drought hazard SSI 10-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139225&authkey=%21AFixmgKPHAClQnM",10,"Cumulated SD"),
        "GIRI Drought hazard SMA 25-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139176&authkey=%21APzWLEnL5AsW1rw",25,"Cumulated SD"),
        "GIRI Drought hazard SPI-6 25-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139260&authkey=%21AG6DJRzrcG2ic6k",25,"Cumulated SD"),
        "GIRI Drought hazard SSI 25-year return period - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139218&authkey=%21AMyvIj9CGlJFdJI",25,"Cumulated SD"),
        #"GIRI Average duration of a drought event (SMA-1) -  SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139365&authkey=%21ACZJWblvk5IqpZo",0,"days"),
        "GIRI Average duration of a drought event (SPI-6) -  SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139407&authkey=%21ALLCafWXSxuVqY4",0,"days"),
        #"GIRI Average duration of a drought event (SSI-1) -  SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139386&authkey=%21ADlfxUqXnk2VyrQ",0,"days"),
        #"GIRI Number of drought events in the analysed period (SMA-1) - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139302&authkey=%21AO%5FqcjkdVcapWKk",0,"# of events"),
        "GIRI Number of drought events in the analysed period (SPI-6) - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139351&authkey=%21AAXORlxObymKCKg",0,"# of events")
        #"GIRI Number of drought events in the analysed period (SSI-1) - SSP5 Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139323&authkey=%21ACixYQlAtpmrah8",0,"# of events")
    }


    Drought_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'Drought/'


    def __init__(self):

        super().__init__() # Call the parent class constructor if needed
        print('Initialized drought class')

        self.hazard_type: str = "Drought"
    pass




class Landslide(Class_Generic_Hazard):
    
    LANDSLIDE_HAZARD_MODELS_AVAILABLES = {
        "ARUP_WB_UNDRR_Global_landslide_hazard_Trigger_Earthquake_Precipitation_comb_2020" : ('https://onedrive.live.com/download?resid=23323221D505E66D%21138045&authkey=%21AIb0XYi5%5F0Ug49Y',0,"SD"),
        "GIRI Susceptibility Class of Landslides Triggered By Earthquakes": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139454&authkey=%21AJTNiFJLuuAYsVI",0,"Quantitative scale. 1=not susceptible; 5=extremely susceptible"),
        "GIRI Susceptibility Class of Landslides Triggered By Precipitation - Existing climate": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139453&authkey=%21AAur6HkrSiBNuTI",0,"Quantitative scale. 1=not susceptible; 5=extremely susceptible"),
        "GIRI Susceptibility Class of Landslides Triggered By Precipitation - Lower bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139456&authkey=%21ACS3kvXmR%2DTRhsI",0,"Quantitative scale. 1=not susceptible; 5=extremely susceptible"),
        "GIRI Susceptibility Class of Landslides Triggered By Precipitation - Upper bound": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139455&authkey=%21AEUu12Qf6v9CRfM",0,"Quantitative scale. 1=not susceptible; 5=extremely susceptible")
    }
    
   
    Landslide_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'Landslide/'


    def __init__(self):

        super().__init__() # Call the parent class constructor if needed
        print('Initialized landslide class')

        self.hazard_type: str = "Landslide"
    
    pass



class Temperature(Class_Generic_Hazard):
    
    TEMPERATURE_HAZARD_MODELS_AVAILABLES = {   
        "NEO_NASA_MOD_LSTD_November_2024" :('https://onedrive.live.com/download?resid=23323221D505E66D%21138064&authkey=%21AJkeRoeWqoK49ls',0,"Celsius degrees"),
    }

    Temperature_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'Temperature/'


    def __init__(self):

        super().__init__() # Call the parent class constructor if needed
        print('Initialized Temperature class')

        self.hazard_type: str = "Temperature"
    pass




class Earthquakes(Class_Generic_Hazard):


    EQ_HAZARD_MODELS_AVAILABLES = {
        "GEM_Peak Gound Acceleration PGA - 475_years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21132382&authkey=%21AIpgYULt2%2DQ%2DoD4',475,"g"),
        "GIRI Peak Ground Acceleration PGA - 250 Years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21138730&authkey=%21ADGFMCDj36olue4',250,"cm/s2"),
        "GIRI Peak Ground Acceleration PGA - 475 Years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21138722&authkey=%21ANGi6p0cwynlcao',475,"cm/s2"),
        "GIRI Peak Ground Acceleration PGA - 975 Years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21138782&authkey=%21AO0BXGyWwhjA4q0',975,"cm/s2"),
        "GIRI Peak Ground Acceleration PGA - 1500 Years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21138789&authkey=%21ADcGxKddZ7ODaaM',1500,"cm/s2"),
        "GIRI Peak Ground Acceleration PGA - 2475 Years": ('https://onedrive.live.com/download?resid=23323221D505E66D%21138790&authkey=%21ALA8dRAf9gfqAAo',2475,"cm/s2"),
        "GAR Peak Ground Acceleration PGA - 475 Years":('https://onedrive.live.com/download?resid=23323221D505E66D!132374&authkey=!AGqBUi8TXEnD7lQ',475,"cm/s2")
 
    }


    
    EQ_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'EQ/'



    def __init__(self):
        '''Initialize a Flood hazard instance'''
        super().__init__() # Call the parent class constructor if needed
        print ('Initialize an Earthquake class')
        
        self.hazard_type: str = "Earthquakes"

class TropicalCyclones_Wind(Class_Generic_Hazard):
    
    TROPICAL_CYCLONES_WIND_MODELS_AVAILABLES = {
        "GIRI Tropical Cyclone Wind CC - 25 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139507&authkey=%21AFRnIR%2DjVWPbTfc",25,"Km/h"),
        "GIRI Tropical Cyclone Wind CC - 50 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139500&authkey=%21ALcp9Cbeae3cZo0",50,"Km/h"),
        "GIRI Tropical Cyclone Wind CC - 100 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139493&authkey=%21AMJr6yjVXiow%5FbI",100,"Km/h"),
        "GIRI Tropical Cyclone Wind CC - 250 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139514&authkey=%21AOg4jX2LiQrbAoY",250,"Km/h"),
        "GIRI Tropical Cyclone Wind - 25 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139486&authkey=%21AHRbkQYHCj0D%2Dzk",25,"Km/h"),
        "GIRI Tropical Cyclone Wind - 50 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139465&authkey=%21AIIdsZjmFUFD4vQ",50,"Km/h"),
        "GIRI Tropical Cyclone Wind - 100 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139472&authkey=%21AN3hHLzVi0%5FIdvo",100,"Km/h"),
        "GIRI Tropical Cyclone Wind - 250 years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139477&authkey=%21AOOZ3MxTBJ8xSdE",250,"Km/h")
    }
    
    TC_wind_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'TC_Wind/'



    def __init__(self):
        '''Initialize a Tropycal Cyclone Wind hazard instance'''
        super().__init__() # Call the parent class constructor if needed
        print ('Initialize an TC Wind class')
        
        self.hazard_type: str = "Tropycal_Cyclone_Wind"
    pass




class Tsunami(Class_Generic_Hazard):
    
    TSUNAMI_MODELS_AVAILABLES = {
        "GIRI Tsunami run-up on costline - 475 Years": ("https://onedrive.live.com/download?resid=23323221D505E66D%21139527&authkey=%21AIQtVtL6a9dXnwQ",475,"m"),
    }
    
    Tsunami_folder_hazard_to_save = HAZARD_DABASE_FOLDER + '/' + 'Tsunami/'



    def __init__(self):
        '''Initialize a Tsunami hazard instance'''
        super().__init__() # Call the parent class constructor if needed
        print ('Initialize an Tsunami class')
        
        self.hazard_type: str = "Tsunami"
    pass








class Earthquake_previous():
    """
    Class Earhquake 
    
    """
    EQ_HAZARD_MODELS_AVAILABLES = [
        'GEM_475_years',
        'GIRI_475_years',
        'GAR_475_years',
        'ALL_models'
    ]
    
    # Earhquake URL database 
    EQ_url_sources = {
        "GEM_475_years": 'https://onedrive.live.com/download?resid=23323221D505E66D%21132382&authkey=%21AIpgYULt2%2DQ%2DoD4',
        "GIRI_475_years": 'https://onedrive.live.com/download?resid=23323221D505E66D%21132384&authkey=%21AFwwbtmaQyy6z10',
        "GAR_475_years":'https://onedrive.live.com/download?resid=23323221D505E66D!132374&authkey=!AGqBUi8TXEnD7lQ'
    }

    EQ_folder_hazard = HAZARD_DABASE_FOLDER + '/' + 'EQ/'

    def __init__(self):
        '''
        Initialize an Earthquakes instance.
        '''
        #print ('Initialize an Earthquakes instance')
        pass



    def print_hazard_models_available(self):
        """
        Print the available hazard models in the EQ_HAZARD_MODELS_AVAILABLES list.
        """
        print("Available Hazard Models:")
        for model in Earthquake_previous.EQ_HAZARD_MODELS_AVAILABLES:
            print(f"- {model}")    

    @staticmethod
    def Download_1_EQ_hazard_file(key: str):
        """
        Download the database from online repository.
        """
        downloadfilename= key + '.tif'

        EIRA_files_handler.download_onedrive_file_2(Earthquake_previous.EQ_url_sources[key],downloadfilename,Earthquake_previous.EQ_folder_hazard)


    def extract_hazard_by_mask(self, Selected_EQ_model: str, shp_file_path: str):
        '''
        '''
        
        # File name
        tif_file_path = Earthquake_previous.EQ_folder_hazard + Selected_EQ_model + '.tif'
        print(tif_file_path)
        # Extra the hazard within affecting the shapefile
        shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_raster_values_within_polygons_large_raster(tif_file_path, shp_file_path)
        #shapefile_with_values, extracted_rasters = EIRA_GIS_tools.extract_focused_raster(tif_file_path, shp_file_path)


        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters(extracted_rasters,tif_file_path,'/root/WB/EIRA_tool/EIRA/eira/output_data/prueba_1.tif')
        #joined_rasters=EIRA_GIS_tools.join_extracted_rasters(extracted_rasters,tif_file_path)
        joined_rasters = EIRA_GIS_tools.join_extracted_rasters_2(extracted_rasters,tif_file_path)
        #EIRA_GIS_tools.plot_extracted_rasters(joined_rasters, shapefile_with_values)
        #print joint raster"
        #EIRA_GIS_tools.plot_mosaic_from_output(joined_rasters)
        EIRA_GIS_tools.plot_raster_in_memory(joined_rasters)
        #EIRA_GIS_tools.plot_in_memory_file(joined_rasters)

        #EIRA_GIS_tools.save_extracted_rasters(extracted_rasters, shapefile_with_values)
        print('Ok')

        return joined_rasters