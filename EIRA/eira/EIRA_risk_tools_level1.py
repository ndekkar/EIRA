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
import os
import geopandas as gpd
from rasterio.features import shapes
import shapely.geometry as geom

# eira dependencies
from eira import EIRA_files_handler
from eira import EIRA_GIS_tools


class Class_ExpertCriteria_Hazard_classifcation:
    '''
    Class Class_ExpertCriteria_Hazard_classifcation
    '''
    #def __init__(self, hazard_type: str, classification_ranges:  dict, hazard_intesity: str):  #this is when we initialize the class with required arguments
    def __init__(self):
        """
        Initialize to Class_ExpertCriteria_Hazard_classifcation.

        Args:
            hazard_type (str): The type of hazard (e.g., earthquake, drought).
            classification_ranges (dict): The ranges for classification.
            intensity_used (str): The intensity measure used.
        """        
        self.hazard_type: str  =""
        self.hazard_intensity_1: str = ""
        self.classification_range_intensity_1: dict = None
        self.hazard_intensity_2: str = ""
        self.classification_range_intensity_2: dict = None
        self.hazard_intensity_3: str = ""
        self.classification_range_intensity_3: dict = None
        

    

HAZARD_DABASE_FODER = '/root/WB/EIRA_tool/EIRA/eira/input_data/DataBases'

# Hazard Classification Criteria Ranges
HAZARD_EXP_CRITERIA_RANGE_DB = {
    'Earthquake_classification_ranges': {
        'low hazard': (-1, 50), 
        'medium hazard': (50, 300), 
        'high hazard': (300, 2000)
    }, #Spectral Aceleration (cm/s2)
    'Drought_classification_range' : {
        'low hazard': (-1, 5), 
        'medium hazard': (5, 50), 
        'high hazard': (50, 200)
    }  #Cumulated standard deviation
}

Drought_exp_criteria_hazard = Class_ExpertCriteria_Hazard_classifcation()
Earthquake_exp_criteria_hazard = Class_ExpertCriteria_Hazard_classifcation()
Flood_exp_criteria_hazard = Class_ExpertCriteria_Hazard_classifcation()
Flood_exp_criteria_hazard_GIRI = Class_ExpertCriteria_Hazard_classifcation()

def Fill_hazard_classification_expert_criteria():

    global Drought_exp_criteria_hazard, Earthquake_exp_criteria_hazard, Flood_exp_criteria_hazard, Flood_exp_criteria_hazard_GIRI

    #Drought_exp_criteria_hazard = Class_ExpertCriteria_Hazard_classifcation()
    Drought_exp_criteria_hazard.hazard_type = 'Drought'
    Drought_exp_criteria_hazard.hazard_intensity_1 = "Cumulated Standard Deviation"
    Drought_exp_criteria_hazard.classification_range_intensity_1= {
        'low hazard': (-1, 5), 
        'medium hazard': (5, 50), 
        'high hazard': (50, 200)
    }  #Cumulated standard deviation
    Drought_exp_criteria_hazard.hazard_intensity_2 = "days"
    Drought_exp_criteria_hazard.classification_range_intensity_2 = {
        'low hazard': (-1, 5), 
        'medium hazard': (5, 50), 
        'high hazard': (50, 200)
    }
    Drought_exp_criteria_hazard.hazard_intensity_3 = "# of events"
    Drought_exp_criteria_hazard.classification_range_intensity_3 =  {
         'low hazard': (-1, 5), 
        'medium hazard': (5, 50), 
        'high hazard': (50, 200)       
    }

    #Earthquake_exp_criteria_hazard = Class_ExpertCriteria_Hazard_classifcation()
    Earthquake_exp_criteria_hazard.hazard_type = 'Earthquake'
    Earthquake_exp_criteria_hazard.hazard_intensity_1= "cm/s2"
    Earthquake_exp_criteria_hazard.classification_range_intensity_1= {
        'low hazard': (0.1, 50), 
        'medium hazard': (50, 300), 
        'high hazard': (300, 2000)
    } #Spectral Aceleration (cm/s2)

    #Floods
    Flood_exp_criteria_hazard.hazard_type = "Flood"
    Flood_exp_criteria_hazard.hazard_intensity_1 = "meters"
    Flood_exp_criteria_hazard.classification_range_intensity_1 = {
        'low hazard': (0, 0.5), 
        'medium hazard': (0.5, 1), 
        'high hazard': (1, 50)
    }

    #Floods -GIRI
    Flood_exp_criteria_hazard_GIRI.hazard_type = "Flood"
    Flood_exp_criteria_hazard_GIRI.hazard_intensity_1 = "cm"
    Flood_exp_criteria_hazard_GIRI.classification_range_intensity_1 = {
        'low hazard': (-100, 50), 
        'medium hazard': (50, 100), 
        'high hazard': (100, 5000)
    }


def perfom_risk_analysis_level1_TypePoint(raster_hazard_region, gdf_exposure_shape, pic_title_name = 'Hazard', pic_x_axis_name='hazard intensity unit'):
    '''
    
    Paratemers:
    raster_for_plotting: This file is not required for any computation. It just can be used for certain plot functions that require a raster file in memory. 
    '''

    # Prepare files to the computation
    #gdf_polygon_raster_hazard_region = prepare_raster_hazard_file(raster_hazard_region)
    #gdf_raster_earthquake_region = EIRA_GIS_tools.raster_to_gdf_type_polygons(raster_hazard_region)
    #print(gdf_raster_earthquake_region.head())
    #EIRA_GIS_tools.plot_vectorGIS_in_memory(gdf_raster_earthquake_region)

    #gdf_risk_results = EIRA_GIS_tools.intersect_and_combine_attributes(gdf_polygon_raster_hazard_region,gdf_exposure_shape)
    gdf_risk_results = EIRA_GIS_tools.extract_raster_values_to_gdf(raster_hazard_region,gdf_exposure_shape)
    

    EIRA_GIS_tools.plot_raster_in_memory_with_gdf_basemap(raster_hazard_region,gdf_risk_results,
                                                          title="Risk results for the exposed infraestructure",
                                                          adjust_window_plot_to="gdf",use_basemap=False)
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(gdf_risk_results, "Risk results for the exposed infraestructure",use_basemap=True)
    #print(gdf_risk_results_earthquakes.info())
    #print(gdf_risk_results_earthquakes.head())
    #EIRA_GIS_tools.plot_vectorGIS_in_memory(gdf_risk_results_earthquakes)

    #Perform risk analysis
    frequency_table = EIRA_GIS_tools.analyze_geodataframe_risk_point_type(gdf_risk_results,plot_title_name=pic_title_name,plot_x_axis_name=pic_x_axis_name)  # Replace 'column_name' with the actual column name

    return frequency_table

def perfom_risk_analysis_level1_TypePoint_method2(raster_hazard_region, gdf_exposure_shape, pic_title_name = 'Hazard', pic_x_axis_name='hazard intensity unit',
                                           raster_for_plotting=None):
    '''
    
    Paratemers:
    raster_for_plotting: This file is not required for any computation. It just can be used for certain plot functions that require a raster file in memory. 
    '''

    # Prepare files to the computation
    gdf_polygon_raster_hazard_region = prepare_raster_hazard_file(raster_hazard_region)

    gdf_risk_results = EIRA_GIS_tools.intersect_and_combine_attributes(gdf_polygon_raster_hazard_region,gdf_exposure_shape)
    

    EIRA_GIS_tools.plot_raster_in_memory_with_gdf_basemap(raster_for_plotting,gdf_risk_results,
                                                          title="Risk results for the exposed infraestructure",
                                                          adjust_window_plot_to="gdf",use_basemap=False)
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(gdf_risk_results, "Risk results for the exposed infraestructure",use_basemap=True)
    #print(gdf_risk_results_earthquakes.info())
    #print(gdf_risk_results_earthquakes.head())
    #EIRA_GIS_tools.plot_vectorGIS_in_memory(gdf_risk_results_earthquakes)

    #Perform risk analysis
    frequency_table = EIRA_GIS_tools.analyze_geodataframe_risk_point_type(gdf_risk_results,plot_title_name=pic_title_name,plot_x_axis_name=pic_x_axis_name)  # Replace 'column_name' with the actual column name

    return frequency_table



def Use_expert_criteria_for_qualitative_classify_hazard_levels(frequency_table_df, classification_ranges: dict, range_column:  str='pixel_value Range'):
    """
    Classifies ranges in the frequency_table_df into qualitative categories and summarizes the data.

    :param frequency_table_df: DataFrame with frequency table containing a 'Frequency' column and range data.
    :param classification_ranges: Dictionary defining quantitative to qualitative mappings, e.g.,
                                   {'low hazard': (0.1, 5), 'medium hazard': (5, 10), 'high hazard': (10, 20)}.
    :param range_column: The name of the column containing the range data as pandas Interval values.
    :return: A summary DataFrame with counts of elements in each qualitative category.
    """
    # Validate inputs
    if 'Frequency' not in frequency_table_df.columns or len(frequency_table_df) == 0:
        raise ValueError("Input DataFrame must contain a 'Frequency' column and cannot be empty.")
    if range_column not in frequency_table_df.columns:
        raise ValueError(f"Range column '{range_column}' not found in DataFrame.")
    
    # Add a qualitative classification column
    def classify_range(row):
        range_start = row[range_column].left
        range_end = row[range_column].right
        for qualitative, (start, end) in classification_ranges.items():
            if start <= range_start and range_end <= end:
                return qualitative
        return 'Undefined'
    
    frequency_table_df['Hazard Classification'] = frequency_table_df.apply(classify_range, axis=1)
    
    # Summarize the counts by qualitative classification
    summary = frequency_table_df.groupby('Hazard Classification')['Frequency'].sum().reset_index()
    summary.columns = ['Hazard Level', 'Total Frequency']

    print(summary)

    return summary



def perfom_EIRA_risk_analysis_level1(geodataframes_dict,raster_hazard_region, Hazard_name:str="Hazard", 
                                     Hazard_intensity_name:str="Hazard_intensity_unit"):
    """
    Analyze all GeoDataFrames in the dictionary based on their geometry type.

    Args:
        geodataframes_dict (dict): Dictionary of GeoDataFrames with keys in the format "<filename>_Type-<geometry_type>".

    Returns:
        dict: A dictionary with the same keys and the results of the analysis.
    """
    analysis_results = {}

    # Prepare the raster file to the computation
    print("Starting EIRA Risk analysis level 1.................................................")


    for key, gdf_exposure in geodataframes_dict.items():
        # Extract the geometry type from the key
        geometry_type = key.split("Type-")[-1]
        
        print(f"EIRA results, risk analysis level 1 for the exposure : {key} ")
        print("Progressing risk results......................................")
        # Perform analysis based on the geometry type
        if geometry_type == "point":
            result = perfom_risk_analysis_level1_TypePoint(raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name)
        elif geometry_type == "linestring":
            
            result = perfom_risk_analysis_level1_TypeLines(raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name)
        elif geometry_type == "polygon":
            
            result = perfom_risk_analysis_level1_TypePoint(raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name)
        else:
            print(f"Unsupported geometry type: {geometry_type} for key {key}")
            result = None

        # Store the result in the analysis_results dictionary (results are by now frequency tables)
        analysis_results[key] = result

        print ("---------------------------------------------------------------------------")
    return analysis_results


def perfom_EIRA_risk_analysis_level1_method2(geodataframes_dict,raster_hazard_region, Hazard_name:str="Hazard", 
                                     Hazard_intensity_name:str="Hazard_intensity_unit"):
    """
    Analyze all GeoDataFrames in the dictionary based on their geometry type.

    Args:
        geodataframes_dict (dict): Dictionary of GeoDataFrames with keys in the format "<filename>_Type-<geometry_type>".

    Returns:
        dict: A dictionary with the same keys and the results of the analysis.
    """
    analysis_results = {}

    # Prepare the raster file to the computation
    print("Preparing hazard file data for the analysis.................................................")
    gdf_polygon_raster_hazard_region = EIRA_GIS_tools.raster_to_gdf_type_polygons(raster_hazard_region)


    for key, gdf_exposure in geodataframes_dict.items():
        # Extract the geometry type from the key
        geometry_type = key.split("Type-")[-1]
        
        print(f"EIRA results, risk analysis level 1 for the exposure : {key} ")
        print("Progressing risk results......................................")
        # Perform analysis based on the geometry type
        if geometry_type == "point":
            #raster_drought_region,gdf_shape_selected, 'Drought', hazard_int_used
            result = perfom_risk_analysis_level1_TypePoint(gdf_polygon_raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name,raster_hazard_region)
        
        elif geometry_type == "linestring":
            result = perfom_risk_analysis_level1_TypeLines(gdf_polygon_raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name,raster_hazard_region)
            
        elif geometry_type == "polygon":
            result = perfom_risk_analysis_level1_TypePoint(gdf_polygon_raster_hazard_region,gdf_exposure,Hazard_name,Hazard_intensity_name,raster_hazard_region)
            
        else:
            print(f"Unsupported geometry type: {geometry_type} for key {key}")
            result = None

        # Store the result in the analysis_results dictionary (results are by now frequency tables)
        analysis_results[key] = result

        print ("---------------------------------------------------------------------------")
    return analysis_results





def perfom_risk_analysis_level1_TypeLines(raster_hazard_region, gdf_exposure_shape, pic_title_name = 'Hazard', pic_x_axis_name='hazard intensity unit'):
    '''
    Paratemers:
    raster_for_plotting: This file is not required for any computation. It just can be used for certain plot functions that require a raster file in memory.    
    '''

    #Rutine to trim or split a shape file (line geometry) in segment with a max length.  
    new_gdf_exposure=EIRA_GIS_tools.segment_lines(gdf_exposure_shape,0.5)
    # Plot the new splitted file
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(new_gdf_exposure, "Exposured File",use_basemap=True)
    #print(new_gdf_exposure['Length_km'])

    # Prepare rastar file to the computation   
    #gdf_polygon_raster_hazard_region = prepare_raster_hazard_file(raster_hazard_region)
    #raster_region_to_gdf = EIRA_GIS_tools.raster_to_gdf_type_polygons(raster_hazard_region)

    #gdf_risk_results = EIRA_GIS_tools.intersect_and_combine_attributes(gdf_polygon_raster_hazard_region,new_gdf_exposure)
    gdf_risk_results =  EIRA_GIS_tools.extract_raster_values_to_gdf(raster_hazard_region,new_gdf_exposure)

    EIRA_GIS_tools.plot_raster_in_memory_with_gdf_basemap(raster_hazard_region,gdf_risk_results,
                                                          title="Risk results for the exposed infraestructure",
                                                          adjust_window_plot_to="gdf",use_basemap=False)
    
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(gdf_risk_results, "Risk results for the exposed infraestructure",use_basemap=True)
    #print(gdf_risk_results) for Debbuging

    #Generate the histogram
    frequency_table = EIRA_GIS_tools.analyze_geodataframe_risk_line_type(gdf_risk_results,bins=3,plot_title_name=pic_title_name,plot_x_axis_name=pic_x_axis_name)

    print (frequency_table)


   
    return frequency_table


def perfom_risk_analysis_level1_TypeLines_method2(raster_hazard_region, gdf_exposure_shape, pic_title_name = 'Hazard', pic_x_axis_name='hazard intensity unit', 
                                          raster_for_plotting=None):
    '''
    Paratemers:
    raster_for_plotting: This file is not required for any computation. It just can be used for certain plot functions that require a raster file in memory.    
    '''

    #Rutine to trim or split a shape file (line geometry) in segment with a max length.  
    new_gdf_exposure=EIRA_GIS_tools.segment_lines(gdf_exposure_shape,0.5)
    # Plot the new splitted file
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(new_gdf_exposure, "Exposured File",use_basemap=True)
    #print(new_gdf_exposure['Length_km'])

    # Prepare rastar file to the computation   
    gdf_polygon_raster_hazard_region = prepare_raster_hazard_file(raster_hazard_region)

    gdf_risk_results = EIRA_GIS_tools.intersect_and_combine_attributes(gdf_polygon_raster_hazard_region,new_gdf_exposure)


    EIRA_GIS_tools.plot_raster_in_memory_with_gdf_basemap(raster_for_plotting,gdf_risk_results,
                                                          title="Risk results for the exposed infraestructure",
                                                          adjust_window_plot_to="gdf",use_basemap=False)
    
    #EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(gdf_risk_results, "Risk results for the exposed infraestructure",use_basemap=True)
    #print(gdf_risk_results) for Debbuging

    #Generate the histogram
    frequency_table = EIRA_GIS_tools.analyze_geodataframe_risk_line_type(gdf_risk_results,bins=3,plot_title_name=pic_title_name,plot_x_axis_name=pic_x_axis_name)

    print (frequency_table)


   
    return frequency_table




def prepare_raster_hazard_file(hazard_file_in_memory):
    '''
    This function review if the hazard file is in the required format for risk analysis level 1 in EIRA (the required format is a geodataframe of polygons) and if not
    it makes the required coversions.
    The hazard file in memory could be a raster or a geodataframe of polygons
    '''
    if isinstance(hazard_file_in_memory, gpd.GeoDataFrame):
        print("The hazard_file is already a GeoDataFrame. No action needed.")
        return hazard_file_in_memory
    else:
        #I assume that the file is a raster file
        try: 
            gdf_hazard = EIRA_GIS_tools.raster_to_gdf_type_polygons(hazard_file_in_memory)
            return gdf_hazard
            print("The hazard_file is a raster file. Converting to GeoDataFrame...")
        except:
            print("The hazard_file is neither a valid vector nor a raster file.")
            return None
    