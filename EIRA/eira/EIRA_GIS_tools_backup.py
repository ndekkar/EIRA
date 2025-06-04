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

define GIS tools

This book defines classes to handle different GIS file types and its correspondining geo-spatial operations with them 
"""

#Libraries
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import json 
import pandas as pd
import geopandas as gpd
import pyproj
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.io import MemoryFile

from shapely.geometry import mapping, box

from shapely.geometry import Point
from shapely.geometry import box
from rasterio.features import geometry_window
from rasterio.merge import merge
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS

from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

import time
import gc
import random
from shapely.validation import make_valid
import contextily as ctx
import xyzservices

import cartopy.crs as ccrs
import cartopy.feature as cfeature



#Functions

def load_Config():
    '''
    Read .json file
    '''

    #Code form Elko
    
    config_path=os.path.join(os.path.dirname(__file__),'..','EIRA_Config.json')
    with open(config_path,"r") as config_file:
        config= json.load(config_file)
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    

    #Code from Ingrid
    # 1. read the configuration (.json) file which contains things that may change: 
    # file names, parameters, etc.
    #jsonName = "EIRA_v1_Config"
    #config = json.load(open(jsonName, encoding='utf-8'))



    # 2.1 read the information from the config file...
    # ...for input
    #inputhEQ = config['inputs']['inputpathEQ']
    #raster_file_path=config['inputs']['filenames']['Hazardfilename']
    #number_of_Points_ERC=config['inputs']['paramvalues_Earthquake']['number_of_Points_ERC']

    return config

def read_raster_file(raster_file_path):
    """
    Function to read a raster a raster file from a path
    
    Parameters
    ----------
    raster_file_path: str
        name of the file

    Returns
    -------
    raster: raster object
        return a raster file
    
    """

    #raster = rasterio.open(raster_file_path)   #Other way to read the raster

    # Open the raster file
    raster = rasterio.open(raster_file_path)
    # Read metadata
    metadata = raster.meta.copy()
    print("Metadata:", metadata)
        
    # Read the first band
    band1 = raster.read(1)
    print("First band data:", band1)
        
    # Read the shape of the raster
    print("Shape:", band1.shape)

    # Get raster bounds
    bounds = raster.bounds
    print("Bounds:", bounds)

    # Get coordinate reference system
    crs = raster.crs
    print("CRS:", crs)

    return raster

def read_vectorGIS_file (filepath : str):
    """
    Function to reads a vector GIS file (geopackage, shape, etc.,) file and returns a GeoDataFrame.

    Parameters
    ---------- 
    filepath: (str)
        file path.

    Returns
    -------
    GeoDataFrame : gdf
        Return a geodataframe

    """
    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(filepath)

        # Get the original coordinate reference system (CRS)
        original_crs = gdf.crs
        print(f"coordinate system: {original_crs}")
        #Display the first few rows of the data
        print(gdf.head())  # 
  
        return gdf
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_gdr_file(grd_file_path, original_crs = 'EPSG:4326', output_filepath: str = None):
    """
    Function to read a .grd (grid file, Golden Software 7 Binary Grid). It is reading as a raster file without coordinate system. Tehn we will asign a coordinate system (src).
    
    Parameters
    ----------
        grd_file_path: str
            path of the raster file
        original_crs: (str), optional (by defaul: "WGS84" = "EPSG:4326")
            The original coordinate reference system (crs) of the grd file. User need to know it externally. By itself, a grd file does not provide information of the crs
        output_filepath : (str), optional
            Path to save the .tif version of the grd file. If empty, no file is exported
    Returns
    -------
    raster: raster object
        return a raster object containing the info of the grd and in addtion the src.
    
    """

    # Open the raster file
    raster = rasterio.open(grd_file_path)  

    
    # Read metadata
    metadata = raster.meta.copy()
    print("Metadata:", metadata)
        
    # Read the first band
    band1 = raster.read(1)
    print("First band data:", band1)
        
    # Read the shape of the raster
    print("Shape:", band1.shape)

    # Get raster bounds
    bounds = raster.bounds
    print("Bounds:", bounds)

    # Check if the raster has a CRS
    if raster.crs is None:
        print("The raster file does not have a CRS. We will assign the coordinate system, by defaul: WGS84 (EPSG:4326).")
            
        # Define the Coodrinate Reference System
        crs_to_assign = original_crs

        # Copy the metadata and assign the CRS
        meta = raster.meta.copy()
        meta.update({
                'crs': crs_to_assign,
                'transform': raster.transform
        })

        # Write in memory a new raster file and asign the coordinate system
        new_raster = rasterio.MemoryFile().open(**meta)
        for i in range(1, raster.count + 1):
            reproject(
                source=raster.read(i),
                destination=rasterio.band(new_raster, i),
                src_transform=raster.transform,
                src_crs=crs_to_assign,
                dst_transform=raster.transform,
                dst_crs=crs_to_assign,
                resampling=Resampling.nearest)
        raster = new_raster

        # Print
        crs = raster.crs
        print("The coodrinate sytem were asign CRS:", crs)

        # Save the updated raster with the new CRS
        if output_filepath != None:
            with rasterio.open(output_filepath, 'w', **meta) as dst:
                for i in range(1, raster.count + 1):
                    dst.write(raster.read(i), i)

            # Reopen the raster file with the new CRS assigned
            raster = rasterio.open('raster_with_wgs84.tif')
            return raster
            
    return raster

def plot_raster_in_memory(raster):
    """
    Plots a raster file that was already readed (it is in memory).

    Paratemers
    ----------

    raster: raster 
        Raster dataset object to plot.
    """

    plt.figure(figsize=(10, 6))

    # Read the first band of the raster for plotting
    data = raster.read(1)
    
    # Get the geographic transform information
    transform = raster.transform
    
    # Calculate the extent (geographical bounds) of the raster
    extent = (
        transform[2],  # xmin
        transform[2] + raster.width * transform[0],  # xmax
        transform[5] + raster.height * transform[4],  # ymin
        transform[5]  # ymax
    )


    # Plot the raster data
    img = plt.imshow(data, cmap='viridis', extent=extent)
    
    # Add a color bar to the plot
    plt.colorbar(img, ax=plt.gca(), orientation='vertical', label='Value')
    
    #Axis labels and title
    plt.title("Raster Plot")
    plt.xlabel('latitude')
    plt.ylabel('Longitude #')
    
    plt.show()

def plot_vectorGIS_in_memory(gdf, title='Map Plot', xlabel='Longitude', ylabel='Latitude'):
    """
    Function to plot a vector GIS file (GeoDataFrame).

    Parameters
    ----------
    gdf : (GeoDataFrame)
        The GeoDataFrame to plot.
    title : (str)
        Title of the plot.
    xlabel : (str)
        Label for the x-axis.
    ylabel : (str)
        Label for the y-axis.
    
    Return
    ------
    Map plot
    
    """
    
    gdf.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_vectorGIS_in_memory_plus_basemap(gdf, title='Map Plot', xlabel='Longitude', ylabel='Latitude', use_basemap=False, basemap_source=ctx.providers.OpenStreetMap.Mapnik):
    """
    Function to plot a vector GIS file (GeoDataFrame) with an optional basemap.
    
    Parameters
    ----------
    gdf : (GeoDataFrame)
        The GeoDataFrame to plot.
    title : (str)
        Title of the plot.
    xlabel : (str)
        Label for the x-axis.
    ylabel : (str)
        Label for the y-axis.
    use_basemap : (bool)
        Whether to include a basemap using contextily.
    basemap_source : (ctx.providers)
        The basemap source to use, default is OpenStreetMap.Mapnik. Other option are available like: ctx.providers.OpenStreetMap.Mapnik, ctx.providers.CartoDB.Positron,ctx.providers.CartoDB.Voyager
                                                
                                                            
    
    Return
    ------
    None
        Displays the map plot.
    """
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, alpha=0.7, edgecolor='k')
    
    # Adding a basemap if requested
    if use_basemap:
        if gdf.crs is None:
            raise ValueError("GeoDataFrame must have a CRS defined to use a basemap.")
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=basemap_source)
    
    # Adding labels and title
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    plt.show()

def read_and_reproject_vector_file(filepath: str, target_crs_name = 'EPSG:4326', output_filepath: str = None):
    """
    function to reads a vector file, prints supported CRS, checks its coordinate system,
    and reprojects it to the target CRS if needed.
    
    Parameters
    ----------
        filepath : (str)
            Path to the input vector file (.shp, .gpkg, etc.). 
        target_crs_name: (str), optional
            The name or EPSG code of the target coordinate system (e.g., "WGS84" or "EPSG:4326"). Default: 'EPSG:4326'
        output_filepath : (str), optional
            Path to save the reprojected file.
    
    Returns
    -------
    GeoDataFrame : reprojected or original 
    gdf : (geodataframe)
        geodataframe with the reprojected or original vector file info.
    """

    try:
        # Read the vector file
        gdf = gpd.read_file(filepath)
        
        # Get the original coordinate reference system (CRS)
        original_crs = gdf.crs
        print(f"Original CRS: {original_crs}")
        #print(gdf.head())  # Display the first few rows of the data

        # Determine the target CRS
        if target_crs_name.lower() == "EPSG:4326":
            target_crs = "EPSG:4326"
        else:
            try:
                target_crs = pyproj.CRS.from_string(target_crs_name).to_authority()[1]
                target_crs = f"EPSG:{target_crs}"
            except pyproj.exceptions.CRSError:
                print(f"Invalid CRS name or code: {target_crs_name}")
                return None
        
        # Check if the CRS is different from the target CRS
        if original_crs != target_crs:
            print ("reproyecting file....")
            # Reproject to the target CRS
            gdf = gdf.to_crs(target_crs)
            if output_filepath:
                # Save the reprojected file if output path is provided
                gdf.to_file(output_filepath, driver='GPKG' if output_filepath.endswith('.gpkg') else 'ESRI Shapefile')
        else:
            print(f"The file is already in {target_crs_name} ({target_crs}) CRS.")
        
        return gdf
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def read_and_reproject_grd_file(input_filepath: str, original_crs: str ='EPSG:4326' , target_crs_name = 'EPSG:4326', output_filepath: str = None):
    
    """
    Reprojects a grd file to a specified coordinate reference system (CRS).
    Given that in a grd file. The coordinate system is defined externally, We need to read the orginal crs and also the target crs.

    Parameters
    ----------

        input_filepath : str 
            Path to the input raster file.
        original_crs : str : (by defaul WG84: EPSG:4326)
            original coodinate refence system of the grd. By itself, a grd does not have info of he crs. User needs to know it externally. 
        target_crs_name : str, optional
            The target CRS for the reprojection (default is 'EPSG:4326').
        output_filepath : str, optional
            Path to save the reprojected raster file. If empty, no files are exported.

    Returns
    -------
    raster : raster file
        reproyected raster to the target coordinate system. 
    """
    
    #red raster file
    raster = rasterio.open(input_filepath)


    try:
        
        # Check if raster doesn't have a as CRS (Coordinate Reference System). If not, this function always will asign a WGS84 system as defaul. 
        if raster.crs is None:
            print("The raster file does not have a CRS. We will assgin the original_crs")
        
            # Copy the metadata and assign the CRS
            meta = raster.meta.copy()
            meta.update({
                'crs': original_crs,
                'transform': raster.transform
            })

            new_raster = rasterio.MemoryFile().open(**meta)
            for i in range(1, raster.count + 1):
                reproject(
                    source=raster.read(i),
                    destination=rasterio.band(new_raster, i),
                    src_transform=raster.transform,
                    src_crs=original_crs,
                    dst_transform=raster.transform,
                    dst_crs=original_crs,
                    resampling=Resampling.nearest)
            raster = new_raster  

            # Print
            crs = raster.crs
            print("The original coodrinate sytem were asign CRS:", crs)
            
        # Check if raster is in the targer coordinate system and reproject raster if not
        if raster.crs.to_string() != target_crs_name:
            print(f"The raster file is not in WGS84 system: {target_crs_name}")
            
            #Reproyect 
            transform, width, height = calculate_default_transform(
                raster.crs, target_crs_name, raster.width, raster.height, *raster.bounds)
            kwargs = raster.meta.copy()
            kwargs.update({
                'crs': target_crs_name,
                'transform': transform,
                'width': width,
                'height': height
            })

            print(f"Reproyecting raster to WGS84")
            reprojected_raster = rasterio.MemoryFile().open(**kwargs)
            for i in range(1, raster.count + 1):
                reproject(
                    source=raster.read(i),
                    destination=rasterio.band(reprojected_raster, i),
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    dst_transform=transform,
                    dst_crs=target_crs_name,
                    resampling=Resampling.nearest)
            raster = reprojected_raster
        else: 
            print("The raster is already in WGS84 (EPSG:4326).")
        return raster
    
    except Exception as e:
        print(f"An error occurred reading the file: {e}")
        return None, None
    
def read_and_reproject_raster_file(input_filepath: str, target_crs_name = 'EPSG:4326', output_filepath: str = None):
    
    """
    Reprojects a raster file to a specified coordinate reference system (CRS).
    The default CRS is WGS84 (EPSG:4326).

    Parameters
    ----------

    input_filepath : str 
        Path to the input raster file.
    target_crs_name : str, optional
        The target CRS for the reprojection (default is 'EPSG:4326').
    output_filepath : str, optional
        Path to save the reprojected raster file.

    Returns
    -------
    raster : raster file
        reproyected raster to the target coordinate system 
    """
    
    #red raster file
    raster = rasterio.open(input_filepath)


    try:             
        # Check if raster is in the targer coorfinate system and reproject raster if not
        if raster.crs.to_string() != target_crs_name:
            print(f"The raster file is not in WGS84 system: {target_crs_name}")
            
            #Reproyect 
            transform, width, height = calculate_default_transform(
                raster.crs, target_crs_name, raster.width, raster.height, *raster.bounds)
            kwargs = raster.meta.copy()
            kwargs.update({
                'crs': target_crs_name,
                'transform': transform,
                'width': width,
                'height': height
            })

            print(f"Reproyecting raster to WGS84")
            reprojected_raster = rasterio.MemoryFile().open(**kwargs)
            for i in range(1, raster.count + 1):
                reproject(
                    source=raster.read(i),
                    destination=rasterio.band(reprojected_raster, i),
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    dst_transform=transform,
                    dst_crs=target_crs_name,
                    resampling=Resampling.nearest)
            raster = reprojected_raster
        else: 
            print("The raster is already in WGS84 (EPSG:4326).")
        return raster
    
    except Exception as e:
        print(f"An error occurred reading the file: {e}")
        return None, None

def Writesupportedcoodinatesys():
    # Print all supported coordinate systems (CRS) by GeoPandas
    print("Supported CRS by GeoPandas:")
    for crs in pyproj.database.query_crs_info():
        print(f"{crs.code}: {crs.name}")

def extract_raster_by_vectorGISfile_mask_previous(raster, vectorGIS_gdf, output_raster_path: str = None):
    """
    Uses the shapefile as a mask to intersect with the raster file and 
    extracts a new raster containing only the pixels within the shapefile.

    Parameters
    ----------
    raster : Rasterio dataset object of the raster file.
    shapefile_gdf : GeoDataFrame  
        Geodataframe comming from reading a vector GIS file.
    output_raster_path  : str, optional
        Path to save the masked raster.

    Returns
    -------
        out_image: contain the raster  out_meta

    """
    # Convert GeoDataFrame to GeoJSON-like format
    shpfile = [feature["geometry"] for feature in vectorGIS_gdf.__geo_interface__["features"]]

    # Mask the raster using the shapefile geometry
    out_image, out_transform = mask(raster, shpfile, crop=True)

    # Update the metadata with the new dimensions and transform
    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    if output_raster_path != None:
        # Write the masked raster to a new file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Masked raster saved to {output_raster_path}")

    return out_image, out_meta

def extract_raster_by_vectorGISfile_mask(raster, vectorGIS_gdf, output_raster_path: str = None):
    """
    Uses the vectorGISfile as a mask to intersect with the raster file and 
    extracts a new raster containing only the pixels within the shapefile.

    Parameters
    ----------
        raster : Rasterio dataset object of the raster file.
        vectorGIS_gdf : GeoDataFrame  
            Geodataframe comming from reading a vector GIS file.
        output_raster_path  : str, optional
            Path to save the masked raster.

    Returns
    -------
        raster: raster object

    """
    # Convert GeoDataFrame to GeoJSON-like format
    shpfile = [feature["geometry"] for feature in vectorGIS_gdf.__geo_interface__["features"]]

    # Mask the raster using the shapefile geometry
    out_image, out_transform = mask(raster, shpfile, crop=True)

    # Update the metadata with the new dimensions and transform
    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    src = raster.crs.to_string()

    # Write in memory a new raster info and extension of the mask
    new_raster = rasterio.MemoryFile().open(**out_meta)
    for i in range(1, raster.count + 1):
        reproject(
            source=raster.read(i),
            destination=rasterio.band(new_raster, i),
            src_transform=raster.transform,
            src_crs=src,
            dst_transform=raster.transform,
            dst_crs=src,
            resampling=Resampling.nearest)
    raster = new_raster


    if output_raster_path != None:
        # Write the masked raster to a new file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Masked raster saved to {output_raster_path}")

    return raster
    

    """
    Plots the new raster created from the intersection and the shapefile used as a mask.

    :param masked_raster_data: Array of the masked raster data.
    :param masked_raster_meta: Metadata of the masked raster.
    :param shapefile_gdf: GeoDataFrame of the shapefile used as a mask.
    
    """
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the raster data
    #min_value = masked_raster_meta('min')
    #max_value = masked_raster_meta('max')
    #nodata_value = masked_raster_meta('nodata')
    
    #band1=masked_raster_data.read(1)
    #min_value= band1.min()
    #max_value=band1.max()
    #vmin=min_value, vmax=max_value

    plt.imshow(masked_raster_data[0], cmap='viridis', extent=(
        masked_raster_meta['transform'][2],
        masked_raster_meta['transform'][2] + masked_raster_meta['width'] * masked_raster_meta['transform'][0],
        masked_raster_meta['transform'][5] + masked_raster_meta['height'] * masked_raster_meta['transform'][4],
        masked_raster_meta['transform'][5]
    ))
    plt.colorbar(label='Pixel values')
    
    #img = plt.imshow(band1, cmap='viridis', vmin=band1.min(), vmax=max_value)
    #plt.colorbar(img, ax=plt.gca(), orientation='vertical', label='Pixel values')


    # Plot the shapefile on top
    shapefile_gdf.boundary.plot(ax=ax, edgecolor='orange')
    
    plt.title('Masked Raster and Shapefile')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_raster_and_shapefile(raster, vectorGIS_gdf,title_of_plot: str ='Plot Raster and vectorGIS file'):
    """
    Plots a raster file and a shapefile.

    Parameters
    ----------
        raster : Rasterio dataset object.
        vectorGIS_gdf : GeoDataFrame.
            GeoDataFrame of the vectorGIS file.

    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the raster data using geographic coordinates
    raster_data = raster.read(1)
    
    # Calculate the min and max values
    min_value = raster_data.min()
    max_value = raster_data.max()
    
    extent = (raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top)
    img = ax.imshow(raster_data, cmap='viridis', extent=extent, vmin=min_value, vmax=max_value, origin='upper')

 
    # Plot the shapefile according to its geometry type
    if any(vectorGIS_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        vectorGIS_gdf[vectorGIS_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])].plot(ax=ax, facecolor="none", edgecolor='blue', linewidth=1.5, label='Polygons')
        
    if any(vectorGIS_gdf.geom_type.isin(['LineString', 'MultiLineString'])):
        vectorGIS_gdf[vectorGIS_gdf.geom_type.isin(['LineString', 'MultiLineString'])].plot(ax=ax, color='red', linewidth=1.5, label='Lines')
        
    if any(vectorGIS_gdf.geom_type.isin(['Point', 'MultiPoint'])):
        vectorGIS_gdf[vectorGIS_gdf.geom_type.isin(['Point', 'MultiPoint'])].plot(ax=ax, color='green', markersize=10, label='Points')


    # Add a color bar
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', label='Pixel values')
    
    plt.title(title_of_plot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def extract_vectorGIS_by_vectorGISfile_mask(shapefile_gdf, mask_gdf):
    """
    Extracts elements from the first shapefile that are within the second shapefile (used as a mask).
    Saves the extracted elements as a new GeoDataFrame in memory.

    Paratemers
    ----------
        shapefile_gdf : GeoDataFrame of the first shapefile.
        mask_gdf : GeoDataFrame of the second shapefile used as a mask.
    
    Returns
    -------
        return : A new GeoDataFrame containing the elements from the first shapefile that are within the mask.
    """
    # Ensure both GeoDataFrames have the same CRS
    if shapefile_gdf.crs != mask_gdf.crs:
        mask_gdf = mask_gdf.to_crs(shapefile_gdf.crs)
    
    # Perform the spatial intersection to extract elements within the mask
    extracted_gdf = gpd.overlay(shapefile_gdf, mask_gdf, how='intersection')
    
    return extracted_gdf


def plot_shapefiles_exposed(folder_path, target_crs="EPSG:4326", plot_title="Infrastructure assets exposed"):
    """
    Function to read, fix, and plot all shapefiles (points, lines, polygons) in a given folder.
    Ensures all shapefiles are in the specified coordinate system and returns the list of GeoDataFrames.
    
    Args:
        folder_path (str): Path to the folder containing shapefiles.
        target_crs (str): EPSG code for the target coordinate system. Defaults to WGS84 (EPSG:4326).
        plot_title (str): Title of the plot. Defaults to "Infrastructure assets exposed".
    
    Returns:
        list: A list of GeoDataFrames read and processed.
    """
    # Initialize a list to store GeoDataFrames and their names
    geoms = []
    file_names = []
    
    # Generate random colors for each file
    colors = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF))
        for _ in range(len(os.listdir(folder_path)))
    ]
    
    # Iterate through the folder to find shapefiles
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.shp'):  # Check if the file is a shapefile
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # Read the shapefile using GeoPandas
                gdf = gpd.read_file(file_path)
                
                # Fix invalid geometries using make_valid
                gdf["geometry"] = gdf["geometry"].apply(make_valid)

                # Remove empty or null geometries
                gdf = gdf[~gdf.is_empty]
                gdf = gdf[gdf["geometry"].notnull()]

                if gdf.empty:
                    print(f"Warning: {file_name} contains no valid geometries after fixing. Skipping.")
                    continue
                
                # Check if the CRS is the target CRS; if not, reproject it
                if gdf.crs is not None and gdf.crs != target_crs:
                    print(f"Reprojecting {file_name} to {target_crs}")
                    gdf = gdf.to_crs(target_crs)
                elif gdf.crs is None:
                    print(f"Warning: {file_name} has no CRS defined. Assigning {target_crs}.")
                    gdf.set_crs(target_crs, inplace=True)
                
                geoms.append(gdf)  # Add the GeoDataFrame to the list
                file_names.append(file_name)  # Keep track of the file names
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each GeoDataFrame with a different color
    for idx, gdf in enumerate(geoms):
        color = colors[idx % len(colors)]
        label = file_names[idx]
        if gdf.geometry.geom_type.isin(['Point', 'MultiPoint']).all():
            gdf.plot(ax=ax, marker='o', color=color, markersize=10, label=label)
        elif gdf.geometry.geom_type.isin(['LineString', 'MultiLineString']).all():
            gdf.plot(ax=ax, color=color, linewidth=2, label=label)
        elif gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
            gdf.plot(ax=ax, color=color, edgecolor='black', alpha=0.5, label=label)
        else:
            print(f"Unknown geometry type in {file_names[idx]}")


    # Customize the plot
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, title='Exposed infraestructure', title_fontsize=10)

    # Show the plot
    plt.show()
    
    # Return the list of GeoDataFrames
    return geoms

def plot_shapefiles_exposed_with_basemap(folder_path, target_crs="EPSG:3857", plot_title="Infrastructure assets exposed"):
    """
    Function to read, fix, and plot all shapefiles (points, lines, polygons) in a given folder.
    Ensures all shapefiles are in the specified coordinate system, overlays them on a basemap,
    and returns the list of GeoDataFrames.
    
    Args:
        folder_path (str): Path to the folder containing shapefiles.
        target_crs (str): EPSG code for the target coordinate system. Defaults to EPSG:3857 (Web Mercator).
        plot_title (str): Title of the plot. Defaults to "Infrastructure assets exposed".
    
    Returns:
        list: A list of GeoDataFrames read and processed.
    """
    # Initialize a list to store GeoDataFrames and their names
    geoms = []
    file_names = []
    
    # Generate random colors for each file
    colors = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF))
        for _ in range(len(os.listdir(folder_path)))
    ]
    
    # Iterate through the folder to find shapefiles
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.shp'):  # Check if the file is a shapefile
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # Read the shapefile using GeoPandas
                gdf = gpd.read_file(file_path)
                
                # Fix invalid geometries using make_valid
                gdf["geometry"] = gdf["geometry"].apply(make_valid)

                # Remove empty or null geometries
                gdf = gdf[~gdf.is_empty]
                gdf = gdf[gdf["geometry"].notnull()]

                if gdf.empty:
                    print(f"Warning: {file_name} contains no valid geometries after fixing. Skipping.")
                    continue
                
                # Check if the CRS is the target CRS; if not, reproject it
                if gdf.crs is not None and gdf.crs.to_string() != target_crs:
                    print(f"Reprojecting {file_name} to {target_crs}")
                    gdf = gdf.to_crs(target_crs)
                elif gdf.crs is None:
                    print(f"Warning: {file_name} has no CRS defined. Assigning {target_crs}.")
                    gdf.set_crs(target_crs, inplace=True)
                
                geoms.append(gdf)  # Add the GeoDataFrame to the list
                file_names.append(file_name)  # Keep track of the file names
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each GeoDataFrame with a different color
    for idx, gdf in enumerate(geoms):
        color = colors[idx % len(colors)]
        label = file_names[idx]
        gdf.plot(ax=ax, color=color, alpha=0.6, label=label)
    
    # Add basemap using contextily
    ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik)
    #ctx.providers.OpenStreetMap.Mapnik
    #ctx.providers.CartoDB.Positron
    #ctx.providers.CartoDB.Voyager

    # Customize the plot
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, title='Shapefiles', title_fontsize=10)

    # Show the plot
    plt.show()
    
    # Return the list of GeoDataFrames
    return geoms












def plot_geodataframes(geodataframes, colors=None, labels=None, title="GeoDataFrame Plot"):
    """
    Parameters
    ----------
    geodataframes : list of geodataframes.
    colors : optional
        list of colors.
    labels : optional
        list of labels
    title : option
        litle of the plot

    Returns
    -------
    Map of geodataframes
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, gdf in enumerate(geodataframes):
        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else None

        # Check if the GeoDataFrame contains polygons
        if gdf.geom_type.iloc[0] == 'Polygon' or gdf.geom_type.iloc[0] == 'MultiPolygon':
            # Set the color fill to transparent for polygons
            gdf.plot(ax=ax, edgecolor=color, facecolor="none", label=label)
        else:
            gdf.plot(ax=ax, color=color, label=label)

    if labels:
        plt.legend()

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def intersect_and_combine_attributes (gdf1, gdf2):
    """
    Prepares gdf1 by renaming the first two columns and then intersects gdf1 with gdf2.
    
    :param gdf1: The first GeoDataFrame (donor).
    :param gdf2: The second GeoDataFrame (receiver).
    :return: A new GeoDataFrame with attributes from both gdf1 and gdf2 for intersecting geometries.
    """
    # Ensure both GeoDataFrames have the same CRS
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    # Get the names of the first two columns in gdf1
    first_col_name = gdf1.columns[0]
    second_col_name = gdf1.columns[1]
    #print(f"First column name of gdf1: {first_col_name}")
    #print(f"Second column name of gdf1: {second_col_name}")

    
    # Perform the intersection and combine attributes
    intersected_gdf = gpd.overlay(gdf2, gdf1, how='intersection')

    return intersected_gdf

def raster_to_gdf_type_point(raster):
    """
    Converts a raster file to a GeoDataFrame where each pixel value is associated with its geographic location.
    
    :param raster_path: Path to the raster file.
    :return: A GeoDataFrame with each pixel represented as a Point geometry.
    """
    
    # Read the raster data
    raster_data = raster.read(1)  # Assuming single band, if multi-band, specify the correct band

        # Get the affine transform to calculate geographic coordinates
    transform = raster.transform

    # Create lists to store pixel values and their corresponding coordinates
    pixel_values = []
    geometries = []

    # Loop through each row and column in the raster
    for row in range(raster_data.shape[0]):
        for col in range(raster_data.shape[1]):
            # Get the pixel value
            value = raster_data[row, col]

            # Skip if the pixel value is nodata
            if value == raster.nodata:
                    continue

            # Calculate the geographic coordinates of the pixel
            x, y = rasterio.transform.xy(transform, row, col)

            # Create a Point geometry from the coordinates
            point = Point(x, y)

            # Store the pixel value and geometry
            pixel_values.append(value)
            geometries.append(point)

    # Create a GeoDataFrame from the pixel values and geometries
    gdf = gpd.GeoDataFrame({'pixel_value': pixel_values, 'geometry': geometries}, crs=raster.crs)

    return gdf

def raster_to_gdf_type_polygons(raster):
    """
    Converts a raster file into a GeoDataFrame where each pixel is represented as a square polygon.

    :param raster_path: Path to the raster file.
    :return: A GeoDataFrame where each pixel of the raster is represented as a polygon.
    """
   
    # Read the raster data
    raster_data = raster.read(1)  # Assuming single band, use .read() for the first band
        
    # Get the affine transform to calculate geographic coordinates
    transform = raster.transform
        
    # Create lists to store pixel values and corresponding polygons
    pixel_values = []
    geometries = []

    # Loop through each row and column in the raster
    for row in range(raster_data.shape[0]):
        for col in range(raster_data.shape[1]):
            # Get the pixel value
            value = raster_data[row, col]

            # Skip if the pixel value is nodata
            if value == raster.nodata:
                continue

            # Get the coordinates of the corners of the pixel (a polygon)
            minx, miny = rasterio.transform.xy(transform, row, col, offset='ul')  # Upper left
            maxx, maxy = rasterio.transform.xy(transform, row, col, offset='lr')  # Lower right
                
            # Create a polygon for the pixel
            pixel_polygon = box(minx, miny, maxx, maxy)

            # Store the pixel value and the corresponding polygon
            pixel_values.append(value)
            geometries.append(pixel_polygon)

    # Create a GeoDataFrame from the pixel values and geometries (polygons)
    gdf = gpd.GeoDataFrame({'pixel_value': pixel_values, 'geometry': geometries}, crs=raster.crs)

    return gdf

def analyze_gdf_column(gdf, column_name:str = 'pixel_value', bins=3, plot_title_name:str = 'Hazard Histogram', plot_x_axis_name: str ='hazard instenisy' ):
    """
    Performs statistical analysis, creates a histogram, and a table-histogram for a specific column in a GeoDataFrame.

    :param gdf: The GeoDataFrame containing the data.
    :param column_name: The name of the column to analyze.
    :param bins: The number of bins for the histogram (default is 10).
    :return: A frequency table (table-histogram) as a pandas DataFrame.
    """
    # Check if the column exists in the GeoDataFrame
    if column_name not in gdf.columns:
        raise ValueError(f"Column '{column_name}' not found in the GeoDataFrame.")

    # Drop rows with missing values in the specified column
    column_data = gdf[column_name].dropna()

    # Calculate statistics
    #print(f"Statistics for column '{column_name}':")
    #print(f"Mean: {column_data.mean()}")
    #print(f"Median: {column_data.median()}")
    #print(f"Standard Deviation: {column_data.std()}")
    #print(f"Minimum: {column_data.min()}")
    #print(f"Maximum: {column_data.max()}")
    #print(f"Total non-null entries: {len(column_data)}")
    #print("\n")

    # Create a histogram
    plt.figure(figsize=(8, 6))
    plt.hist(column_data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of affected assests by {plot_title_name}')
    plt.xlabel(plot_x_axis_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Create a table-histogram (frequency table)
    print(f"Table-Histogram (frequency table) for column '{column_name}':")
    frequency_table = pd.cut(column_data, bins=bins).value_counts().sort_index()
    total_count = len(column_data)

    # Create the frequency table DataFrame
    frequency_table_df = frequency_table.reset_index()
    frequency_table_df.columns = [f'{column_name} Range', 'Frequency']

    # Calculate the relative frequency and add it as a new column
    frequency_table_df['Relative Frequency'] = frequency_table_df['Frequency'] / total_count

    print(frequency_table_df)


    # Count missing values (NaN) or pixel value = 0
    missing_or_zero_count = column_data.isna().sum() + (column_data == 0).sum()

    # Calculate the ratio between missing/zero values and total elements
    total_count = len(column_data)
    ratio_of_asset_affected = (total_count-missing_or_zero_count) / total_count

    # Print the results for missing or zero values  
    #print(f"Number of elements with missing values or pixel value = 0: {missing_or_zero_count}")
    #print(f"Total elements in the column: {total_count}")
    print(f"Percentage of assets affected : {ratio_of_asset_affected * 100 :.4f}")
    print("\n")


    return frequency_table_df



def read_tif_file(tif_path):
    """
    Lee un archivo TIF y retorna el objeto raster.
    
    Parámetros:
        tif_path (str): Ruta al archivo TIF.
        
    Devuelve:
        rasterio.DatasetReader: Objeto raster abierto.
    """
    return rasterio.open(tif_path)

## ------------------------------------------------------------------------------------------------------------------------- ##

def extract_raster_values_within_polygons(raster, shapefile_path, output_crs=None):
    """
    Extrae valores de un archivo raster dentro de los polígonos de un shapefile y asigna NaN a los valores fuera de los polígonos.
    
    Parámetros:
        raster (rasterio.DatasetReader): Objeto raster abierto.
        shapefile_path (str): Ruta al archivo shapefile.
        output_crs (str, opcional): CRS en el que quieres trabajar. Si no se especifica, usa el CRS del shapefile.
        
    Devuelve:
        geopandas.GeoDataFrame: Shapefile con una nueva columna que contiene los valores medios del raster.
        list: Lista de cada raster extraído (matriz de datos y transformación).
    """
    shapefile = gpd.read_file(shapefile_path)
    
    if output_crs:
        shapefile = shapefile.to_crs(output_crs)
    
    if shapefile.crs != raster.crs:
        shapefile = shapefile.to_crs(raster.crs)

    # Crear columna 'id' si no existe
    if 'id' not in shapefile.columns:
        shapefile['id'] = shapefile.index

    mean_values = []
    extracted_rasters = []  # Guardar rasters extraídos para cada polígono

    for index, row in shapefile.iterrows():
        geom = [mapping(row['geometry'])]
        
        try:
            # Aplicar máscara y convertir a float para manejar np.nan
            out_image, out_transform = mask(raster, geom, crop=True)
            out_image = out_image[0].astype(float)  # Convertir a float

            # Asignar np.nan a los valores fuera del polígono
            out_image[out_image == raster.nodata] = np.nan

            # Calcular el valor medio solo dentro del polígono
            valid_data = out_image[~np.isnan(out_image)]
            mean_value = np.mean(valid_data) if valid_data.size > 0 else np.nan
            mean_values.append(mean_value)
            extracted_rasters.append((out_image, out_transform))

        except Exception as e:
            print(f"Error al procesar el polígono {row['id']}: {e}")
            mean_values.append(np.nan)
            extracted_rasters.append((None, None))

    shapefile['mean_value'] = mean_values
    return shapefile, extracted_rasters

## ------------------------------------------------------------------------------------------------------------------------- ##


def plot_raster(raster):
    """
    Grafica el archivo TIFF completo con una barra de color.
    
    Parámetros:
        raster (rasterio.DatasetReader): Objeto raster abierto.
    """
    # Leer los datos del raster y crear una figura
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Cargar los datos y graficar el raster
    img = show(raster, ax=ax, title="Full raster (TIFF)", cmap='viridis')
    
    # Agregar una barra de color ajustada al tamaño de la figura
    cbar = plt.colorbar(img.get_images()[0], ax=ax, orientation='vertical', fraction=0.04, pad=0.04)
    cbar.set_label("Values")
    
    # Labels
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
## ------------------------------------------------------------------------------------------------------------------------- ##

def plot_raster_with_shapefile(raster, shapefile):
    """
    Grafica el archivo TIFF con el shapefile superpuesto.
    """
    # Crear una figura
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Cargar los datos y graficar el raster
    img = show(raster, ax=ax, title="Raster Completo (TIFF)", cmap='viridis')
    
    # Agregar una barra de color ajustada al tamaño de la figura
    cbar = plt.colorbar(img.get_images()[0], ax=ax, orientation='vertical', fraction=0.04, pad=0.04)
    cbar.set_label("Values")
    
    # Adicionar el shapefile
    shapefile.boundary.plot(ax=ax, edgecolor='red')
    
    # Labels
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

## ------------------------------------------------------------------------------------------------------------------------- ##

def plot_extracted_rasters(extracted_rasters, shapefile):
    """
    Grafica cada porción del raster extraído dentro de cada polígono.
    """
    for i, (out_image, out_transform) in enumerate(extracted_rasters):
        if out_image is not None:
            # Crear una figura
            fig, ax = plt.subplots(figsize=(10, 10))
            
#             ax.set_title(f"Raster Extraído para Polígono {shapefile.iloc[i]['id']}")
            
            # Cargar los datos y graficar el raster
            img = show(out_image, transform=out_transform, ax=ax, title="Raster Extraído para Polígono (TIFF)", cmap='viridis')
            
            # Agregar una barra de color ajustada al tamaño de la figura
            cbar = plt.colorbar(img.get_images()[0], ax=ax, orientation='vertical', fraction=0.04, pad=0.04)
            cbar.set_label("Values")
    
    
#             show(out_image, transform=out_transform, ax=ax)
            
            # Labels
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()
        else:
            print(f"Raster no disponible para el polígono {shapefile.iloc[i]['id']}")
            
## ------------------------------------------------------------------------------------------------------------------------- ##


def extract_raster_values_within_polygons_large_raster(tiff_path, shapefile_path, output_crs=None):
    """
    Extracts values from a large raster file within the polygons of a shapefile, preserving resolution.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        shapefile_path (str): Path to the shapefile.
        output_crs (str, optional): CRS to use. If not specified, uses the CRS of the shapefile.
        
    Returns:
        geopandas.GeoDataFrame: Shapefile with a new column containing the mean raster values.
        list: List of each extracted raster (data matrix and transformation).
    """
    # Load shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Reproject if output_crs is specified
    if output_crs:
        shapefile = shapefile.to_crs(output_crs)
    
    mean_values = []
    extracted_rasters = []

    with rasterio.open(tiff_path) as raster:
        raster_crs = raster.crs  # Save the original CRS

        # If the raster lacks a CRS, assign WGS84
        if raster_crs is None:
            print("Raster file has no CRS. Assigning WGS84 (EPSG:4326).")
            with rasterio.open(tiff_path, 'r+') as raster:
                raster.crs = CRS.from_epsg(4326)

    # Reopen raster in read mode after potential update
    with rasterio.open(tiff_path) as raster:

        # Align shapefile CRS to raster CRS if needed
        if shapefile.crs != raster.crs:
            shapefile = shapefile.to_crs(raster.crs)

        # Add an 'id' column if missing
        if 'id' not in shapefile.columns:
            shapefile['id'] = shapefile.index

        # Process each polygon
        for index, row in shapefile.iterrows():
            geom = [mapping(row['geometry'])]
            
            try:
                # Extract raster values within polygon, preserving resolution
                out_image, out_transform = mask(raster, geom, crop=True)
                out_image = out_image[0].astype(float)  # Ensure float to handle NaNs
                
                # Set no-data values to NaN
                out_image[out_image == raster.nodata] = np.nan

                # Calculate mean within polygon
                valid_data = out_image[~np.isnan(out_image)]
                mean_value = np.mean(valid_data) if valid_data.size > 0 else np.nan
                mean_values.append(mean_value)
                extracted_rasters.append((out_image, out_transform))

            except Exception as e:
                print(f"Error processing polygon {row['id']}: {e}")
                mean_values.append(np.nan)
                extracted_rasters.append((None, None))

    # Add mean values to GeoDataFrame
    shapefile['mean_value'] = mean_values
    return shapefile, extracted_rasters


## ------------------------------------------------------------------------------------------------------------------------- ##

def plot_extracted_rasters_with_shapefile(extracted_rasters, shapefile_with_values):
    """
    Plots each extracted raster section within the corresponding polygon boundary.
    
    Parameters:
        extracted_rasters (list): List of tuples with each extracted raster (data array and transformation).
        shapefile_with_values (geopandas.GeoDataFrame): GeoDataFrame with the polygons and extracted raster statistics.
    """
    for i, (out_image, out_transform) in enumerate(extracted_rasters):
        if out_image is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            polygon = shapefile_with_values.iloc[i]['geometry']  # Get the polygon geometry
            
            # Plot the extracted raster section
            rasterio.plot.show(out_image, transform=out_transform, ax=ax, cmap='viridis')
            
            # Plot the shapefile polygon boundary
            gpd.GeoSeries(polygon).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5)
            
            # Add colorbar
            cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Raster Value Intensity")
            
            # Titles and labels
            ax.set_title(f"Extracted Raster for Polygon {shapefile_with_values.iloc[i]['id']}")
            plt.xlabel("East Coordinate")
            plt.ylabel("North Coordinate")
            plt.show()
        else:
            print(f"Raster not available for polygon {shapefile_with_values.iloc[i]['id']}")



def extract_focused_raster(tiff_path, shapefile_path, output_crs=None):
    """
    Extrae una ventana de un archivo raster grande basada en los límites del shapefile, luego extrae
    valores dentro de cada polígono sin perder resolución.
    
    Parámetros:
        tiff_path (str): Ruta al archivo TIFF.
        shapefile_path (str): Ruta al archivo shapefile.
        output_crs (str, opcional): CRS en el que quieres trabajar. Si no se especifica, usa el CRS del shapefile.
        
    Devuelve:
        geopandas.GeoDataFrame: Shapefile con una nueva columna que contiene los valores medios del raster.
        list: Lista de cada raster extraído (matriz de datos y transformación).
    """
    # Leer el shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Reproyectar el shapefile si se especifica un CRS diferente
    if output_crs:
        shapefile = shapefile.to_crs(output_crs)
    
    # Crear la columna 'id' si no existe
    if 'id' not in shapefile.columns:
        shapefile = shapefile.reset_index().rename(columns={'index': 'id'})

    # Obtener el límite del shapefile
    bounds = shapefile.total_bounds  # [minx, miny, maxx, maxy]
    
    with rasterio.open(tiff_path) as raster:
        raster_crs = raster.crs  # Save the original CRS

        # If the raster lacks a CRS, assign WGS84
        if raster_crs is None:
            print("Raster file has no CRS. Assigning WGS84 (EPSG:4326).")
            with rasterio.open(tiff_path, 'r+') as raster:
                raster.crs = CRS.from_epsg(4326)

    # Reopen raster in read mode after potential update
    with rasterio.open(tiff_path) as raster:         
        
        # Verificar y ajustar CRS
        if shapefile.crs != raster.crs:
            shapefile = shapefile.to_crs(raster.crs)

        # Calcular la ventana para el área de interés del shapefile
        window = raster.window(bounds[0], bounds[1], bounds[2], bounds[3])
        focused_transform = raster.window_transform(window)
        
        # Leer solo la ventana enfocada del raster
        focused_raster = raster.read(1, window=window)

        mean_values = []
        extracted_rasters = []  # Guardar rasters extraídos para cada polígono

        for index, row in shapefile.iterrows():
            geom = [mapping(row['geometry'])]
            
            try:
                # Extraer solo la parte del raster dentro del polígono dentro de la ventana enfocada
                out_image, out_transform = mask(raster, geom, crop=True)
                out_image = out_image[0].astype(float)  # Convertir a float para manejar np.nan
                
                # Asignar np.nan a los valores fuera del polígono
                out_image[out_image == raster.nodata] = np.nan

                # Calcular el valor medio solo dentro del polígono
                valid_data = out_image[~np.isnan(out_image)]
                mean_value = np.mean(valid_data) if valid_data.size > 0 else np.nan
                mean_values.append(mean_value)
                extracted_rasters.append((out_image, out_transform))

            except Exception as e:
                print(f"Error al procesar el polígono {row['id']}: {e}")
                mean_values.append(np.nan)
                extracted_rasters.append((None, None))

    # Agregar valores medios calculados al GeoDataFrame
    shapefile['mean_value'] = mean_values
    
    return shapefile, extracted_rasters

## ------------------------------------------------------------------------------------------------------------------------- ##

def plot_extracted_rasters_with_shapefile(extracted_rasters, shapefile_with_values):
    """
    Plots each extracted raster section within the corresponding polygon boundary and times the plotting process.
    
    Parameters:
        extracted_rasters (list): List of tuples with each extracted raster (data array and transformation).
        shapefile_with_values (geopandas.GeoDataFrame): GeoDataFrame with the polygons and extracted raster statistics.
    """
    start_time = time.time()  # Start the timer

    for i, (out_image, out_transform) in enumerate(extracted_rasters):
        if out_image is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            polygon = shapefile_with_values.iloc[i]['geometry']  # Get the polygon geometry
            
            # Plot the extracted raster section
            rasterio.plot.show(out_image, transform=out_transform, ax=ax, cmap='viridis')
            
            # Plot the shapefile polygon boundary
            gpd.GeoSeries(polygon).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.0, linestyle=':')
            
            # Add colorbar
            cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Raster Value Intensity")
            
            # Titles and labels
            ax.set_title(f"Extracted Raster for Polygon {shapefile_with_values.iloc[i]['id']}")
            plt.xlabel("East Coordinate")
            plt.ylabel("North Coordinate")
            plt.show()
            
            # Cerrar la figura explícitamente y liberar memoria
            plt.close(fig)
            del out_image  # Eliminar datos de la imagen
            gc.collect()   # Forzar la liberación de memoria
        else:
            print(f"Raster not available for polygon {shapefile_with_values.iloc[i]['id']}")
            
## ------------------------------------------------------------------------------------------------------------------------- ##

def save_extracted_rasters(extracted_rasters, shapefile_with_values, output_dir="/root/WB/EIRA_tool/EIRA/eira/output_data/"):
    """
    Guarda cada sección extraída del raster en un archivo TIF separado.
    
    Parámetros:
        extracted_rasters (list): Lista de tuplas con cada raster extraído (matriz de datos y transformación).
        shapefile_with_values (geopandas.GeoDataFrame): GeoDataFrame con los polígonos y estadísticas de raster extraídas.
        output_dir (str): Directorio donde se guardarán los archivos TIFF extraídos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (out_image, out_transform) in enumerate(extracted_rasters):
        if out_image is not None:
            polygon_id = shapefile_with_values.iloc[i]['id']
            output_path = os.path.join(output_dir, f"extracted_raster_{polygon_id}.tif")
            
            # Save the extracted raster
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=out_image.shape[0],
                width=out_image.shape[1],
                count=1,
                dtype=out_image.dtype,
                crs=shapefile_with_values.crs,
                transform=out_transform
            ) as dst:
                dst.write(out_image, 1)
            
            print(f"Raster guardado para el polígono {polygon_id} en {output_path}")



def join_extracted_rasters(extracted_rasters, tiff_path, output_path=None):
    """
    Joins extracted raster arrays into a single mosaic, retrieves the CRS from the original TIFF file,
    and optionally writes to a new raster file. Keeps nodata values assigned to the final mosaic for pixels with no data
    and replaces any pixel values greater than 2E9 with nodata.

    Parameters:
        extracted_rasters (list): List of tuples (array, transform) from the extracted rasters.
        tiff_path (str): Path to the original TIFF file, used to obtain the CRS and other metadata.
        output_path (str, optional): Path for the output raster file. If not provided, only the mosaic and
                                     transform are returned.

    Returns:
        list: [mosaic (numpy masked array), mosaic_transform (Affine)]
    """
    # Filter out None values in extracted rasters
    valid_rasters = [raster for raster, transform in extracted_rasters if raster is not None]
    valid_transforms = [transform for raster, transform in extracted_rasters if raster is not None]
    
    if not valid_rasters:
        raise ValueError("No valid raster data to join.")

    # Retrieve CRS and metadata from the original TIFF file
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        dtype = src.dtypes[0]
        nodata = src.nodata if src.nodata is not None else -9999  # Use -9999 as default nodata if not specified

    # List to hold open MemoryFile objects
    memfiles = []

    # Create MemoryFile datasets and keep them open until merge completes
    try:
        for raster, transform in zip(valid_rasters, valid_transforms):
            memfile = MemoryFile()
            memfiles.append(memfile)
            with memfile.open(
                driver="GTiff",
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype=dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ) as dataset:
                dataset.write(raster, 1)

        # Open each MemoryFile as a dataset for merging
        datasets = [memfile.open() for memfile in memfiles]
        
        # Merge datasets with masked=True to handle nodata values
        mosaic, mosaic_transform = merge(datasets, nodata=nodata, masked=True)

        # Assign nodata to pixels with values greater than 2E9
        mosaic = np.ma.masked_where(mosaic > 2e9, mosaic)

    finally:
        # Ensure all MemoryFiles are closed
        for memfile in memfiles:
            memfile.close()

    # Optionally write the combined raster to the specified output path
    if output_path:
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": 1,  # single band
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "crs": crs,  # Use the CRS from the original TIFF file
            "transform": mosaic_transform,
            "nodata": nodata
        }

        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic.filled(nodata)[0, :, :], 1)  # Ensure correct shape by selecting the first band
    
    # Return the masked mosaic array and transformation
    return [mosaic, mosaic_transform]


def join_extracted_rasters_2(extracted_rasters, tiff_path, output_path=None):
    """
    Joins extracted raster arrays into a single mosaic, retrieves the CRS from the original TIFF file,
    and optionally writes to a new raster file. Keeps nodata values assigned to the final mosaic for pixels with no data
    and replaces any pixel values greater than 2E9 with nodata.

    Parameters:
        extracted_rasters (list): List of tuples (array, transform) from the extracted rasters.
        tiff_path (str): Path to the original TIFF file, used to obtain the CRS and other metadata.
        output_path (str, optional): Path for the output raster file. If not provided, only the mosaic and
                                     transform are returned.

    Returns:
        list: [mosaic (numpy masked array), mosaic_transform (Affine)]
    """
    # Filter out None values in extracted rasters
    valid_rasters = [raster for raster, transform in extracted_rasters if raster is not None]
    valid_transforms = [transform for raster, transform in extracted_rasters if raster is not None]
    
    if not valid_rasters:
        raise ValueError("No valid raster data to join.")

    # Retrieve CRS and metadata from the original TIFF file
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        dtype = src.dtypes[0]
        nodata = src.nodata if src.nodata is not None else -9999  # Use -9999 as default nodata if not specified

    # List to hold open MemoryFile objects
    memfiles = []

    # Create MemoryFile datasets and keep them open until merge completes
    try:
        for raster, transform in zip(valid_rasters, valid_transforms):
            memfile = MemoryFile()
            memfiles.append(memfile)
            with memfile.open(
                driver="GTiff",
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype=dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ) as dataset:
                dataset.write(raster, 1)

        # Open each MemoryFile as a dataset for merging
        datasets = [memfile.open() for memfile in memfiles]
        
        # Merge datasets with masked=True to handle nodata values
        mosaic, mosaic_transform = merge(datasets, nodata=nodata, masked=True)

        # Assign nodata to pixels with values greater than 2E9
        mosaic = np.ma.masked_where(mosaic > 2e9, mosaic)

    finally:
        # Ensure all MemoryFiles are closed
        for memfile in memfiles:
            memfile.close()

    # Optionally write the combined raster to the specified output path
    if output_path:
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": 1,  # single band
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "crs": crs,  # Use the CRS from the original TIFF file
            "transform": mosaic_transform,
            "nodata": nodata
        }

        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic.filled(nodata)[0, :, :], 1)  # Ensure correct shape by selecting the first band
    
    # Create an in-memory raster dataset object with the mosaic
    in_memory_file = MemoryFile()
    in_memory_raster = in_memory_file.open(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=mosaic_transform,
        nodata=nodata
    )
    in_memory_raster.write(mosaic.filled(nodata)[0, :, :], 1)

    # Return the in-memory raster dataset object
    return in_memory_raster








def plot_in_memory_file(in_memory_file, band=1):
    """
    Plots the raster data from an in-memory raster file.

    Parameters:
        in_memory_file (MemoryFile): The in-memory raster file containing the data.
        band (int): The band number to plot (default is 1).

    Returns:
        None: Displays the plot.
    """
    # Open the in-memory file
    with in_memory_file.open() as dataset:
        # Read the specified band
        data = dataset.read(band, masked=True)
        
        # Plot the raster data using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis')
        plt.colorbar(label="Pixel Value")
        plt.title(f"Raster Band {band}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.grid(False)
        plt.show()





def plot_mosaic_from_output(output_list, title="Hazard models selected", nodata=None):
    """
    Plots the final mosaic raster using the output of the join_extracted_rasters function.

    Parameters:
        output_list (list): Output from join_extracted_rasters, containing [mosaic (numpy array), mosaic_transform (Affine)].
        title (str): Title for the plot.
        nodata (float, optional): Value in the raster representing no data. If provided, these values will be masked.
    """
    # Unpack the output list
    mosaic, mosaic_transform = output_list

    # Remove the first dimension if mosaic is 3D with a single band (1, height, width)
    if mosaic.ndim == 3 and mosaic.shape[0] == 1:
        mosaic = mosaic[0]

    # Mask nodata values if provided
    if nodata is not None:
        mosaic = np.ma.masked_equal(mosaic, nodata)

    # Create extent for spatial coordinates from the transform
    left, top = mosaic_transform * (0, 0)
    right, bottom = mosaic_transform * (mosaic.shape[1], mosaic.shape[0])
    extent = [left, right, bottom, top]

    # Plot the raster using plt.imshow
    plt.figure(figsize=(10, 8))
    img = plt.imshow(mosaic, cmap="viridis", extent=extent)
    plt.colorbar(img, label="Pixel Value")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()