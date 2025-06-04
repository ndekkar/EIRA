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

from shapely.geometry import LineString, MultiPoint, Point
from pyproj import CRS

import warnings

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

def plot_raster_in_memory(raster, title: str = "Raster plot", units_pixel_value_bar: str = "Value"):
    """
    Plots a raster file that was already readed (it is in memory).

    Paratemers
    ----------
    title: (str) Title of the graphic.
    units_pixel_value_bar: (str) Units of the pixel values
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
    plt.colorbar(img, ax=plt.gca(), orientation='vertical', label= units_pixel_value_bar)
    
    #Axis labels and title
    plt.title(title)
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
    
    #plt.show()

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



def plot_and_read_geodataframes_exposed_with_basemap_from_folder_path(data_dict, target_crs="EPSG:4326", plot_title="Infrastructure assets exposed"):
    """
    Function to process and plot a dictionary of GeoDataFrames.
    Ensures all GeoDataFrames are in the specified coordinate system, overlays them on a basemap,
    and returns the processed dictionary. The difference with "plot_shapefiles_exposed_with_basemap" that it just plot geodataframes that I already read before

    Args:
        data_dict (dict): Dictionary where keys are filenames and values are GeoDataFrames.
        target_crs (str): EPSG code for the target coordinate system. Defaults to EPSG:3857 (Web Mercator).
        plot_title (str): Title of the plot. Defaults to "Infrastructure assets exposed".

    Returns:
        dict: A dictionary with cleaned and reprojected GeoDataFrames.
    """
    processed_geoms = {}  # Store processed GeoDataFrames
    colors = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) 
        for _ in range(len(data_dict))
    ]

    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, (file_name, gdf) in enumerate(data_dict.items()):
        try:
            # Fix invalid geometries
            gdf["geometry"] = gdf["geometry"].apply(make_valid)

            # Remove empty or null geometries
            gdf = gdf[~gdf.is_empty]
            gdf = gdf[gdf["geometry"].notnull()]

            if gdf.empty:
                print(f"Warning: {file_name} contains no valid geometries after fixing. Skipping.")
                continue

            # Ensure CRS consistency
            if gdf.crs is not None and gdf.crs.to_string() != target_crs:
                print(f"Reprojecting {file_name} to {target_crs}")
                gdf = gdf.to_crs(target_crs)
            elif gdf.crs is None:
                print(f"Warning: {file_name} has no CRS defined. Assigning {target_crs}.")
                gdf.set_crs(target_crs, inplace=True)

            processed_geoms[file_name] = gdf  # Store processed GeoDataFrame

            # Plot each GeoDataFrame with a different color
            gdf.plot(ax=ax, color=colors[idx % len(colors)], alpha=0.6, label=file_name)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Add basemap
    ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik)

    # Customize the plot
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, title='Shapefiles', title_fontsize=10)

    # Show the plot
    plt.show()
    

def plot_geodataframes_exposed_with_basemap_from_dic(geodataframes_dict, target_crs="EPSG:4326",plot_title="Infrastructure assets exposed"):
    """
    Plot all GeoDataFrames in the dictionary on a map with a base map, axis titles, and a legend.

    Args:
        geodataframes_dict (dict): Dictionary of GeoDataFrames with keys in the format "<filename>_Type-<geometry_type>".
        target_crs (str): Coordinate reference system for the plot (default is "EPSG:4326").
    """
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each GeoDataFrame
    for key, gdf in geodataframes_dict.items():
        # Reproject to the target CRS if necessary
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Plot the GeoDataFrame
        gdf.plot(ax=ax, label=key)

    # Add a base map (OpenStreetMap.Mapnik)
    ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik)

    # Set axis titles
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a legend
    ax.legend(title="Infraestructure Exposed", loc="upper right")

    # Set plot title
    ax.set_title(plot_title)

    # Show the plot
    plt.show()










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
            #gdf.plot(ax=ax, edgecolor=color, facecolor="none", label=label)
            gdf.plot(ax=ax, edgecolor=color, label=label)
        else:
            gdf.plot(ax=ax, color=color, label=label)

    if labels:
        plt.legend()

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def intersect_and_combine_attributes (gdf1, gdf2,output_path: str = None):
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
    #print(f"First column name of gdf1: {first_col_name}")   # For Debbuging
    #print(f"Second column name of gdf1: {second_col_name}") # For Debbuging

    
    # Perform the intersection and combine attributes
    intersected_gdf = gpd.overlay(gdf2, gdf1, how='intersection')

    # Export to shapefile if output_path is provided
    if output_path:
        intersected_gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"GeoDataFrame exported to {output_path}")
    return intersected_gdf



def extract_raster_values_to_gdf(raster, gdf, column_name='pixel_value'):
    """
    Extracts pixel values from a raster that overlap, intersect, or touch each element of a GeoDataFrame.

    Parameters:
    -----------
    raster : rasterio.DatasetReader
        The raster file already read into memory using rasterio.
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing geometries (e.g., points, lines, polygons).
    column_name : str, optional
        The name of the new column to store the extracted pixel values. Default is 'pixel_value'.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with an additional column (specified by `column_name`) containing the extracted pixel values.
    """
    # Create a copy of the input GeoDataFrame to avoid modifying the original
    new_gdf = gdf.copy()
    
    # Initialize a list to store the extracted raster values
    raster_values = []

    # Iterate over each geometry in the GeoDataFrame
    for index, row in new_gdf.iterrows():
        geometry = row.geometry
        
        # Use rasterio's mask function to extract the pixel values that intersect with the geometry
        try:
            out_image, out_transform = mask(raster, [mapping(geometry)], crop=True, all_touched=True)
            out_image = out_image[0]  # Remove the extra dimension
            
            # Calculate the mean value of the pixels that intersect with the geometry
                        # Suppress the specific RuntimeWarning for empty slices
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_value = np.nanmean(out_image[out_image != raster.nodata])
            raster_values.append(mean_value)
        except Exception:
            # If there's an error (e.g., no overlap), append NaN silently
            raster_values.append(np.nan)
            
            
            
            # Suppress warnings for empty slices
            #with np.errstate(all='ignore'):
            #    mean_value = np.nanmean(out_image[out_image != raster.nodata])
            #raster_values.append(mean_value)
        #except Exception as e:     (if you have a message)
        except Exception:
            # If there's an error (e.g., no overlap), append NaN
            raster_values.append(np.nan)
    
    # Add the extracted raster values as a new column in the GeoDataFrame
    new_gdf[column_name] = raster_values
    
    return new_gdf



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

def analyze_geodataframe_risk_point_type(gdf, column_name:str = 'pixel_value', bins=3, plot_title_name:str = 'Hazard', plot_x_axis_name: str ='hazard instenisy unit' ):
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
    
    #Extract original data incluidng null values( We will use to compute the element not affected)
    column_data_with_null = gdf[column_name]

    # Drop rows with missing values in the specified column
    column_data = gdf[column_name].dropna()

    if column_data.empty:
        print (f"The element exposed are not affected")
        print(f" Total element exposed: {len(column_data_with_null)}" )
        print(f"Percentage of assets affected : {0 * 100 :.1f} % ")
        print("\n")
        return

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
    plt.figure(figsize=(6, 5))
    plt.hist(column_data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of affected assets by {plot_title_name}')
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
    Null_or_missing_values=column_data_with_null.isna().sum()
    #missing_or_zero_count = column_data.isna().sum() + (column_data == 0).sum()

    # Calculate the ratio between missing/zero values and total elements
    total_count = len(column_data)   # Total number of elements affected with some level of hazard intensity
    total_count_including_nulls=len(gdf)
    ratio_of_asset_affected = (total_count_including_nulls-Null_or_missing_values) / total_count_including_nulls

    # Print the results for missing or zero values  
    #print(f"Number of elements with missing values or pixel value = 0: {missing_or_zero_count}")
    #print(f"Total elements in the column: {total_count}")
    print(f" Total element exposed: {total_count_including_nulls}" )
    print(f"Percentage of assets affected : {ratio_of_asset_affected * 100 :.1f} % ")
    print(f"Number of elements not affected : {Null_or_missing_values}")
    print (f"Total affected elements: {total_count}")
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


def extract_raster_values_within_polygons_large_raster_2(tiff_path, gdf_shapefile, output_crs=None):
    """
    Extracts values from a large raster file within the polygons of a shapefile, preserving resolution.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        gdf_shapefile (gdf): geodataframe of the shapefile.
        output_crs (str, optional): CRS to use. If not specified, uses the CRS of the shapefile.
        
    Returns:
        geopandas.GeoDataFrame: Shapefile with a new column containing the mean raster values.
        list: List of each extracted raster (data matrix and transformation).
    """
    
    # Load shapefile
    shapefile = gdf_shapefile.copy()  # Avoid modifying original data
    
    # Keep only essential columns
    shapefile = shapefile[['geometry']]  # Keep only geometry to prevent long column names
   
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


def extract_raster_values_within_polygons_large_raster_3(tiff_path, gdf_shapefile, output_crs=None):
    """
    Extracts values from a large raster file within the polygons of a shapefile, preserving resolution.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        gdf_shapefile (gdf): geodataframe of the shapefile.
        output_crs (str, optional): CRS to use. If not specified, uses the CRS of the shapefile.
        
    Returns:
        geopandas.GeoDataFrame: Shapefile with a new column containing the mean raster values.
        list: List of each extracted raster (data matrix and transformation).
    """
    # Load shapefile
    shapefile = gdf_shapefile.copy()  # Avoid modifying original data
    
    # Keep only essential columns
    shapefile = shapefile[['geometry']]  # Keep only geometry to prevent long column names

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

            # Validate geometry
            if not row['geometry'].is_valid:
                print(f"Invalid geometry at index {index}, attempting to fix...")
                row['geometry'] = row['geometry'].buffer(0)  # Fix invalid geometries

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
    If I want to plot de uotput file, use "plot_mosaic_from_output"

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
            #print(f"Raster unique values BEFORE writing: {np.unique(raster)}")  # Debug
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
        #print(f"Mosaic unique values BEFORE mask: {np.unique(mosaic)}")  # Debug
        # Assign nodata to pixels with values greater than 2E9
        mosaic = np.ma.masked_where(mosaic > 2e9, mosaic)
        #print(f"Mosaic unique values AFTER mask: {np.unique(mosaic)}")  # Debug

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


def plot_mosaic_from_output(output_list, title="Hazard models selected", nodata=None):
    """
    Plots the final mosaic raster using the output of the join_extracted_rasters function.
    This functions works just with the joint "join_extracted_rasters"

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




def join_extracted_rasters_3(extracted_rasters, tiff_path, output_path=None):
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
        rasterio.DatasetReader: In-memory raster dataset object containing the merged raster.
    """
    valid_rasters = [raster for raster, transform in extracted_rasters if raster is not None]
    valid_transforms = [transform for raster, transform in extracted_rasters if raster is not None]
    
    if not valid_rasters:
        raise ValueError("No valid raster data to join.")

    # Retrieve CRS and metadata from the original TIFF file
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        dtype = "float32"  # Force dtype to float32 for consistency
        nodata = src.nodata

    # Ensure nodata is within a safe range
    if nodata is None or nodata < -1e10 or nodata > 1e10:
        nodata = -9999  # Override nodata with a safe value

    #print(f"Using nodata value: {nodata}")  # Debugging

    memfiles = []
    try:
        for i, (raster, transform) in enumerate(zip(valid_rasters, valid_transforms)):
            #print(f"Raster {i} unique values BEFORE writing: {np.unique(raster)}")  # Debugging

            # Ensure masked arrays are properly converted
            if np.ma.is_masked(raster):
                raster = raster.filled(nodata)

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

            # Read back for debugging
            with memfile.open() as check_ds:
                check_array = check_ds.read(1)
                #print(f"Raster {i} unique values AFTER writing: {np.unique(check_array)}")  # Debugging

        # Open datasets from MemoryFiles
        datasets = [memfile.open() for memfile in memfiles]
        
        # Ensure nodata handling in merge
        mosaic, mosaic_transform = merge(datasets, nodata=nodata, masked=True)
        
        # Replace values greater than 2E9 with nodata
        mosaic = np.ma.masked_where(mosaic > 2e9, mosaic)
        #print(f"Mosaic unique values AFTER merge: {np.unique(mosaic)}")  # Debugging

    finally:
        # Close all MemoryFiles
        for memfile in memfiles:
            memfile.close()

    # Optionally write to output file
    if output_path:
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": 1,
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "crs": crs,
            "transform": mosaic_transform,
            "nodata": nodata
        }

        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic.filled(nodata)[0, :, :], 1)

    # Create in-memory raster
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

    return in_memory_raster



def extract_raster_values_within_polygons_large_raster_2(tiff_path, shapefile_path, output_crs=None):
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
        raster_crs = raster.crs  # Save original CRS

        # If raster has no CRS, assign WGS84
        if raster_crs is None:
            print("Raster file has no CRS. Assigning WGS84 (EPSG:4326).")
            raster_crs = CRS.from_epsg(4326)

        # Ensure CRS consistency
        if shapefile.crs != raster_crs:
            shapefile = shapefile.to_crs(raster_crs)

        # Add an 'id' column if missing
        if 'id' not in shapefile.columns:
            shapefile['id'] = shapefile.index

        # Ensure nodata is set properly
        nodata_value = raster.nodata if raster.nodata is not None else -9999

        # Process each polygon
        for index, row in shapefile.iterrows():
            geom = [mapping(row['geometry'])]

            try:
                # Extract raster values within polygon
                out_image, out_transform = mask(raster, geom, crop=True)

                # Ensure data type is float to handle NaNs
                out_image = out_image.astype(float)

                # Debug: Print unique values before replacing nodata
                print(f"Polygon {index} unique values BEFORE masking nodata: {np.unique(out_image)}")

                # Set explicit nodata values
                out_image[out_image == nodata_value] = np.nan

                # Debug: Print unique values after replacing nodata
                print(f"Polygon {index} unique values AFTER masking nodata: {np.unique(out_image)}")

                # Calculate mean only for valid pixels
                valid_data = out_image[~np.isnan(out_image)]
                mean_value = np.mean(valid_data) if valid_data.size > 0 else np.nan

                mean_values.append(mean_value)
                extracted_rasters.append((out_image[0], out_transform))  # Take first band

            except Exception as e:
                print(f"Error processing polygon {row['id']}: {e}")
                mean_values.append(np.nan)
                extracted_rasters.append((None, None))

    # Add mean values to GeoDataFrame
    shapefile['mean_value'] = mean_values
    return shapefile, extracted_rasters


def extract_country_geodataframe(filepath: str, selected_country_name: str, target_crs_name='EPSG:4326', output_filepath=None):
    """
    Extracts a GeoDataFrame of a specific country from a shapefile of the world.

    Parameters
    ----------
    filepath : (str)
        Path to the input vector file (.shp, .gpkg, etc.).
    selected_country_name : (str)
        Name of the country to extract, matching the "NAME_LONG" column in the shapefile.
    target_crs_name : (str), optional
        The name or EPSG code of the target coordinate system (e.g., "WGS84" or "EPSG:4326"). Default: 'EPSG:4326'.
    output_filepath : (str), optional
        Path to save the new GeoDataFrame for the extracted country.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing only the data for the selected country.
    """
    try:
        # Step 1: Read and reproject the vector file using the provided routine
        gdf = read_and_reproject_vector_file(filepath, target_crs_name)

        if gdf is None:
            raise ValueError("Failed to load or reproject the shapefile.")

        # Step 2: Filter the GeoDataFrame for the selected country
        country_gdf = gdf[gdf["NAME_LONG"] == selected_country_name]

        if country_gdf.empty:
            raise ValueError(f"Country '{selected_country_name}' not found in the shapefile.")

        print(f"Extracted GeoDataFrame for country: {selected_country_name}")

        # Step 3: Save the filtered GeoDataFrame, if an output path is provided
        if output_filepath:
            country_gdf.to_file(output_filepath, driver='GPKG' if output_filepath.endswith('.gpkg') else 'ESRI Shapefile')
            print(f"Saved extracted GeoDataFrame to: {output_filepath}")

        return country_gdf

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    


def segment_lines(gdf, control_length: float =  1.00, output_path=None):
    """
    Segments LineString geometries in a GeoDataFrame based on a given control length.
    Preserves all original vertices and only interpolates within segments that exceed max_length.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame with geometries in WGS84 (EPSG:4326).
    - control_length (float): Maximum segment length in kilometers. default 1km. 
    - output_path (str, optional): If provided, saves the new GeoDataFrame as a shapefile.

    Returns:
    - GeoDataFrame: A new GeoDataFrame with segmented LineStrings and segment lengths.
    """
    # Filter only LineString geometries
    gdf = gdf[gdf.geometry.type == "LineString"].copy()

    # Temporary reproject to EPSG:3857 (meters) to compute centroid correctly
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid.unary_union

    # Ensure the centroid is a single point
    if centroid.geom_type == "MultiPoint":
        centroid = centroid.centroid  # Get centroid of multipoint

    # Compute UTM Zone and ensure it is between 1 and 60
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_zone = max(1, min(utm_zone, 60))  # Ensure valid range

    # Determine correct EPSG code for UTM projection
    projected_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone

    # Create CRS for UTM projection
    projected_crs = CRS.from_epsg(projected_epsg)

    # Reproject original data to UTM for accurate length calculations
    gdf = gdf.to_crs(projected_crs)


    def split_line(line, max_length):
        """
        Splits a LineString into segments of max_length meters.
        Returns a list of (segment, segment_length_km).
        """
        coords = list(line.coords)
        new_lines = []
        current_line = [coords[0]]
        current_length = 0

        for i in range(1, len(coords)):
            seg = LineString([coords[i-1], coords[i]])
            seg_length = seg.length  # Length in meters

            if seg_length <= max_length:
                current_line.append(coords[i])
                
                #segment = LineString(current_line)
                #new_lines.append((segment, segment.length / 1000))  # Convert to km
                #current_line = [coords[i-1], coords[i]]
           
                if len(current_line) > 1:
                    segment = LineString(current_line)
                    new_lines.append((segment, segment.length / 1000))  # Convert to km
                    current_line = [coords[i]]
            else:
                # Break the segment into equal parts
                num_parts = int(seg_length // max_length) + 1                
                step = seg_length/num_parts
                
                # Generate equally spaced points along the segment
                #interpolated_coords = [[seg.interpolate(i * step).x, seg.interpolate(i * step).y] for i in range(1, num_parts)]
                
                
                interpolated_points = [
                    seg.interpolate(j * step)
                    for j in range(1, num_parts)
                ]

                # Check if the LineString has z coordinates (3D)
                has_z = len(coords[0]) == 3  # True if z coordinate exists

                # Extract coordinates based on whether z exists
                if has_z:
                    # Include z coordinate if it exists
                    interpolated_coords = [
                        (point.x, point.y, point.z) for point in interpolated_points
                    ]
                else:
                    # Use only x and y coordinates if z does not exist
                    interpolated_coords = [
                        (point.x, point.y) for point in interpolated_points
                    ]

                #interpolated_coords = [
                #     (point.x, point.y) for point in interpolated_points  # Only uses x and y coordinates
                #]
                
                #interpolated_coords = [
                #    (point.x, point.y, point.z) for point in 
                #    interpolated_points]

                #interpolated_coords = [
                #    [point.x, point.y] for point in 
                #    (seg.interpolate(j * max_length / num_parts) for j in range(1, num_parts))
                #]                
                              
                # Add original vertex + interpolated points
                current_line.extend(interpolated_coords)
                current_line.append(coords[i])  # Always add the original vertex        

                current_line2 = [current_line[0]]

                for ii in range(1,len(current_line)):
                    seg2 = LineString([current_line[ii-1], current_line[ii]])
                    seg_length2 = seg2.length  # Length in meters

                    if seg_length2 <= max_length:
                        current_line2.append(current_line[ii])
                
                        #segment = LineString(current_line2)
                        #new_lines.append((segment, segment.length / 1000))  # Convert to km
                        #current_line2 = [current_line[ii]]                    

                        if len(current_line2) > 1:
                            segment2 = LineString(current_line2)
                            new_lines.append((segment2, segment2.length / 1000))  # Convert to km
                            current_line2 = [current_line[ii]]
                
                current_line = [coords[i]]
        return new_lines    

    # Convert control length to meters
    max_length_meters = control_length * 1000

    # Process each line while preserving attributes
    new_rows = []
    for idx, row in gdf.iterrows():
        original_geometry = row.geometry
        if original_geometry.length > max_length_meters:
            segments = split_line(original_geometry, max_length_meters)
            for segment, length_km in segments:
                new_row = row.copy()
                new_row.geometry = segment
                new_row["Length_km"] = length_km
                new_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row["Length_km"] = original_geometry.length / 1000  # Convert to km
            new_rows.append(new_row)

    # Create new GeoDataFrame
    new_gdf = gpd.GeoDataFrame(new_rows, geometry="geometry", crs=projected_crs)

    # Reproject back to WGS84
    new_gdf = new_gdf.to_crs(epsg=4326)

    # Save to file if an output path is given
    if output_path:
        new_gdf.to_file(output_path, driver="ESRI Shapefile")

    return new_gdf



def analyze_geodataframe_risk_line_type(gdf, column_name: str = 'pixel_value', name_column_to_sum: str = 'Length_km', bins:str = 3, plot_title_name: str = 'Hazard', plot_x_axis_name: str = 'Hazard'):
    """
    :param gdf: The GeoDataFrame containing the data.
    :param column_name: The name of the column to analyze.
    :param name_column_to_sum: The column whose values need to be summed per bin.
    :param bins: The number of bins for the histogram.
    :param plot_title_name: Title for the histogram plot.
    :param plot_x_axis_name: Label for the x-axis.
    :return: A frequency table (table-histogram) as a pandas DataFrame.
    """
    # Validate if columns exist
    if column_name not in gdf.columns:
        raise ValueError(f"Column '{column_name}' not found in the GeoDataFrame.")
    if name_column_to_sum not in gdf.columns:
        raise ValueError(f"Column '{name_column_to_sum}' not found in the GeoDataFrame.")
    
    # Drop rows with missing values in the specified column
    column_data = gdf[column_name].dropna()
    valid_data = gdf.dropna(subset=[column_name])
    
    #Extrac the columns including nulls.  (null means that no pixel values which means that the element where not affected)
    column_data_with_nulls = gdf[column_name]

    #Lenght
    total_length_exposed=gdf[name_column_to_sum].sum()

    total_length_column_data = valid_data[name_column_to_sum].sum()
    total_length_valid_data = valid_data[name_column_to_sum].sum()
    
    if valid_data.empty:
        print (f"The element exposed are not affected")
        print(f"Total length exposed : {total_length_exposed} Km")
        print(f"Percentage of lenght affected : {0 * 100 :.1f} % ")
        print("\n")
        return




    #print(f"total length exposed:  {total_length_column_data}")
    #print(f"total length with any kind of affected level:  {total_length_valid_data}")

    # Compute histogram bins
    counts, bin_edges = np.histogram(column_data, bins=bins)
    
    #print (f"before change {bin_edges}")    # to debugging
    #print (f"after change {bin_edges[:-1]}")  # to debugging

    # Compute sum of values for name_column_to_sum per bin   
    bin_indices = np.digitize(valid_data[column_name], bins = bin_edges, right=False)     
    bin_indices[bin_indices > len(bin_edges) - 1] = len(bin_edges) - 1  # Ensure bin indices are within valid range /#This part is strange but work. No sure why I just have 3 bins but the indices from 1 to 4. 
    
    #print(valid_data[column_name]) # to debugging
    #print (bin_indices) # to debugging
    
    bin_sums = np.zeros(len(bin_edges) - 1)
    for i in range(1, len(bin_edges)):
        bin_sums[i-1] = valid_data.loc[bin_indices == i, name_column_to_sum].sum() if (bin_indices == i).any() else 0
    
    # Compute relative and cumulative frequency by element (given that I am counting segment of line what can be of any size,  this historgram (of element) is not much relevent)
    total_count = len(column_data)
    relative_freq = counts / total_count
    cumulative_freq = np.cumsum(relative_freq)
    relative_sum_values = bin_sums / bin_sums.sum()

    # Construct frequency table
    freq_table = pd.DataFrame({
        "Bin Range": [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_edges)-1)],
        #"Frequency_num_of_elements": counts,
        #"Relative Frequency_elements": relative_freq,
        #"Cumulative Frequency": cumulative_freq,
        "Sum of Longitude": bin_sums,
        "Distribution of the affected length": relative_sum_values
    })
    
    # Compute missing or zero value stats (just to Debugging)
    #missing_or_zero_count = gdf.isna().sum() + (gdf == 0).sum()
    
    ratio_of_asset_affected = (total_length_column_data) / total_length_exposed
    total_length_no_affected = total_length_exposed - total_length_column_data

    print(f"Total length exposed : {total_length_exposed} Km")
    print(f"Percentage of assets affected: {ratio_of_asset_affected * 100:.1f} % ")
    print(f"Total length not affected : {total_length_no_affected} Km")
    print(f"Total length affected: {total_length_column_data} Km")
    #print("\n")
    
    # Plot histogram by element (more useful for Debugging)
    #plt.figure(figsize=(6, 5))
    #plt.hist(column_data, bins=bin_edges, edgecolor='black', alpha=0.7)
    #plt.title(f'Histogram of affected assets by {plot_title_name}')
    #plt.xlabel(plot_x_axis_name)
    #plt.ylabel('Frequency')
    #plt.grid(True)
    #plt.show()
    
    # Plot sum of values
    plt.figure(figsize=(6, 5))
    plt.bar(bin_edges[:-1], bin_sums, width=np.diff(bin_edges), alpha=0.7, color='r')
    plt.title(f' Histogram of total affected length by {plot_title_name}')
    plt.xlabel(plot_x_axis_name)
    plt.ylabel(f'Sum of {name_column_to_sum}')
    plt.grid(True)
    plt.show()
    
    return freq_table


def extract_gdf_by_geometries_from_geofiles(folder_path, target_crs="EPSG:4326"):
    """
    Process all supported vector files in a folder, split them by geometry type,
    and ensure they are in the desired coordinate system (default is WGS84).

    Args:
        folder_path (str): Path to the folder containing vector files.
        target_crs (str): Desired coordinate reference system (default is "EPSG:4326").

    Returns:
        dict: A dictionary where keys are in the format "<filename>_Type-<geometry_type>"
              and values are the corresponding GeoDataFrames in the target CRS.
    """
    # List of supported file extensions by GeoPandas
    supported_extensions = [".shp", ".gpkg", ".geojson", ".json"]  # Add more if needed

    # Initialize a dictionary to store all GeoDataFrames
    output_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Get the file extension
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if the file has a supported extension
        if file_ext in supported_extensions:
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)

            try:
                # Read the file into a GeoDataFrame
                gdf = gpd.read_file(file_path)

                # Check if the CRS matches the target CRS
                if gdf.crs != target_crs:
                    print(f"Warning: {filename} is not in {target_crs}. Converting...")
                    gdf = gdf.to_crs(target_crs)  # Convert to the target CRS

                # Validate geometries
                if not gdf.geometry.is_valid.all():
                    print(f"Warning: {filename} contains invalid geometries. Attempting to fix...")
                    gdf.geometry = gdf.geometry.buffer(0)  # Attempt to fix invalid geometries

                # Get the unique geometry types in the GeoDataFrame
                geometry_types = gdf.geometry.type.unique()

                # Split the GeoDataFrame by geometry type
                for geom_type in geometry_types:
                    # Filter rows with the current geometry type
                    gdf_filtered = gdf[gdf.geometry.type == geom_type]

                    # Create a key in the format "<filename>_Type-<geometry_type>"
                    key = f"{os.path.splitext(filename)[0]}_Type-{geom_type.lower()}"

                    # Add the filtered GeoDataFrame to the dictionary
                    output_dict[key] = gdf_filtered

                    # Print confirmation
                    #print(f"Processed {key} with {len(gdf_filtered)} rows in {target_crs}.") for Debbuding

            except Exception as e:
                # Handle errors (e.g., invalid geometries or unsupported formats)
                print(f"Error processing {filename}: {e}")
                continue
    

    # Print the dictionary of all GeoDataFrames
    print("\nExposured file Read and Validated:")
    for key, gdf in output_dict.items():
        print(key)
    #    print(f"GeoDataFrame: {gdf}")

    # Return the dictionary of all GeoDataFrames
    return output_dict


def extract_gdf_by_geometries_from_geofiles2(folder_path):
    """
    Process all supported vector files in a folder, split them by geometry type,
    and ensure they are in the WGS84 (EPSG:4326) coordinate system.

    Args:
        folder_path (str): Path to the folder containing vector files.

    Returns:
        dict: A dictionary where keys are in the format "<filename>_Type-<geometry_type>"
              and values are the corresponding GeoDataFrames in WGS84.
    """
    # List of supported file extensions by GeoPandas
    supported_extensions = [".shp", ".gpkg", ".geojson", ".json"]  # Add more if needed

    # Initialize a dictionary to store all GeoDataFrames
    output_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Get the file extension
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if the file has a supported extension
        if file_ext in supported_extensions:
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)

            try:
                # Read the file into a GeoDataFrame
                gdf = gpd.read_file(file_path)

                # Check if the CRS is WGS84 (EPSG:4326)
                if gdf.crs != "EPSG:4326":
                    print(f"Warning: {filename} is not in WGS84 (EPSG:4326). Converting...")
                    gdf = gdf.to_crs("EPSG:4326")  # Convert to WGS84

                # Validate geometries
                if not gdf.geometry.is_valid.all():
                    print(f"Warning: {filename} contains invalid geometries. Attempting to fix...")
                    gdf.geometry = gdf.geometry.buffer(0)  # Attempt to fix invalid geometries

                # Get the unique geometry types in the GeoDataFrame
                geometry_types = gdf.geometry.type.unique()

                # Split the GeoDataFrame by geometry type
                for geom_type in geometry_types:
                    # Filter rows with the current geometry type
                    gdf_filtered = gdf[gdf.geometry.type == geom_type]

                    # Create a key in the format "<filename>_Type-<geometry_type>"
                    key = f"{os.path.splitext(filename)[0]}_Type-{geom_type.lower()}"

                    # Add the filtered GeoDataFrame to the dictionary
                    output_dict[key] = gdf_filtered

                    # Print confirmation
                    #print(f"Processed {key} with {len(gdf_filtered)} rows in WGS84.") for debbuding

            except Exception as e:
                # Handle errors (e.g., invalid geometries or unsupported formats)
                print(f"Error processing {filename}: {e}")
                continue

    # Return the dictionary of all GeoDataFrames
    return output_dict



def plot_raster_in_memory_with_gdf_basemap(raster, gdf, title="Infrastructure affected by the Hazard", units_pixel_value_bar="Value", adjust_window_plot_to="gdf", use_basemap=False, basemap_source=ctx.providers.OpenStreetMap.Mapnik):
    """
    Plots a raster and a geodataframe in the same figure, ensuring both are in WGS84.
    Includes a color bar for the raster values and an optional basemap with 70% transparency.

    Parameters:
        raster (rasterio.DatasetReader): The raster dataset already in memory.
        gdf (geopandas.GeoDataFrame): The geodataframe.
        title (str): Title of the plot. Default is "Infrastructure affected by the Hazard".
        units_pixel_value_bar (str): Units of the pixel values. Default is "Value".
        adjust_window_plot_to (str): Adjust the plot window to the boundaries of the "gdf" or "raster".
            Default is "gdf".
        use_basemap (bool): Whether to include a basemap. Default is False.
        basemap_source: The source of the basemap. Default is ctx.providers.OpenStreetMap.Mapnik.
    """
    # Verify and convert CRS to WGS84 (EPSG:4326) if necessary
    if raster.crs.to_string() != CRS.from_epsg(4326).to_string():
        print("Raster is not in WGS84. Converting...")
        # Create a destination array for the reprojected data
        dst_shape = raster.shape
        dst_data = np.empty(dst_shape, dtype=raster.dtypes[0])

        # Reproject the raster
        reproject(
            source=raster.read(1),
            destination=dst_data,
            src_transform=raster.transform,
            src_crs=raster.crs,
            dst_transform=raster.transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.nearest
        )
        raster_data = dst_data
        raster_transform = raster.transform
    else:
        raster_data = raster.read(1)
        raster_transform = raster.transform

    # Verify and convert GeoDataFrame CRS to WGS84 if necessary
    if gdf.crs.to_string() != CRS.from_epsg(4326).to_string():
        print("GeoDataFrame is not in WGS84. Converting...")
        gdf = gdf.to_crs(epsg=4326)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate the extent (geographical bounds) of the raster
    raster_extent = (
        raster_transform[2],  # xmin
        raster_transform[2] + raster.width * raster_transform[0],  # xmax
        raster_transform[5] + raster.height * raster_transform[4],  # ymin
        raster_transform[5]  # ymax
    )

    # Calculate the extent (geographical bounds) of the geodataframe
    gdf_extent = (
        gdf.total_bounds[0],  # xmin
        gdf.total_bounds[2],  # xmax
        gdf.total_bounds[1],  # ymin
        gdf.total_bounds[3]   # ymax
    )

    # Determine the plot extent based on the `adjust_window_plot_to` parameter
    if adjust_window_plot_to.lower() == "gdf":
        plot_extent = gdf_extent
    elif adjust_window_plot_to.lower() == "raster":
        plot_extent = raster_extent
    else:
        raise ValueError("Invalid value for `adjust_window_plot_to`. Use 'gdf' or 'raster'.")

    # Plot the raster data
    img = ax.imshow(raster_data, cmap='viridis', extent=raster_extent, alpha=0.9 if use_basemap else 1.0) #alpha control the transparency 

    # Add a color bar to the plot
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_label(units_pixel_value_bar)

    # Plot the geodataframe
    gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5)

    # Add a basemap if requested (with 70% transparency)
    if use_basemap:
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=basemap_source, alpha=0.5)  #alpha control the transparency   (the lower value the more transparence)

    # Set the plot extent
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])

    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Show the plot
    plt.show()