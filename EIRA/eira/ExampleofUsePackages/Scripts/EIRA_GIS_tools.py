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

This book defines classes to handle different file types and its corresponining geo spatial operations with them 
"""

#Libraries
import rasterio
import matplotlib.pyplot as plt


def read_raster_1(raster_file_path, band_number=None):
    """
    Function to read a raster of bands 
    
    Parameters:
    
        raster_file_path: (str) : name of the file
        band_number: list(int), optional : band number to read. Default: 1 

    -------

    Returns:
        tuple: A tuple containing the first band1 data (numpy array) and metadata (dict).
        band1: (numpy array): first band data of the raster file
        metadata: (dict): metadaba of the raster file
    """

    if not band_number:
        band_number=[1]

    # Path to your raster file
    #raster_file_path = 'path/to/your/rasterfile.tif'

    # Open the raster file
    with rasterio.open(raster_file_path) as src:
        # Read metadata
        metadata = src.meta.copy()
        print("Metadata:", metadata)
        
        # Read the first band
        band1 = src.read(band_number)
        print("First band data:", band1)
        
        # Read the shape of the raster
        print("Shape:", band1.shape)

        # Get raster bounds
        bounds = src.bounds
        print("Bounds:", bounds)

        # Get coordinate reference system
        crs = src.crs
        print("CRS:", crs)

    return band1, metadata



def plot_raster_1(band_data, title='Raster Plot', cmap='viridis'):
    """
    Fucntion to plot the given raster band data.
    
    Parameters:
    band_data (numpy array): The raster band data to plot.
    title (str): Title of the plot.
    cmap (str): Colormap to use for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(band_data, cmap=cmap)
    plt.colorbar(label='Pixel values')
    plt.title(title)
    plt.xlabel('Column #')
    plt.ylabel('Row #')
    plt.show()