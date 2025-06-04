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

User interface object for notebooks

This book defines classes, function and methods to interact with a user in a notebook 
"""

#Libraries
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt 

from eira import EIRA_GIS_tools
from eira.EIRA_utils import load_config 



def create_dropdown(csv_file_path: str = load_config()['inputs']['list_of_countries_path'], column_name: str = "NAME_LONG", separator: str = ';' ):
    """
    Creates an interactive dropdown menu with options given in a column of a cvs file readed
    
    
    Parameters
    ----------
        csv_file_path : str
            path of the csv file containing the list
        column_name : str
            name of column of which we want to generate the list
        separator : str optional
            separator used in the csv file or database (by defaul: ",")

    Returns
    -------
    object with the dropdown menue

        dropdown : dropdown

    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv (csv_file_path, sep = separator)
    df.head()
    # Extract the values form the specified column into a list
    value_list = df[column_name].to_list()

    # Add to the list of countries some options to:
     # Add a customized region
    value_list.append('customize region, upload a file')

    # Define the dropdown widget
    dropdown = widgets.Dropdown(
        options=value_list,
        value=None,  # Initially no value selected
        description='Select:',
        disabled=False,
    )


    # Display the dropdown menu
    display(dropdown)
    
    return dropdown

def check_if_selection_changed_return_country_gdf(dropdown, messagewhenchange='The country selected is: '):
    """
    Function to monitor dropdown selection changes and store the new selection in a variable.

    Parameters
    ----------
    dropdown: dropdown object
        The dropdown widget to monitor for changes.
    messagewhenchange: str, optional
        Message to prepend to the current selection output. Default: 'The country selected is: '

    Returns
    -------
    get_selected_geodataframe: function
        A function that returns the geospatial data of the selected country.
        In addition. This function also plot the geodataframe in a referential map
    """

    selected_country = None  # Variable to store the selected country
    gdf_shape_referential = None  # Variable to store the selected country's geospatial data

    def on_selection_change(change):
        nonlocal selected_country  # Allow modifying the outer variable
        nonlocal gdf_shape_referential 

        if change['type'] == 'change' and change['name'] == 'value':
            selected_country = change['new']  # Update the variable
            print(f"{messagewhenchange}{selected_country}")  # Print the updated selection


             # Clear previous plots to prevent overlap
            plt.close('all')  # Close all existing figures

            #Plot
            #Extraction of the selected country
            world_map_path = load_config()['inputs']['WorldMap_path']

            gdf_shape_referential= EIRA_GIS_tools.extract_country_geodataframe(world_map_path,selected_country)
            EIRA_GIS_tools.plot_vectorGIS_in_memory_plus_basemap(gdf_shape_referential, selected_country,use_basemap=True)
    
    # Link the dropdown change to the handler function
    dropdown.observe(on_selection_change, names='value')

    #Return a function that allows access to the geospatial data
    return lambda: gdf_shape_referential  # Return a function to access to the geospatial data



def check_if_selection_changed_return_country_name(dropdown, messagewhenchange='The country selected is: '):
    """
    Function to monitor dropdown selection changes and store the new selection in a variable.

    Parameters
    ----------
    dropdown: dropdown object
        The dropdown widget to monitor for changes.
    messagewhenchange: str, optional
        Message to prepend to the current selection output. Default: 'The country selected is: '

    Returns
    -------
    selected_country: Lamba function:  function to access the selected value   
        The name of the currently selected country.
    """
    selected_country = None  # Variable to store the selected country

    def on_selection_change(change):
        nonlocal selected_country  # Allow modifying the outer variable
        if change['type'] == 'change' and change['name'] == 'value':
            selected_country = change['new']  # Update the variable
        print(f"{messagewhenchange} {change['new']}")  # Print the updated selection

    # Link the dropdown change to the handler function
    dropdown.observe(on_selection_change, names='value')
    return lambda: selected_country  # Return a function to access the selected value
