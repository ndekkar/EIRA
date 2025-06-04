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

define utilities

This book defines classes to handle different utilities 
"""

#Libraries

import os
import sys
import json

def load_config():
    """
    Read config.json
    """
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'EIRA_Config.json'))
    #config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EIRA_Config.json'))         #'..'  : go one directory up
    #config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'EIRA_Config.json'))   #'..' ,'..'  : go two direcrory up
   
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
        #print(config_path)
    
    #root_folder_path = os.path.abspath('Auxname.txt') 
    #   /root/WB/EIRA_tool/EIRA/docs/../eira/../EIRA_Config.json'

    # 1. read the configuration (.json) file which contains things that may change: 
    # file names, parameters, etc.
    #jsonName = "/root/WB/EIRA_tool/EIRA/eira/EIRA_Config.json"

    #if os.path.exists(jsonName):
    #    with open(jsonName, encoding='utf-8') as f:
    #        config = json.load(f)
    #else:
    #    print (f" Error. File '{jsonName}' not found.")

     # 2.1 read the information from the config file...
    # ...for input
    #input_folder_path = config['inputs']['input_folder_path']
    #list_of_countries_path = config['inputs']['list_of_countries_path']
    #Worldmap_path =config['inputs']['WorldMap_path']    


    return config



