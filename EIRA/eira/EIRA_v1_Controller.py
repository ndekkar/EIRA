# -*- coding: utf-8 -*-
"""
@author: ivc83 and JALT
"""

import json
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append('../')  # reference to relative path
import vulProbabilityDamageRatiosKernel as pdfLoss


# 1. read the configuration (.json) file which contains things that may change: 
# file names, parameters, etc.
jsonName = "vulProbabilityDamageRatiosConfig.json"
config = json.load(open(jsonName, encoding='utf-8'))

# 2. data access
print("  Loading inputs...", datetime.now())

# 2.1 read the information from the config file...
# ...for input
inpath = config['inputs']['inputpath']
file1 = config['inputs']['filenames']['mdrfile']
filenamesFV=config['inputs']['filenames']['filenamesFV']
NumpointsPDF = config['inputs']['paramvalues']['NumpointsPDF']
#D0 = config['inputs']['paramvalues']['D0']
#r= config['inputs']['paramvalues']['r']

# ... for output (at the end)
writeOutput = config['outputs']['write']
outpath = config['outputs']['outputpath']
outputfilename = config['outputs']['outputfilename']

#load the .csv file with the names of the FV's files
FVnames=pd.read_csv(inpath + '/' + filenamesFV)
#FVnames=pd.read_excel(inpath + '/' + filenamesFV)


# 2.3 Create the new folder where we save the output files
today = datetime.today()
outputrunid = today.strftime("%Y%m%d%H%M%S")
writepath = outpath + '/' + outputrunid
if config['outputs']['write']:
    print("  Writing outputs...", datetime.now())
    os.makedirs(writepath)
else:
    print("  Test run - no results written to file...", datetime.now())
    exit()

#3 Do de calculations 
#3.1 loop to work with several FVs
numFV=0
for ii, row in FVnames.iterrows():
    numFV=numFV+1
    #print(row['filenames'])
    #print(numFV)
    # 2.2 load the input data
    df1 = pd.read_csv(inpath + '/' + row['filenames'])


    # 3.2 do the calculations using functions stored in the kernel
    print("  Processing..." , "PDF_FV_" , str(numFV) , "_",  datetime.now())

    outputdf = pdfLoss.calculatePDFLoss(df1, NumpointsPDF)
    #print(outputdf)

    # 4. output the new file (write the .csv files)
    Aux_FV_name=row[0]       #extract the original name of the file
    Aux_FV_name=os.path.splitext(Aux_FV_name)[0]    #delete extension

    outputdf.to_csv(writepath + '/' + Aux_FV_name + '_PFD' + '.csv' , index=False)
    with open(writepath + '/' + jsonName, 'a', encoding="utf-8") as file:
        json.dump(config, file)
    file.close()
    
print("  Done.", datetime.now())

    

