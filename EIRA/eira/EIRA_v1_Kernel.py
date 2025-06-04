# -*- coding: utf-8 -*-
"""
Created on Thu July 29, 20234
@author: Jose Antonio Leon
"""

from scipy.stats import beta
import pandas as pd
import numpy as np

def calculatePDFLoss(df, NumpointsPDF, FV_id_Col=0, IM=1,mdrcol=2, sd_mdr_col=3, Type_MDR_0_uncertainty = "MDR_0_without_uncertainty"):
    #IM,mdrcol,sd:mdr_col are the numbers of the columns in the FV input file. It better given that the column names could change.
    Mylist=[]

    column_headers = list(df.columns.values)  # It will be used to add intensity column to the final files and also to keep the original names of the columns
    

    for index, row in df.iterrows():
        
        #Parameters of the beta distribution
        FV_id=row[FV_id_Col]
        IntensityMeasure=row[IM]
        MDR= row[mdrcol]
        SD=row[sd_mdr_col]
        
        # Create the Beta Ditribution for NumpointsDF
                
        ##Generate the discrete points of damage for which we are going to compute the PDF
        NumSD=5  #number of stard deviations to consider from the mean
        min_damage= MDR-NumSD*SD
        if min_damage < 0:
            min_damage =0
        max_damage= MDR+NumSD*SD
        if max_damage > 1:
            max_damage = 1

        damage_step=(max_damage-min_damage)/(NumpointsPDF-1)
        
        if MDR==0 and SD==0:
            DamageVector=[0]
        else:
            DamageVector=np.arange(0,NumpointsPDF)*damage_step+min_damage

        #DamageVector=np.linspace(0,1,NumpointsPDF)    
    

        if MDR == 0:
            
            match Type_MDR_0_uncertainty:
                case "MDR_0_without_uncertainty":
                    
                    #aux_loss_0=np.linspace(0,0,NumpointsPDF-1)
                    aux_loss_0=np.linspace(0,0,0)
                    aux_loss_0=np.insert(aux_loss_0,0,1)
                
                    Mydf=pd.DataFrame(data={column_headers[mdrcol]:DamageVector,'BetaPDFLoss':aux_loss_0})
                    
                    Mydf[column_headers[IM]]=IntensityMeasure
                    Mydf[column_headers[FV_id_Col]]=FV_id


                    Mydf=Mydf[[column_headers[FV_id_Col],column_headers[IM],column_headers[mdrcol],'BetaPDFLoss']]
                    #Mydf=Mydf[[IM,'DamageRatioBin','BetaPDFLoss']]
                    Mylist.append(Mydf)
                    continue

                case "MDR_0_uncertainty_lineal":
                    aaa=1#Write code
                case "MDR_0_uncertainty_RMS_method":
                    aaa=1 #Write code

        C=SD/MDR
        a= (1-MDR-MDR*C**2)/C**2  
        b= a*((1-MDR)/MDR)
                 
                           #beta.cdf(1, a, b, loc=0, scale=1)
        Beta_CPDF=beta.cdf(DamageVector, a, b)
        
        BetaPDF= np.diff(Beta_CPDF)
        BetaPDF = np.insert(BetaPDF, 0, 0)
        
        
        #Small ajust to complete the full probabiltity (given that damage vector is trucated to certain num of SDs) 
        
        #MaxPDFvalue=np.max(BetaPDF)
        uu= np.unravel_index(BetaPDF.argmax(), BetaPDF.shape) #loaction of the max value in BetaPDF
        
        sumaBetaPDF = np.sum(BetaPDF)
        deltaFitPDF = 1-sumaBetaPDF 
        BetaPDF[uu[0]] = BetaPDF[uu[0]]+ deltaFitPDF
        # End ajust
    
               
        #d = {'col1': [1, 2], 'col2': [3, 4]}
        #df = pd.DataFrame(data=d)

        #Mydf=pd.DataFrame(data={'DamageRatioBin':DamageVector,'BetaPDFLoss':BetaPDF})
        Mydf=pd.DataFrame(data={column_headers[mdrcol]:DamageVector,'BetaPDFLoss':BetaPDF})
        Mydf[column_headers[IM]]=IntensityMeasure
        Mydf[column_headers[FV_id_Col]]=FV_id
         
        Mydf=Mydf[[column_headers[FV_id_Col],column_headers[IM],column_headers[mdrcol],'BetaPDFLoss']]
        Mylist.append(Mydf)

    df = pd.concat(Mylist)    
    

    return(df)