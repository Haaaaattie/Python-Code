# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:08:58 2020

To Upload flash test data to MongoDB 

@author: yhe
"""

# Import Libraries 
from pymongo import MongoClient
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import pandas as pd 
import numpy as np

WorkOrder = [x for x in input("Enter the Work Order Number: ").split(" ")]
bondCounter = [x for x in input("Enter Bond Counts seperated by spaces: ").split(" ")]
#print(WorkOrder)

client = MongoClient('10.6.50.91', 27017)
db = client.UltraWire
bondData = db.BondData
directory = "Z:\\Rui\Flash Test"


for workOrder in WorkOrder:
    for count in bondCounter:
        path = os.path.join(directory + '\\' + str(workOrder) + '\\' + str(workOrder) +'_' + str(count)+ '.xlsx')
        #print(path)
        df = pd.read_excel(path, header = None)
        keys = list(df.iloc[0, np.r_[10:12,14:16,20:22,37,39,40,45,47,48]])
        keys = [str(key) for key in keys]
        #print(keys)
        values = list(df.iloc[1,np.r_[10:12,14:16,20:22,37,39,40,45,47,48]])
        update = dict(zip(keys,values))
        #print(update)
        bondData.update_one(
                {'BondCounter':int(count)},
                {'$set': update},
                upsert=True)
        
        
#    data = bondData.find_one({'BondCounter':int(count)})
        print(update)
        
        
    