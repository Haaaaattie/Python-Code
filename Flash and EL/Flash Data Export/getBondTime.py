# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:04:32 2020

@author: yhe
"""

from pymongo import MongoClient
import pandas as pd 


client = MongoClient('10.6.50.91', 27017)
db = client.UltraWire
bondData = db.BondData
directory = "Z:\\Rui\Flash Test"
WorkOrder = input("Enter the Work Order Number: ")

bondCounter = [x for x in input("Enter the Bond Counts seperated by spaces: ").split(" ")]
EndTime=[]
HotPlateTemp=[]
BondHeadTemp=[]
for count in bondCounter:
# Retrieve data from Mongodb            
    data = bondData.find_one({"BondCounter":int(count)})
    for key,val in data.items():
        if 'EndTime' in key:
            val=str(val)
            EndTime.append(val)
        if 'HotPlateTemp' in key:
            val=str(val)
            HotPlateTemp.append(val)
        if 'BondHeadTemp' in key:
            val=str(val)
            BondHeadTemp.append(val)
            
df_list=list(zip(EndTime,HotPlateTemp, BondHeadTemp))          
df=pd.DataFrame(df_list, columns=['Bond Date','HotPlateTemp','BondHeadTemp'])
#print(df)

import os
out_path = os.path.join(directory + '\\' + str(WorkOrder)+'\\'+str(WorkOrder)+'_BondDate.xlsx')
writer = pd.ExcelWriter(out_path , engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
