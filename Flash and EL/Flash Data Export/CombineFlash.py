# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:45:13 2020

@author: yhe
"""

from skimage import io
from pymongo import MongoClient
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd 
import numpy as np
import glob
from skimage import color
import io

directory = "Z:\\Rui\Flash Test"
WorkOrder = input("Enter the Work Order Number: ")

path = os.path.join(directory +'/'+ str(WorkOrder))
files=glob.glob(path+"/*.xlsx")

all_data = pd.DataFrame()
for f in files:
    df = pd.read_excel(f)
    all_data = all_data.append(df,ignore_index=True)
    
out_path = os.path.join(directory +'/'+str(WorkOrder)+'/'+str(WorkOrder)+'.xlsx')
#' + str(WorkOrder)+'
writer = pd.ExcelWriter(out_path , engine='xlsxwriter')
all_data.to_excel(writer, sheet_name='Sheet1')
writer.save()
