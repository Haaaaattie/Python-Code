# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:48:04 2020

@author: yhe
"""


from pymongo import MongoClient
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import numpy as np


bondCounter = [x for x in input("Enter Bond Counts seperated by spaces: ").split(" ")]
date = input("Enter the date today, for example: 2020-04-04: ")

client = MongoClient('10.6.50.91', 27017)
db = client.UltraWire
bondData = db.BondData

for count in bondCounter:
    directory = "C:/Users/yhe/Desktop/"+str(date)
    for file in os.listdir(directory):
        if str(count) in file:
            path=os.path.join(directory+"\\"+str(file)+"/InfraredImagePass2.tif")
            print(path)
    img=Image.open(path)
    fig=plt.figure(figsize=(25, 4))
    plt.axis('off')
    width, height = img.size 
    top=0
    bottom = height
    image=np.array(img)
    y=int(height/2)
    cropPoint=np.array(np.where(image[y,:]!=0))
    left=cropPoint[0,0]-10
    right=left+6000
    image=img.crop((left, top, right, bottom))
    plt.imshow(image,cmap="gray")
    plt.savefig(directory+"\\"+str(file)+"/InfraredImagePass2.png", bbox_inches = 'tight',pad_inches = 0)
    byteImgIO = io.BytesIO()
    images_to_read = (directory+"\\"+str(file)+"/InfraredImagePass2.png")
    byteImg=Image.open(images_to_read)
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()
    bondData.update_one(
            {'BondCounter':int(count)},           
            {'$set':{"EL Image":byteImg}},
            upsert=True)