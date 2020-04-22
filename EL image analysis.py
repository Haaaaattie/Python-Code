# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:25:04 2020

EL image analysis only work for 3-cell string 

EL test

@author: yhe
"""
from pymongo import MongoClient
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import os
import pandas as pd 
import numpy as np
import glob
from matplotlib.pyplot import cm
from skimage import color
import cv2 as cv
import re
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline

import seaborn as sns

def read_image(filename):
    #Read the raw data in
    imarray = color.rgb2gray(io.imread(filename))
    #the images area 1024x1024 unless cropped
    (x,y)=np.shape(imarray)
    #make the x and y axes
    x = np.linspace(1,x,x)
    y = np.linspace(1,y,y)
    return imarray,x,y

def get_allhulls(im,thresRatio_contour):
    im = im/np.max(im)*255    # convert each pixel to 0-255 span
    thres=thresRatio_contour*np.max(im) # threshold = 0.15
    
    mx = np.array(im.astype(np.uint8))
    dst, th3 = cv.threshold(mx, thres, 255, cv.THRESH_BINARY);
    contours,hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour
    areas = [cv.contourArea(c) for c in contours]
    ind_sort = np.argsort(areas)
    
    pad1=contours[ind_sort[-1]]
    pad2=contours[ind_sort[-2]]
    pad3=contours[ind_sort[-3]]
    
    hull1 = cv.convexHull(pad1)
    hull1 = hull1.reshape(int(hull1.size/2),2)
    hull1 = np.append(hull1,[hull1[0,:]],axis=0)
    
    hull2 = cv.convexHull(pad2)
    hull2 = hull2.reshape(int(hull2.size/2),2)
    hull2 = np.append(hull2,[hull2[0,:]],axis=0)
    
    hull3 = cv.convexHull(pad3)
    hull3 = hull3.reshape(int(hull3.size/2),2)
    hull3 = np.append(hull3,[hull3[0,:]],axis=0)
    
    mask = np.zeros(mx.shape,np.uint8)

    cv.drawContours(mask,[hull1,hull2,hull3],-1,255,-2)
    return hull1,hull2,hull3

def get_singleHull(im,hullCount):
    im = im/np.max(im)*255    # convert each pixel to 0-255 span
    mx = np.array(im.astype(np.uint8))
    
    mask_single = np.zeros(mx.shape,np.uint8)
    cv.drawContours(mask_single,[hullCount],-1,255,-2)
    
    out = np.zeros_like(im) # Extract out the object and place into output image
    out[mask_single == 255] = im[mask_single == 255]
    (y, x) = np.where(mask_single == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    
    width=int(bottomx-topx)
    height=int(bottomy-topy)
    
    length=int(((max(width,height))/2))
    
    center_x=int(width/2)
    center_y=int(height/2)
    
    cropx_length=length-center_x
    cropy_length=length-center_y
    

    im_cropped = im[topy-cropy_length:bottomy+cropy_length, topx-cropx_length:bottomx+cropx_length]
# get hull pad for single cell       
    
    size_x,size_y = np.shape(im_cropped)
    if size_x % 2 == 0 and size_y % 2== 0:
        im_cropped=im_cropped
    elif size_x % 2 !=0 and size_y % 2== 0:
        im_cropped=im_cropped[:-1,:]
    elif size_y % 2 !=0 and size_x % 2 == 0:
        im_cropped=im_cropped[:,:-1]
    elif size_x % 2 !=0 and size_y % 2 !=0:
        im_cropped=im_cropped[:-1,:-1]
    
    
    im_cropped = np.array(im_cropped.astype(np.uint8))             # important 
    
 
#th3 = cv.adaptiveThreshold(im,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#                           cv.THRESH_BINARY,17,2)

#im = cv.medianBlur(im_cropped,5)
#th3 = cv.adaptiveThreshold(im,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#                           cv.THRESH_BINARY,17,2)
#th_blur = cv.medianBlur(th3,5)    
#contours,hierarchy = cv.findContours(th_blur, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ret,th = cv.threshold(im_cropped,28,255,cv.THRESH_BINARY)  
    
    
    contours,hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contours]
    ind_sort = np.argsort(areas)
    
    pad_single=contours[ind_sort[-1]]
    hull_single = cv.convexHull(pad_single)
    hull_single = hull_single.reshape(int(hull_single.size/2),2)
    hull_single = np.append(hull_single,[hull_single[0,:]],axis=0)
    
    mask_single = np.zeros(im_cropped.shape,np.uint8)
    cv.drawContours(mask_single,[hull_single],-1,255,-2)
    return im_cropped,hull_single,mask_single


"""
find blocking contact for each cell
"""
def find_blockContact(im_cropped,thresRatio_blockContact):
    im = np.array(im_cropped.astype(np.uint8))
    thres=thresRatio_blockContact*np.max(im)
    ret,th = cv.threshold(im,thres,255,cv.THRESH_BINARY)   
    #number of labels, label matrix, stat matrix, centroid matrix
    connectivity = 4
    num_labels,labels,stats,centroids = cv.connectedComponentsWithStats(th, connectivity, cv.CV_32S)
    indexes_group = np.argsort(stats[:, cv.CC_STAT_AREA])
    stats = stats[indexes_group] 
    for component_id, stat in zip(indexes_group, stats):
        single_component = (labels == component_id).astype(np.uint8) * 255
    return single_component


def find_sudo_hulls(imex,rect_approx,wafer_size,thres):
    
    # check if im_cropped is a square
    size_x,size_y = np.shape(imex)
    if size_x % 2 == 0 and size_y % 2 == 0:
        imex=imex
    elif size_x % 2 !=0 and size_y % 2== 0:
        imex=imex[:-1,:]
    elif size_y % 2 !=0 and size_x% 2 == 0:
        imex=imex[:,:-1]
    elif size_x % 2 !=0 and size_y % 2 !=0:
        imex=imex[:-1,:-1]
    
    imex = imex/np.max(imex)*255

    mx = np.array(imex.astype(np.uint8))
    Nx,Ny = imex.shape
    Xgrid,Ygrid = np.meshgrid(np.arange(Nx),np.arange(Ny))
    dst, th3 = cv.threshold(mx, thres, 255, cv.THRESH_BINARY);

    
    contours,hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Find the index of the largest contour
    areas = [cv.contourArea(c) for c in contours]
    ind_sort = np.argsort(areas)

    max_index = ind_sort[-1] #np.argmax(areas)
    cnt=contours[max_index]

    hull = cv.convexHull(cnt)
    hull = hull.reshape(int(hull.size/2),2)
    hull = np.append(hull,[hull[0,:]],axis=0)
    
    mask = np.zeros(mx.shape,np.uint8)
    cv.drawContours(mask,[hull],0,255,-1)
    
    rect = cv.minAreaRect(hull)
    (cX,cY),(width,height),theta = rect
    cX = int(cX)
    cY = int(cY)
    
    if rect_approx:
        s = np.mean(np.array((width,height)))
        if wafer_size == '5in':
            d = 166/125*s
        elif wafer_size == 'M2':
            d = 210/157*s
        else:
            d = 211/161.7*s

        box = cv.boxPoints(rect)
        box = np.int0(box)
        cnt = box

        mask_rect = np.zeros(mx.shape,np.uint8)
        cv.drawContours(mask_rect,[cnt],0,255,-1)

        mask_disk = np.zeros(mx.shape,np.uint8)
        mask_disk[(Xgrid-cX)**2+(Ygrid-cY)**2 <= (d/2)**2] = 255

        mask = np.minimum(mask_rect,mask_disk)

        th3 = cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,71,0)
        contours,hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        
    hull = cv.convexHull(cnt)
    hull = hull.reshape(int(hull.size/2),2)
    hull = np.append(hull,[hull[0,:]],axis=0)
    
    mask = np.zeros(mx.shape,np.uint8)
    cv.drawContours(mask,[hull],0,255,-1)
    
        
    return hull,mask//255,cX,cY,theta, mask_disk,mask_rect



"""
 find hulls for 3-cell string 
 """
directory = "C:/Users/yhe/Desktop"
images_to_read = "C:/Users/yhe/Desktop/EL/EL636.tif"
im,x,y = read_image(images_to_read)


# increase the image size 

top, bottom, left, right = [200,200,0,0]

im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0,0,0])

thresRatio_contour=0.15
thresRatio_blockContact=0.3


hull1,hull2,hull3=get_allhulls(im,thresRatio_contour)
fig=plt.figure(figsize=(10, 20))
plt.imshow(im,cmap="gray") 
plt.plot(hull1[:,0],hull1[:,1],hull2[:,0],hull2[:,1],hull3[:,0],hull3[:,1],'-r')
plt.title('3-cell Contour of a string')
plt.axis('off')
#plt.savefig(directory+"\\"+"3-cell contour.png", bbox_inches = 'tight',pad_inches = 0)

"""
single cell test for block contact 
"""

hullList=[hull1,hull2,hull3]

#for hull in hullList:

"""
Get single cropped image  
"""   
im_cropped,hull,mask=get_singleHull(im,hull2)    # change here


mask = mask.astype(float)
mask[mask==0] = np.nan
fig=plt.figure(figsize=(5, 5))
plt.imshow(mask)
plt.axis('off')
#plt.savefig(directory+"\\"+"contourmask.png", bbox_inches = 'tight',pad_inches = 0)

"""
Get the sudo contour mask
"""

hull_sudo,mask_sudo,centerx,centery,theta, mask_disk,mask_rect = find_sudo_hulls(im_cropped,True,'M4',58)
mask_total=np.sum(mask_sudo)
fig=plt.figure(figsize=(5, 5))
plt.imshow(im_cropped,cmap="gray") 
plt.plot(hull_sudo[:,0],hull_sudo[:,1],'-r')
plt.title('Cropped Image with Sudo Mask')
plt.axis('off')

"""
Detect blocking contact defect
"""
component = find_blockContact(im_cropped,thresRatio_blockContact)
component = component.astype(float)
component[component==0] = np.nan
fig=plt.figure(figsize=(5, 5))
plt.imshow(component)
plt.title('Blocking Contact Defects')
plt.axis('off')
#plt.savefig(directory+"\\"+"line defect.png", bbox_inches = 'tight',pad_inches = 0)

percentage_blockContact=(mask_total-np.sum(~np.isnan(component)))/(mask_total)*100
if (percentage_blockContact < 0 or percentage_blockContact==0):
    print("This cell doesn't have block contact")
else:
    print("The precentage of block contact of this cell is: ", percentage_blockContact)


"""
Detect Line defect 
"""
image_cropped= np.array(im_cropped.astype(np.uint8))   
im_test = cv.medianBlur(image_cropped,13)
th_defect = cv.adaptiveThreshold(im_test,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)

#th_defect = th_defect.astype(float)
th_defect[th_defect==255]=1
th_defect=th_defect.astype(np.uint8)

"""
defect_matrix = mask*th_defect
defect_matrix = defect_matrix.astype(float)
defect_matrix[defect_matrix==0] = np.nan
"""


result=component*th_defect
mask_defect = result.astype(float)
mask_defect[mask_defect==0] = np.nan
fig=plt.figure(figsize=(5, 5))
plt.imshow(mask_defect)
plt.title('Line/Scrath Defects')
plt.axis('off')

#percentage_linedefects=((np.sum(mask_sudo))-np.sum(~np.isnan(mask_defect)))/(np.sum(mask_sudo))*100
percentage_linedefects=(np.sum(~np.isnan(component))-np.sum(~np.isnan(mask_defect)))/(np.sum(mask_sudo))*100
print("The precentage of line/scratch defects of this cell is: ", percentage_linedefects)