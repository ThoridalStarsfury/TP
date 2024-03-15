#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:31:14 2024

@author: zhanpeng.xi@etu.umontpellier.fr
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

rgbImg = np.array(Image.open('rainbow.jpeg'))
rawimg = Image.fromarray(rgbImg)
#rawimg.show ( )

print(type(rgbImg))
print(rgbImg.dtype) 
print(rgbImg.shape)


def saveAndShowGrey(npImg, string):
    img = Image.fromarray(npImg)
    img = img.convert("L")#mode niveau de gris
    img.save(string)
    img.show( )
    return img

def saveHsv(npImg, string):
    img = Image.fromarray(npImg)
    img = img.convert("HSV")#mode HSV
    h,s,v = img.split()
    h.save(string)
    h.show()
    return img
    

rows = rgbImg.shape[0]
cols = rgbImg.shape[1]

greyImg = np.zeros((rows,cols), np.dtype('d'))






for i in range(rows):
    for j in range(cols):
        greyImg[i][j] = np.dot(rgbImg[i][j],[0.2989,0.5870,0.1140])


#grayscale_pic = np.expand_dims(np.dot(pic_array[...,:3],[0.299, 0.587, 0.144]),axis = 0)  [...,:3]?



# faire les calculs opportuns
# pour chaque pixel de l’image
# donc dans deux boucles for


#saveAndShowGrey(rgbImg,'1− greyImg.png')
"""
rawimg2 = Image.fromarray(greyImg) 
rawimg2.show ( ) # On a la même image que saveAndShowGrey
"""

#saveHsv(rgbImg,'1− HSVImg.png')

"""-----------------------------------------------------------------"""

rgb_tool_Img = np.array(Image.open('tool.jpg'))
raw_tool_img = Image.fromarray(rgb_tool_Img)
raw_tool_img.show()


greyImg2 = np.zeros((rows,cols), np.dtype('d'))

tool_rows = rgb_tool_Img.shape[0]
tool_cols = rgb_tool_Img.shape[1]

greyImg2 = np.zeros((tool_rows,tool_cols), np.dtype('d')) #initiation de grey_tool_Img

def getAndDrawHisto (gImg):
    histo = np.zeros(256,dtype= np.uint)
    histo_cumulation = np.zeros(256,dtype= np.uint)
    histo_egalisation = np.zeros(256,dtype= np.uint)
    # faire des choses...
    for i in range (tool_rows):
        for j in range (tool_cols):
            greyImg2[i][j] = np.dot(gImg[i][j],[0.2126,0.7152,0.0722])
            a = int(greyImg2[i][j])
            histo[a] += 1
            
    
    histo_cumulation[0] = histo[0]
    histo_egalisation[0] = 255/(tool_rows*tool_cols)*histo_cumulation[0]
    for i in range (1,256,1):
        histo_cumulation[i] = histo_cumulation[i-1] + histo[i]
        histo_egalisation[i] = 255/(tool_rows*tool_cols)*histo_cumulation[i]
        
        
    
        
    print("total pixel is",tool_rows*tool_cols)
    print("\nhisto is ", histo)
    plt.plot(histo)
    plt.show()
    plt.figure()
    plt.plot(histo_cumulation)
    plt.show()
    plt.figure()
    plt.plot(histo_egalisation)
    plt.show()
    return (histo)


getAndDrawHisto(rgb_tool_Img)
