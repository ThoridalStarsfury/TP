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
rawimg.show ( )

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
saveAndShowGrey(rgbImg,'1− greyImg.png')
saveHsv(rgbImg,'1− HSVImg.png')

