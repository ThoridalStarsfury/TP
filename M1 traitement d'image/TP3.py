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






def getAndDrawHisto (gImg):
    histo = np.zeros(256,dtype= np.uint)
    histo_cumulation = np.zeros(256,dtype= np.uint)
    histo_egalisation = np.zeros(256,dtype= np.uint)
    Img3 = np.zeros(256,dtype= np.float64)
    Img2 = np.zeros(256,dtype= np.uint)

    rows = gImg.shape[0]
    cols = gImg.shape[1]
    greyImg2 = np.zeros((rows,cols), np.dtype('d')) #initiation de grey_tool_Img
    
    # faire des choses...
    for i in range (rows):
        for j in range (cols):
            #greyImg2[i][j] = np.dot(gImg[i][j],[0.2126,0.7152,0.0722])
            greyImg2[i][j] = np.dot(gImg[i][j],[0.2989,0.5870,0.1140])
            a = int(greyImg2[i][j])
            histo[a] += 1
            
    
    histo_cumulation[0] = histo[0]
    Img3[0] = 255/(rows*cols)*histo_cumulation[0]
    for i in range (1,256,1):
        histo_cumulation[i] = histo_cumulation[i-1] + histo[i]
        Img3[i] = 255/(rows*cols)*histo_cumulation[i]
        
    """"print(Img3)"""
    
    histo_cum = np.cumsum(histo)
    Img2 = 255/(histo_cum.max()) * histo_cum
    
    print(Img2)
    print("total pixel is",rows*cols)
    print(len(Img2))

    
    """for z in gImg.flatten():
        algo = int(Img2[(z)])
        histo_egalisation[algo] += 1"""
    
    for z in gImg.flatten():
        algo = int(Img2[(z)])
        histo_egalisation[algo] += 1
    
    for k in range(len(histo_egalisation)):
        histo_egalisation[k] = histo_egalisation[k]/3
        
    histo_eg = np.zeros_like(greyImg2)
    for i in range(greyImg2.shape[0]):
        for j in range(greyImg2.shape[1]):
            histo_eg[i,j] = Img2[int(greyImg2[i,j])]
        
    print(histo_eg.shape)
        
    
        
    print("total pixel is",rows*cols)
    print("\nhisto is ", histo)
    plt.plot(histo)
    plt.show()
    plt.figure()
    plt.plot(histo_cumulation)
    plt.show()
    plt.figure()
    plt.plot(histo_egalisation)
    plt.show()
    return histo_eg


histo_egalisation = getAndDrawHisto(rgbImg)



#%%
rgb_lenaImg = np.array(Image.open('lena.jpg'))

print(type(rgb_lenaImg))

print(rgb_lenaImg.shape)

rows = rgb_lenaImg.shape[0]
cols = rgb_lenaImg.shape[1]

grey_lena = np.zeros((rows,cols), np.dtype('d'))
print(grey_lena.shape)

for i in range(rows):
    for j in range(cols):
        grey_lena[i][j] = np.dot(rgb_lenaImg[i][j],[0.2989,0.5870,0.1140])

plt.imshow(grey_lena,cmap = 'gray')
plt.title('Grey Image')
plt.axis('off')
plt.show()


histo_lena_egalisation = getAndDrawHisto(rgb_lenaImg)
plt.figure()
plt.imshow(histo_lena_egalisation,cmap = 'gray')
plt.title('Equallized Grayscale Image')
plt.axis('off')
plt.show()


#%%


