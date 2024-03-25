# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:29:17 2023

@author: DELL
"""



import control as crtl
from scipy import signal
import numpy.linalg as alg

import numpy as np
import os
from control.matlab import *
import matplotlib.pyplot as plt  # MATLAB plotting functions

deltat = 0.02


A = np.array([[1,deltat],[0,1]])

B = np.array([[deltat**2/2],[deltat]])

hp = 5000

h = 2500

hc = 2500 #horizon de contrôle

"""------------------------------------"""
F = A
matriceD = np.array([[0.]*2 for i in range(hp*2)]) #initialiser la matrice D
for m in range(0,hp*2,2):
    matriceD[m,:] = F[0,:] # On entre des valeurs de m-ième ligne 
    matriceD[m+1,:] = F[1,:] # On entre des valeurs de m+1-ième ligne 
    F = F@A
"""------------------------------------"""

    
matriceE = np.array([[0.]*hc for i in range(hp*2)]) #initialiser la matrice E
D = B
for a in range(0,hp*2,2):# a-ième ligne
    if a < hc*2:
        C = B # remettre à zero ou réinitialiser la matrice C
        for b in range(a//2,-1,-1):# b-ième colonne et b <= a (a//2 est un nombre entier)
            matriceE[a,b] = C[0,:] # On entre des valeurs de [a,b] élement
            matriceE[a+1,b] = C[1,:] # On entre des valeurs de [a,b] élement
            C = A@C # l'itération
    else:
        D = A@D #l'itération de la dernière ligne de la matrice E
        E = D # RAZ ou réinitialiser la matrice E
        for b in range(hc-1,-1,-1):
            matriceE[a,b] = E[0,:] # On entre des valeurs de [a,b] élement
            matriceE[a+1,b] = E[1,:] # On entre des valeurs de [a,b] élement
            E = A@E # l'itération

matriceE_trans = matriceE.transpose()

kphi = 1
matrice_kphi = kphi*np.identity(hc)  

kj = 1
matrice_kj = kj*np.identity(hp*2)         

matriceSd = np.array([[0.] for i in range(hp*2)])
for a in range(0,hp*2,2):
    if a < hp:
        matriceSd[a] = 1
    else:
        matriceSd[a] = 10

t = np.zeros(int(hp))
t[0] = 0
for i in range(int(hp-1)):
    t[i+1] = t[i]+1
    
my_array = A

matricesk           = np.array([[1.0],[0.0]])
Dy                  = np.array([[0.] for i in range(hp*2)])
Dy                  = np.matmul(matriceD,matricesk) 
      
Q = matrice_kphi + matriceE_trans@matrice_kj@matriceE

middlefonction1 = Dy-matriceSd

middlefonction2 = middlefonction1.T

middlefonction3 = middlefonction2@matrice_kj@matriceE

C = middlefonction3.T

u = -np.linalg.inv(Q)@C

Eu = matriceE@u

desired_s = Dy + Eu
s = np.zeros(hp)
for a in range (0,hp*2,2):
    s[int(a/2)] = desired_s[a]
    
sprime = np.zeros(hp)
for a in range (0,hp*2,2):
    sprime[int(a/2)] = matriceSd[a]
    
plt.plot(t,s)
plt.plot(t,sprime)
