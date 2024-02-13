#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:42:23 2023

@author: serroukha
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
import numpy.linalg as alg


dt = 0.001

N = 1000


s = np.zeros((2,N))
x = np.zeros(N)
v = np.zeros(N)
f = np.ones((N,1))
v[0]=10

A = np.array([[1,dt],[0, 1]])
B = np.array([[dt**2/2],[dt]])

for i in range(N-1):
    # x[i+1]= x[i]+dt*v[i] + (dt**2/2)*f[i]
    # v[i+1]= v[i]+ dt*f[i]
    s[:,i+1] =  np.dot(A,s[:,i]) + np.dot(B,f[i])
    i=i+1

t = np.linspace(0,N,N)

print(x)
plt.figure()
plt.plot(t,s[0,:],'r')
plt.title('x(k)')
plt.grid('True')
plt.show()

print(v)
plt.figure()
plt.plot(t,s[1,:],'b')
plt.title('v(k)')
plt.grid('True')
plt.show()


A = np.array([[1,dt],[0, 1]])
# Create an empty array
my_array = A
# Loop to append elements
for i in range(2,N+1):
    # Generate a new element
    new_element = alg.matrix_power(A,i)
    
    # Append the new element to the array
    my_array = np.concatenate((my_array, new_element))

# Print the resulting array
print("Matrice D :")
print(my_array)


hp = 1000

E = np.zeros((hp, hp))

# Appliquer la récurrence pour calculer les puissances successives de A * B
for i in range(hp):
    A_puissance = np.linalg.matrix_power(A, i)

    # Remplir la partie triangulaire inférieure de la matrice carrée
    for j in range(i + 1):
        E[i, j] = np.dot(A_puissance[j % 2, :], B)

print("Matrice carrée triangulaire inférieure  E :")
print(E)


K=1
Ku = K*np.eye(1000)

Kg = 1
Kgamma = Kg*np.eye(1000)


# J(y,u,k) = u_trans(Ku + E_trans*Kgamma*E)u + 2 (Dy - Gamma_d)Kgamma*E*u
# Q = Ku + E_trans*Kgamma*E
# c = Dy - Gamma_d

D = my_array
Dy = np.dot(D,s)
print("Matrice D * y")
print(Dy)

# calcul de Q

KE = np.dot(Kgamma,E)
Et = np.transpose(E)
Q = Ku + np.dot(Et,KE)
print("Q =")
print(Q)

# calcul de c

n = 2000

# Initialiser la matrice avec des zéros
Gamma_d = np.zeros((n, 1))

for i in range(n):
    Gamma_d[i, 0] = 1 - i % 2

# Afficher la matrice gamma
print("Matrice désirée :")
print(Gamma_d)

c = Dy - Gamma_d
ct = np.transpose(c)
ce = c@KE
# calcul de u

Qinv = alg.inv(Q)
print("Q-1 :", Qinv)

u = -1*np.dot(ce,Qinv)

# Calcul de la nouvelle trajetoire avec la commande u

mu = u[:, :1]

y = np.zeros((2,N))

for i in range(N-1):
    y[:,i+1] =  np.dot(A,y[:,i]) + np.dot(B,mu[i])
    i=i+1
    
    
print(x)
plt.figure()
plt.plot(t,y[0,:],'r')
plt.title('x(k)')
plt.grid('True')
plt.show()

print(v)
plt.figure()
plt.plot(t,y[1,:],'b')
plt.title('v(k)')
plt.grid('True')
plt.show()
