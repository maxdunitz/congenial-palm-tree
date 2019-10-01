#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Comparaison méthodes
#
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2, ifft2, fftshift
import imageio
plt.ion()

## Init
def rmse(u,u0):
    N = np.shape(u)[0]
    v0 = u0[N//8:(7*N)//8, :]
    v = u[N//8:(7*N)//8, :]
    return np.sqrt(np.mean(np.abs((v-v0)**2)))


N = int(input('N:'))
M = 3*N+1
s = int(input('s:'))
sigma = np.sqrt(N)/1.5

image=imageio.imread("PlanisphereSatellite_bw.png")[100:100+N, 450:450+M]
image=np.concatenate((image, image), axis = 1) 
F = lambda x,y: 1.0 - 2*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)

def visibiliteBruitdft2(j,s):
    T = np.copy(image[:, j:j+N]).astype(float)
    v0 = np.array([[T[k,l]*F(k,l) for l in range(N)] for k in range(N)])
    V = fft2(v0)/N   #les visibilités
    Vs = np.zeros((N,N), dtype = complex)  #visibilités sous-echantillonnées
    Noise = np.random.normal(scale = sigma/np.sqrt(2), size = (N,N)).astype(complex)
    Noise += 1j*np.random.normal(scale = sigma/np.sqrt(2), size = (N,N))      
    Noise[0,0] = 0  
    for l in range(0,N//2+1,s):
        Vs[:,l] = V[:,l] + Noise[:,l]
        Vs[:,-l] = V[:,-l] + Noise[:,-l]
    return Vs

##
"""
"""
#------------- Méthode 1
ti = time.time()

nmbRepet = 100

if s ==1:
    for i in range(nmbRepet):
        L = np.arange(N)
        visi = visibiliteBruitdft2(N,s)
        sol1 = ifft2(visi)*N
        sol1 = np.array([sol1[k,L]/F(k,L) for k in range(N)])
    
t1 = time.time() - ti
t1 = t1/nmbRepet

"""
"""
#------------- Méthode 2
# Fonction qui construit la matrice et renvoie sa pseudo-inverse
def PInv(xi,s):
    Mat = np.zeros((N,2*s-1))
    for j in range(s):
        for k in range(j*N//s, (j+1)*N//s):
            for l in range(j,j+s):
                Mat[k,l] = F(xi, N-k-1 +(l-s+1)*N//s)/s
    return np.linalg.pinv(Mat, rcond = 10**-9)

ti = time.time()
#pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse
sol2=np.zeros((N,M), dtype = complex)
listCliche = []
for j in range(M):
    listCliche.append(ifft2(visibiliteBruitdft2(j,s))*N)
    
for xi in range(0,N):
    A = PInv(xi,s)
    for eta in range(0,M):
        vecteur_cliche=np.zeros(N, dtype = complex)
        for j in range(0,N):
            vecteur_cliche[j]=listCliche[eta - (N-1) + j][xi,-j-1]
        S=np.dot(A,vecteur_cliche)
        sol2[xi,eta]=S[s-1]

t2 = time.time() - ti
t2 = t2/(M/N)

"""
"""
#------------- Méthode 3
ti = time.time()

# Création pour tout j et jHat de V et Vtilde
listV = []
for j in range(M):
    listV.append(visibiliteBruitdft2(j,s))
listV = np.array(listV)
listVtilde = list(fft(listV, axis=0))
# Tranformation matrice -> vecteur:
for jHat in range(M):
    Mat = np.array([[listVtilde[jHat][k,l] for l in range(0,N,s)] for k in range(N)])
    listVtilde[jHat] = np.ravel(Mat)
    
# Création des matrices G pour tout jHat
def g2(xi,jHat):  #comme g, mais on ne prend pas qu'un seul coef de la fft
    TF = fft(F(xi,np.arange(N))*np.exp(2j*np.pi/M*(np.arange(N)*jHat)))
    return TF/N

listG = []
for jHat in range(M):
    listG.append(np.zeros((N**2//s,N), dtype=complex))
    for xi in range(N):
        TF = g2(xi,jHat)
        for k in range(N):
            listG[jHat][k*(N//s):(k+1)*(N//s),xi] = TF[::s]*np.exp(2j*np.pi/N*(-k*xi))
listG = np.array(listG)

# Les pseudo-inverse
listGPlus = np.linalg.pinv(listG)
listTtilde = np.array([listGPlus[jHat].dot(listVtilde[jHat]) for jHat in range(M)])    
listT = ifft(listTtilde, axis = 0)
sol3 = np.transpose(listT)

t3 = time.time() - ti
t3 = t3/(M/N)

#----- Méthode 4
def PInvGlobal(xi,s):
    Mat = np.zeros((M*N//s,M))
    for j in range(M):
        for k in range(j*N//s, (j+1)*N//s):
            for l in range(k+j-j*N//s, k+j-j*N//s + N, N//s):
                Mat[k,l%M] = F(xi, l-j)/s
    return np.linalg.pinv(Mat)

solGlobale = np.zeros((N,M))
for xi in range(N):
    A = PInvGlobal(xi,s)
    vecteur_cliche=np.zeros(M*N//s)
    for j in range(M):
        for k in range(j*N//s, (j+1)*N//s):
            vecteur_cliche[k]=listCliche[j][xi,k%(N//s)]
    S = np.dot(A, vecteur_cliche)
    solGlobale[xi,:] = S

t4=time.time()-ti
t4 = t4/(M/N)

## Affichage
plt.figure(1)
plt.clf()
plt.subplot(2,2,1)
plt.imshow(image[:,N:2*N], cmap = "gray")
plt.title("Image originale")

if s == 1:
    plt.subplot(2,2,2)
    plt.imshow(np.abs(sol1), cmap = "gray")
    plt.title("Méthode 1")
    print("Méthode 1: rmse =",str(rmse(image[:,N:2*N],sol1)),"t1 =", t1)
    print("En + fair: rmse =",str(rmse(image[:,N:2*N],sol1)/np.sqrt(N)),"t1 =", t1)

plt.subplot(2,2,3)
plt.imshow(np.abs(sol2[:,N:2*N]), cmap = "gray")
plt.title("Méthode 2")
print("Méthode 2: rmse =",str(rmse(image[:,N:2*N],sol2[:,N:2*N])),"t2 =", t2)

plt.subplot(2,2,4)
plt.imshow(np.abs(sol3[:,N:2*N]), cmap = "gray")
plt.title("Méthode 3")
print("Méthode 3: rmse =",str(rmse(image[:,N:2*N],sol3[:,N:2*N])),"t3 =", t3)

print("Méthode 4: rmse =",str(rmse(image[:,N:2*N],solGlobale[:,N:2*N])),"t4 =",t4)
