#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Reconstruction et débruitage de l'image par pseudo-inverse
#
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2, ifft2, fftshift
import scipy.misc as imageio
plt.ion()

## Initialisation
ti = time.time()
N,M = (48,3*48+1)
image=imageio.imread("PlanisphereSatellite_bw.png")[100:100+N, 450:450+M]
image=np.concatenate((image, image), axis = 1) 
# On concatene deux fois la même image pour simuler la bande (on n'affichera 
# que la premiere moitié)

F = lambda x,y: np.max((0,1 - 4*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)))**(3/2)

def rmse(u,u0):
    return np.sqrt(np.mean((np.abs(u-u0))**2))


## Avec s quelconque
sigma = 4.6
s = 2

#création de la fonction qui renvoie la j-ème visibilité
def visibiliteBruitdft2(j,s):
    T = np.copy(image[:, j:j+N]).astype(float)
    v0 = np.array([[T[k,l]*F(k,l) for l in range(N)] for k in range(N)])
    V = fft2(v0)/N   #les visibilités
    Vs = np.zeros((N,N), dtype = complex)  #visibilités sous-echantillonnées
    
    Noise = np.random.normal(scale = sigma/np.sqrt(2), size = (N,N)).astype(complex)
    Noise += 1j*np.random.normal(scale = sigma/np.sqrt(2), size = (N,N))
    
    for l in range(0,N//2+1,s):
        Vs[:,l] = V[:,l] + Noise[:,l]
        Vs[:,-l] = V[:,-l] + Noise[:,-l]

    return Vs



# Création pour tout j et jHat de V et Vtilde
listV = []
for j in range(M):
    listV.append(visibiliteBruitdft2(j,s))
listV = np.array(listV)
listVtilde = list(fft(listV, axis=0))


#listV.shape, listVtilde.shape => (M, N, N)
#listV[0] et listeVtilde[0] on bien N*N/s coefficients non nuls

# Tranformation matrice -> vecteur:
for jHat in range(M):
    Mat = np.array([[listVtilde[jHat][k,l] for l in range(0,N,s)] for k in range(N)])
    listVtilde[jHat] = np.ravel(Mat)


## Création des matrices G pour tout jHat
def g(xi, jHat, I):  
    k,l = I//(N//s), I%(N//s)*s 
    S = fft(F(xi,np.arange(N))*np.exp(2j*np.pi/M*(np.arange(N)*jHat)))[l]

    return S*np.exp(2j*np.pi/N*(-k*xi))


def g2(xi,jHat):  #comme g, mais on ne prend pas qu'un seul coef de la fft
    l= []
    for n in range(N):
        l.append(F(xi,n)*np.exp(2j*np.pi/M*(n*jHat)))
    TF = fft(np.array(l))
    return TF/N


# listG = np.array([[[g(xi,jHat,I) for xi in range(N)] for I in range(N**2//s)] for jHat in range(M)])

listG = []
for jHat in range(M):
    listG.append(np.zeros((N**2//s,N), dtype=complex))
    for xi in range(N):
        TF = g2(xi,jHat)
        for k in range(N):
            listG[jHat][k*(N//s):(k+1)*(N//s),xi] = TF[::s]*np.exp(2j*np.pi/N*(-k*xi))

listG = np.array(listG)

tEcoule = time.time() - ti
print(tEcoule)
ti = time.time()
## Les pseudo-inverse
listGPlus = np.linalg.pinv(listG)

listTtilde = np.array([listGPlus[jHat].dot(listVtilde[jHat]) for jHat in range(M)])

listT = ifft(listTtilde, axis = 0)
# assert np.allclose(0,np.imag(listT))  #si sigma !=0, ce ne sera pas vrai
listT = np.transpose(listT)

tEcoule = time.time() - ti
print(tEcoule)
##
plt.figure(1)
plt.clf()
plt.subplot(3,1,1)
plt.imshow(image[:,:M], cmap = "gray", vmin = 0, vmax = 255)
plt.title("Image originale")
plt.subplot(3,1,2)
plt.imshow(np.real(listT), cmap = "gray", vmin = 0, vmax = 255)
plt.title("Image reconstruite")
plt.subplot(3,1,3)
plt.imshow(np.abs(image[:,:M]-listT), cmap = "gray")
plt.title("Différence")
plt.suptitle("Reconstruction par transformée de Fourier partielle")

print("RMSE =",rmse(image[:,:M],listT))




