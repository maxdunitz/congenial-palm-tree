#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Reconstruction de l'image par pseudo-inverse
#
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as imageio
from numpy.fft import fft2, ifft2, fftshift
plt.ion()



## Initialisation

# image = np.zeros((256,256))
# image[120,150] = 255

N,M=(24,170)
image=imageio.imread("lena.png")[120:120+N, 50:50+M]
image=np.concatenate((image, image), axis = 1) 
# On concatene deux fois la même image pour simuler la bande (on n'affichera 
# que la premiere moitié)
F = lambda x,y: 1/2*N**2-((x-N//2)**2 + (y-N//2)**2) #x,y entre 0 et N-1
v=1
p=M-N

def rmse(u,u0):
    return np.sqrt(np.mean((u-u0)**2))


## Cas s=2
s=2

#création de la fonction qui renvoie le i-ème cliché
def cliche2(i):
    v2=np.zeros((N,N))
    v0=image[:, i*v:i*v+N]
    
    for k in range(N):
        for l in range(N//2):        
            v2[k,l] = v0[k,l]*F(k,l)/2 + v0[k,l+N//2]*F(k,l+N//2)/2
        for l in range(N//2,N):
            v2[k,l] = v0[k,l-N//2]*F(k,l-N//2)/2 + v0[k,l]*F(k,l)/2
    return v2
#pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse

def cliche2dft(i):
    v0 = np.copy(image[:, i*v:i*v+N]).astype(float)
    for k in range(N):
        for l in range(N):
            v0[k,l] = v0[k,l]*F(k,l)
    vtilde = fft2(v0)
    vtildeS = np.zeros((N,N), dtype = complex)
    for l in range(0,N//2+1,2):
        vtildeS[:,l] = vtilde[:,l]
        vtildeS[:,-l] = vtilde[:,-l]
        
    v2 = np.real(ifft2(vtildeS))
    return v2


sol=np.zeros((N,M))
listCliche = []
for j in range(M):
    listCliche.append(cliche2dft(j))
    
for xi in range(0,N):
    # Construction de la matrice et de sa pseudo-inverse
    Mat=np.zeros((N,3))
    for j in range(0,N//2):
        Mat[j,:]=0.5*np.array([F(xi, N-j-1-N/2),F(xi, N-j-1),0])
    for j in range(N//2,N):
        Mat[j,:]=0.5*np.array([0,F(xi, N-j-1), F(xi, N-j-1+N/2)])
    A=np.linalg.pinv(Mat, rcond = 10**-6)
    #
    
    
    for eta in range(0,M):
        vecteur_cliche=np.zeros(N)
        for j in range(0,N):
            vecteur_cliche[j]=listCliche[eta - (N-1) + j][xi,-j-1]
        
        S=np.dot(A,vecteur_cliche)
        sol[xi,eta]=S[1]


plt.figure(1)
plt.clf()
plt.subplot(3,1,1)
plt.imshow(image[:, :M], cmap = "gray")
plt.title("Image originale")
plt.subplot(3,1,2)
plt.imshow(sol, cmap='gray')
plt.title("Image reconstruite")
plt.subplot(3,1,3)
plt.imshow(np.abs(image[:, :M] - sol), cmap = "gray")
plt.title("Différence")
plt.suptitle("Cas s=2")

print("Cas s="+str(s), "rmse="+str(rmse(image[:,:M],sol)))



## Cas s=3
s=3
#création de la fonction qui renvoie le i-ème cliché
def cliche3(i):
    v2=np.zeros((N,N))
    v0=image[:, i*v:i*v+N]
    for k in range(N):
        for l in range(N//3):
            v2[k,l] = v0[k,l]*F(k,l)/3 + v0[k,l+N//3]*F(k,l+N//3)/3 + v0[k,l+2*N//3]*F(k,l+2*N//3)/3
        for l in range(N//3,2*N//3):
            v2[k,l] = v0[k,l-N//3]*F(k,l-N//3)/3 + v0[k,l+N//3]*F(k,l+N//3)/3 + v0[k,l]*F(k,l)/3
        for l in range(2*N//3,N):
            v2[k,l] = v0[k,l]*F(k,l)/3 + v0[k,l-N//3]*F(k,l-N//3)/3 + v0[k,l-2*N//3]*F(k,l-2*N//3)/3
    return v2
    
def cliche3dft(i):
    v0 = np.copy(image[:, i*v:i*v+N]).astype(float)
    for k in range(N):
        for l in range(N):
            v0[k,l] = v0[k,l]*F(k,l)
    vtilde = fft2(v0)
    vtildeS = np.zeros((N,N), dtype = complex)
    for l in range(0,N//2+1,3):
        vtildeS[:,l] = vtilde[:,l]
        vtildeS[:,-l] = vtilde[:,-l]
        
    v2 = np.real(ifft2(vtildeS))
    return v2
    
#pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse


sol=np.zeros((N,M))
listCliche = []
for j in range(M):
    listCliche.append(cliche3dft(j))
    
for xi in range(0,N):
    # Construction de la matrice et de sa pseudo-inverse
    Mat=np.zeros((N,5))
    for j in range(0,N//3):
        Mat[j,:]=np.array([F(xi, N-j-1-2*N/3),F(xi, N-j-1-N/3),F(xi, N-j-1),0,0])/3
    for j in range(N//3,2*N//3):
        Mat[j,:]=np.array([0,F(xi, N-j-1-N/3),F(xi, N-j-1),F(xi, N-j-1+N/3),0])/3
    for j in range(2*N//3,N):
        Mat[j,:]=np.array([0,0,F(xi, N-j-1),F(xi, N-j-1+N/3),F(xi, N-j-1+2*N/3)])/3
    A=np.linalg.pinv(Mat, rcond = 10**-6)
    #
    
    
    for eta in range(0,M):
        vecteur_cliche=np.zeros(N)
        for j in range(0,N):
            vecteur_cliche[j]=listCliche[eta - (N-1) + j][xi,-j-1]
        
        S=np.dot(A,vecteur_cliche)
        sol[xi,eta]=S[2]


plt.figure(2)
plt.clf()
plt.subplot(3,1,1)
plt.imshow(image[:,:M], cmap = "gray")
plt.title("Image originale")
plt.subplot(3,1,2)
plt.imshow(sol, cmap='gray')
plt.title("Image reconstruite")
plt.subplot(3,1,3)
plt.imshow(np.abs(image[:,:M] - sol), cmap = "gray")
plt.title("Différence")
plt.suptitle("Cas s=3")

print("Cas s="+str(s), "rmse="+str(rmse(image[:,:M],sol)))




## Cas s quelconque

s = 4

#création de la fonction qui renvoie le i-ème cliché
def cliche(i,s):
    v2=np.zeros((N,N))
    v0=image[:, i*v:i*v+N]
    for j in range(s):
        for k in range(N):
            for l in range(j*N//s, (j+1)*N//s):
                S = 0
                for i in range(-j,s-j):
                    S += v0[k,l + i*N//s]*F(k,l + i*N//s)/s
                v2[k,l] = S
    return v2

def clichedft(i,s):
    v0 = np.copy(image[:, i*v:i*v+N]).astype(float)
    for k in range(N):
        for l in range(N):
            v0[k,l] = v0[k,l]*F(k,l)
    vtilde = fft2(v0)
    vtildeS = np.zeros((N,N), dtype = complex)
    for l in range(0,N//2+1,s):
        vtildeS[:,l] = vtilde[:,l]
        vtildeS[:,-l] = vtilde[:,-l]
        
    v2 = np.real(ifft2(vtildeS))
    return v2


# Fonction qui construit la matrice et renvoie sa pseudo-inverse
def PInv(xi,s):
    Mat = np.zeros((N,2*s-1))
    
    for j in range(s):
        for k in range(j*N//s, (j+1)*N//s):
            for l in range(j,j+s):
                Mat[k,l] = F(xi, N-k-1 +(l-s+1)*N//s)/s
    
    return np.linalg.pinv(Mat, rcond = 10**-6)

#pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse


sol=np.zeros((N,M))
listCliche = []
for j in range(M):
    listCliche.append(clichedft(j,s))
    
for xi in range(0,N):
    # Construction de la matrice et de sa pseudo-inverse
    A = PInv(xi,s)
    #
    
    
    for eta in range(0,M):
        vecteur_cliche=np.zeros(N)
        for j in range(0,N):
            vecteur_cliche[j]=listCliche[eta - (N-1) + j][xi,-j-1]
        
        S=np.dot(A,vecteur_cliche)
        sol[xi,eta]=S[s-1]


plt.figure(3)
plt.clf()
plt.subplot(3,1,1)
plt.imshow(image[:,:M], cmap = "gray")
plt.title("Image originale")
plt.subplot(3,1,2)
plt.imshow(sol, cmap='gray')
plt.title("Image reconstruite")
plt.subplot(3,1,3)
plt.imshow(np.abs(image[:,:M] - sol), cmap = "gray")
plt.title("Différence")
plt.suptitle("Cas s=" + str(s))

print("Cas s="+str(s), "rmse="+str(rmse(image[:,:M],sol)))





