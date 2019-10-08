#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Implémentation des modèles 1 et 2 de reconstruction d'image à partir
# des visibilités
'''

import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy.fft import fft2,fftshift

## Modele 1
#Initialisation: on crée l'image v0, et les visibilités vtilde

v0 = imageio.imread("lena.png")[123:143,123:143] #taille 20*20
N = v0.shape[1]

vtilde  = np.zeros(v0.shape, dtype = complex)
for k in range(N):
    for l in range(N):
        S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m+n*l)/N) for m in range(N)]) for n in range(N)])
        vtilde[k,l] = S/N**2


# DFT inverse pour obtenir les températures de brillance, w

w = np.zeros(v0.shape, dtype = complex)
for m in range(N):
    for n in range(N):
        S = sum([sum([vtilde[k,l]*np.exp(1j*2*np.pi*(k*m+n*l)/N) for k in range(N)]) for l in range(N)])
        w[m,n] = S

w = np.real(w)


# Affichage des résultats
plt.figure(1)
plt.clf()
plt.subplot(2,2,1)
plt.imshow(v0, cmap='gray')
plt.title("Image originale")
plt.subplot(2,2,2)
plt.imshow(np.log(1+np.abs(fftshift(vtilde))), cmap = "gray")
plt.title("Visibilités")
plt.subplot(2,2,3)
plt.imshow(np.log(1+np.abs(fftshift(fft2(v0)/N**2))), cmap = "gray")
plt.title("fft normalisée")
plt.subplot(2,2,4)
plt.imshow(w, cmap = 'gray')
plt.title("Image reconstituée")
plt.suptitle("Modèle 1")
plt.show()



## Modèle 2, s = 2
s = 2 #facteur entier
#Initialisation: on crée l'image v0, et les visibilités vtilde

v0 = imageio.imread("lena.png")[123:143,123:143] #taille 20*20
N = v0.shape[1]

vtilde  = np.zeros(v0.shape, dtype = complex)
for k in range(N):
    for l in range(0,N//2+1,s): #avec un pas s
        S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m+n*l)/N) for m in range(N)]) for n in range(N)])
        vtilde[k,l] = S/(N)**2
        
        if l != 0: #on remplit en même temps vtilde[k,-l]
            S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m-n*l)/N) for m in range(N)]) for n in range(N)])
            vtilde[k,-l] = S/N**2



# DFT inverse pour obtenir les températures de brillance, w

w = np.zeros(v0.shape, dtype = complex)
for m in range(N):
    for n in range(N):
        S = sum([sum([vtilde[k,l]*np.exp(1j*2*np.pi*(k*m+n*l)/N) for k in range(N)]) for l in range(N)])
        w[m,n] = S
wImag = np.imag(w)

assert np.allclose(wImag,wImag*0)
w = np.real(w)


# Affichage des résultats
plt.figure(2)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(v0, cmap='gray')
plt.title("Image originale")
plt.subplot(1,3,2)
plt.imshow(np.log(1+np.abs(fftshift(vtilde))), cmap = "gray")
plt.title("Visibilités (partiellement remplies)")
plt.subplot(1,3,3)
plt.imshow(w, cmap = "gray")
plt.title("Image reconstituée")
plt.suptitle("Modèle 2, facteur s = " + str(s))
plt.show()


# Comparaison pour N pair, s = 2
v1 = np.zeros(v0.shape, dtype = float) #l'image théorique pour N pair, s = 2
for l in range(N//2):
    v1[:,l] = v0[:,l]/2 + v0[:,l+N//2]/2
for l in range(N//2,N):
    v1[:,l] = v0[:,l-N//2]/2 + v0[:,l]/2
    
    
# La formule theorique de "moyenne de pixels" n'est valable que pour N pair
err = np.abs(v1-w)
assert np.allclose(err, err*0)

plt.figure(3)
plt.clf()
plt.subplot(2,2,1)
plt.imshow(v0, cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(w, cmap='gray')
plt.title("w")
plt.subplot(2,2,3)
plt.imshow(v1, cmap = "gray")
plt.title("v1")
plt.subplot(2,2,4)
plt.imshow(err, cmap = "gray")
plt.title("Différences")
plt.suptitle("Différences entre theorique et w, s ="+str(s))
plt.show()


## Modèle 2, s = 3
s = 3 #facteur entier
#Initialisation: on crée l'image v0, et les visibilités vtilde

v0 = imageio.imread("lena.png")[123:144,123:144] #taille 21*21
N = v0.shape[1]

vtilde  = np.zeros(v0.shape, dtype = complex)
for k in range(N):
    for l in range(0,N//2+1,s): #avec un pas s
        S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m+n*l)/N) for m in range(N)]) for n in range(N)])
        vtilde[k,l] = S/N**2
        
        if l != 0: #on remplit en même temps vtilde[k,-l]
            S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m-n*l)/N) for m in range(N)]) for n in range(N)])
            vtilde[k,-l] = S/N**2



# DFT inverse pour obtenir les températures de brillance, w

w = np.zeros(v0.shape, dtype = complex)
for m in range(N):
    for n in range(N):
        S = sum([sum([vtilde[k,l]*np.exp(1j*2*np.pi*(k*m+n*l)/N) for k in range(N)]) for l in range(N)])
        w[m,n] = S
wImag = np.imag(w)

assert np.allclose(wImag,wImag*0)
w = np.real(w)


# Affichage des résultats
plt.figure(4)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(v0, cmap='gray')
plt.title("Image originale")
plt.subplot(1,3,2)
plt.imshow(np.log(1+np.abs(fftshift(vtilde))), cmap = "gray")
plt.title("Visibilités (partiellement remplies)")
plt.subplot(1,3,3)
plt.imshow(w, cmap = "gray")
plt.title("Image reconstituée")
plt.suptitle("Modèle 2, facteur s = " + str(s))
plt.show()


# Comparaison pour N impair, s = 3
v1 = np.zeros(v0.shape, dtype = float) #l'image théorique pour N impair, s = 3
for l in range(N//3):
    v1[:,l] = v0[:,l]/3 + v0[:,l+N//3]/3 + v0[:,l+2*N//3]/3
for l in range(N//3,2*N//3):
    v1[:,l] = v0[:,l]/3 + v0[:,l+N//3]/3 + v0[:,l-N//3]/3
for l in range(2*N//3,N):
    v1[:,l] = v0[:,l]/3 + v0[:,l-N//3]/3 + v0[:,l-2*N//3]/3
    
    
# La formule theorique de "moyenne de 3 pixels" n'est valable que pour N
# multiple de 3
err = np.abs(v1-w)
assert np.allclose(err, err*0)

plt.figure(5)
plt.clf()
plt.subplot(2,2,1)
plt.imshow(v0, cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(w, cmap='gray')
plt.title("w")
plt.subplot(2,2,3)
plt.imshow(v1, cmap = "gray")
plt.title("v1")
plt.subplot(2,2,4)
plt.imshow(err, cmap = "gray")
plt.title("Différences")
plt.suptitle("Différences entre theorique et w, s ="+str(s))
plt.show()


## Modèle 2, s quelconque (tant que s divise N)
def formuleTheorique(v0,s):
    N = v0.shape[1]
    v1 = np.zeros(v0.shape, dtype = float) #l'image théorique
    for k in range(s):
        for l in range(k*N//s, (k+1)*N//s):
            S = 0
            for i in range(-k,s-k):
                S += v0[:,l + i*N//s]/s
            v1[:,l] = S
    return v1

s = 2 #facteur entier
#Initialisation: on crée l'image v0, et les visibilités vtilde
N = 48
if False:
    v0=imageio.imread("PlanisphereSatellite_bw.png")[168:168+N, 632:632+N]
    N = v0.shape[1]

    vtilde  = np.zeros(v0.shape, dtype = complex)
    for k in range(N):
        for l in range(0,N//2+1,s): #avec un pas s
            S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m+n*l)/N) for m in range(N)]) for n in range(N)])
            vtilde[k,l] = S/N
            
            if l != 0: #on remplit en même temps vtilde[k,-l]
                S = sum([sum([v0[m,n]*np.exp(-1j*2*np.pi*(k*m-n*l)/N) for m in range(N)]) for n in range(N)])
                vtilde[k,-l] = S/N



    # DFT inverse pour obtenir les températures de brillance, w

    w = np.zeros(v0.shape, dtype = complex)
    for m in range(N):
        for n in range(N):
            S = sum([sum([vtilde[k,l]*np.exp(1j*2*np.pi*(k*m+n*l)/N) for k in range(N)]) for l in range(N)])
            w[m,n] = S/N
    wImag = np.imag(w)

    assert np.allclose(wImag,wImag*0)
    w = np.real(w)


    # Affichage des résultats
    plt.figure(6)
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(v0, cmap='gray', vmin = 25, vmax = 190)
    plt.colorbar()
    plt.title("Image originale")
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(fftshift(vtilde))), cmap = "gray")
    plt.title("Visibilités")
    plt.subplot(1,3,3)
    plt.imshow(w, cmap = "gray", vmin = 25, vmax = 190)
    plt.title("Image obtenue par TFD-1")
    plt.colorbar()
    # plt.suptitle("Modèle 2, facteur s = " + str(s))
    plt.show()


    # Comparaison
    v1 = formuleTheorique(v0,s)
        
    err = np.abs(v1-w)
    assert np.allclose(err, err*0)

    plt.figure(7)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(v0, cmap = 'gray')
    plt.subplot(2,2,2)
    plt.imshow(w, cmap='gray')
    plt.title("w")
    plt.subplot(2,2,3)
    plt.imshow(v1, cmap = "gray")
    plt.title("v1")
    plt.subplot(2,2,4)
    plt.imshow(err, cmap = "gray")
    plt.title("Différences")
    plt.suptitle("Différences entre theorique et w, s ="+str(s))
    plt.show()

    plt.figure(8)
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(v0, cmap = 'gray')
    plt.title("Image originale")
    plt.subplot(1,3,2)
    plt.imshow(w, cmap='gray')
    plt.title("Image repliée par TFD-1")
    plt.subplot(1,3,3)
    plt.imshow(v1, cmap = "gray")
    plt.title("Image théorique")
    plt.show()

    ##

    plt.figure(10)
    plt.clf()
    plt.imshow(v0, cmap = 'gray', vmin = 0, vmax = 255)

    plt.figure(11)
    plt.imshow(np.log(1+np.abs(fftshift(fft2(v0))/N)), cmap = "gray")

    plt.figure(12)
    plt.imshow(np.log(1+np.abs(fftshift(vtilde))), cmap = "gray")


    plt.figure(13)
    plt.imshow(w, cmap='gray', vmin = 0, vmax = 255)
