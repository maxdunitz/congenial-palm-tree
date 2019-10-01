#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Plots
#
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2, ifft2, fftshift
import imageio
plt.ion()

def rmse(u,u0):
    N = np.shape(u)[0]
    v0 = u0[N//8:(7*N)//8, :]
    v = u[N//8:(7*N)//8, :]
    return np.sqrt(np.mean(np.abs((v-v0)**2)))

    
def snr(u,u0):
    """ snr - calcule le signal-to-noise ratio
        v = snr( u, u0 )
        u est l'image modifie'
        u0 est l'image original de reference
    """
    return -10 * np.log10( np.mean((u-u0)**2) / np.mean(u0**2) )

def debruitage(sigma,N,s):
    M = 101
    image=imageio.imread("PlanisphereSatellite_bw.png")[100:100+N, 450:450+M]
    image=np.concatenate((image, image), axis = 1) 
    F = lambda x,y: np.max((0,1 - 4*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)))**(3/2)
    
    
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
    
    # Tranformation matrice -> vecteur:
    for jHat in range(M):
        Mat = np.array([[listVtilde[jHat][k,l] for l in range(0,N,s)] for k in range(N)])
        listVtilde[jHat] = np.ravel(Mat)
    
    #
    ti = time.time()
    #
    
    # Ce qui prend du temps: création des matrices G pour tout jHat
    def g2(xi,jHat):  #comme g, mais on ne prend pas qu'un seul coef de la fft
        l= []
        for n in range(N):
            l.append(F(xi,n)*np.exp(2j*np.pi/M*(n*jHat)))
        TF = fft(np.array(l))
        return TF/N
    
    listG = []
    for jHat in range(M):
        listG.append(np.zeros((N**2//s,N), dtype=complex))
        for xi in range(N):
            TF = g2(xi,jHat)
            for k in range(N):
                listG[jHat][k*(N//s):(k+1)*(N//s),xi] = TF[::s]*np.exp(2j*np.pi/N*(-k*xi))
    
    listG = np.array(listG)
    
    #
    tEcoule1 = time.time() - ti
    ti = time.time()
    #
    
    # Ce qui prend un peu moins de temps: les pseudo-inverse
    listGPlus = np.linalg.pinv(listG, rcond = 10*-15)
    listTtilde = np.array([listGPlus[jHat].dot(listVtilde[jHat]) for jHat in range(M)])    
    listT = ifft(listTtilde, axis = 0)
    listT = np.transpose(listT)
    
    
    #
    tEcoule2 = time.time() - ti
    #
    
    #affichage
    print("Cas s = "+str(s), "N = " + str(N), "sigma = " + str(sigma))
    print("rmse          = "+str(rmse(image[:,:M],listT)))
    print("")
    return rmse(image[:,:M],listT)
    return tEcoule1,tEcoule2


## Dépendance en sigma
s = 2
N = 24
Sigma = np.linspace(1,20,20)
RMSE = np.array([debruitage(sigma,N,s) for sigma in Sigma])
P = np.polyfit(Sigma,RMSE,1)
plt.figure(1)
plt.clf()
plt.plot(Sigma, RMSE, 'o')
plt.plot(Sigma,P[1]+Sigma*P[0], label = "Regression; pente = {:5f}".format(P[0]))
plt.legend()
plt.title("rmse en fonction de sigma; N = " + str(N) + " s = " +str(s))


print("\n\nDépendance en sigma linéaire, pente = {:5f}".format(P[0]))

## Dépendance en N
s = 2
sigma = 10
Tailles = np.arange(18,39,2)
RMSE = np.array([debruitage(sigma,N,s) for N in Tailles])
P = np.polyfit(np.log10(Tailles),np.log10(RMSE),1)
plt.figure(2)
plt.clf()
plt.loglog(Tailles, RMSE, 'o')
plt.loglog(Tailles,10**P[1]*Tailles**P[0], label = "Regression; pente = {:8f}".format(P[0]))
plt.legend()
plt.title("rmse en fonction de N; sigma = " + str(sigma) + " s = " +str(s))

print("\n\nDépendance en N en puissance {:5f}".format(P[0]))


## Dépendance en s
sigma = 10
N = 24
S = np.array([1,2,3,4])
RMSE = np.array([debruitage(sigma,N,s) for s in S])
P = np.polyfit(np.log10(S),np.log10(RMSE),1)
plt.figure(3)
plt.clf()
plt.loglog(S, RMSE, 'o')
plt.loglog(S,10**P[1]*S**P[0], label = "Regression; pente = {:8f}".format(P[0]))
plt.legend()
plt.title("rmse en fonction de s; N = " + str(N) + " sigma = " +str(sigma))

print("\n\nDépendance en s en puissance {:5f}".format(P[0]))

##

'''
# On peut en conclure la fonction suivante pour la rmse:
#
# rmse = k * sigma*s**2/sqrt(N)
#
# Avec k une constante
#
'''

## Tableau deux entrées s et sigma: rapide
"""
N = 24
S = np.array([2,3,4,6])
Sigma = np.arange(5,16)
k = np.zeros(S.shape)

print("N = 24: k = rmse/sigma*np.sqrt(N)/s**2")
print("Sigma      s = 2      s = 3      s = 4      s = 6")
for sigma in Sigma:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**2
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3]))
"""


## Tableau deux entrées s et sigma: lent
N = 60
S = np.array([2,3,4,5,6])
Sigma = np.arange(5,16)
k = np.zeros(S.shape)

print("N = 60: k = rmse/sigma*np.sqrt(N)/s**2")
print("Sigma      s = 2      s = 3      s = 4      s = 5      s = 6")
for sigma in Sigma:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**2
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))


## Tableau deux entrées s et N
Tailles = np.arange(12,61,6)
S = np.array([2,3,6])
sigma = 10
k = np.zeros(S.shape)

print("sigma = 10: k = rmse/sigma*np.sqrt(N)/s**2")
print("    N      s = 2      s = 3      s = 6")
for N in Tailles:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**2
    print("{:5.1f}   {:4f}   {:4f}   {:4f}".format(N,k[0],k[1],k[2]))


## Tableau deux entrées sigma et N, s = 2
Tailles = np.array([12, 18, 22, 26, 30])
s = 2
Sigma = np.arange(5,16)
k = np.zeros(Tailles.shape)

print("s = 2: k = rmse/sigma*np.sqrt(N)/s**2")
print("Sigma     N = 12     N = 18     N = 22     N = 26     N = 30")
for sigma in Sigma:
    for i,N in enumerate(Tailles):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**2
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))

## Tableau deux entrées sigma et N, s = 3
Tailles = np.array([12, 18, 21, 27, 30])
s = 3
Sigma = np.arange(5,16)
k = np.zeros(Tailles.shape)

print("s = 3: k = rmse/sigma*np.sqrt(N)/s**2")
print("Sigma     N = 12     N = 18     N = 21     N = 27     N = 30")
for sigma in Sigma:
    for i,N in enumerate(Tailles):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**2
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))

## Complexite en temps (on change la fonction debruitage)
"""
s = 2
sigma = 10
Tailles = np.arange(18,83,2)
T = np.array([debruitage(sigma,N,s) for N in Tailles])
T1 = T[:,0]
T2 = T[:,1]
P1 = np.polyfit(np.log10(Tailles),np.log10(T1),1)
P2 = np.polyfit(np.log10(Tailles),np.log10(T2),1)

plt.figure(1)
plt.clf()
plt.loglog(Tailles, T1, 'o', label = "t1")
plt.loglog(Tailles,10**P1[1]*Tailles**P1[0], label = "Regression t1; pente = {:8f}".format(P1[0]))

plt.loglog(Tailles, T2, 'o', label = "t2")
plt.loglog(Tailles,10**P2[1]*Tailles**P2[0], label = "Regression t2; pente = {:8f}".format(P2[0]))
plt.legend()
plt.title("t1,t2 en fonction de N; sigma = " + str(sigma) + " s = " +str(s))
"""



print(debruitage(10,24,6))


