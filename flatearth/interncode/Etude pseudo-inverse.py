#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Plots
#
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2, ifft2, fftshift
import scipy.misc as imageio
plt.ion()

def rmse(u,u0):
    return np.sqrt(np.mean((u-u0)**2))


def debruitage(M,N,s,matricesOnly = False):
    image=imageio.imread("map_humidity_bw.png")[50:50+N, 900:900+M]
    image=np.concatenate((image, image), axis = 1) 
    F = lambda x,y: 1 - 2*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)
    
    # Ce qui prend du temps: création des matrices G pour tout jHat
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
    
    if matricesOnly:
        return listG

    
    listFinal = np.linalg.eig([np.linalg.inv((listG[jHat].T).dot(listG[jHat])) for jHat in range(M)])

    #affichage
    print("Cas s = "+str(s), "N = " + str(N))
    print("")
    return listFinal


## Dépendance en N
s = 5
M = 107
Tailles = np.arange(15,51,5)
eigV = [debruitage(M,N,s)[0] for N in Tailles]
rapports = np.array([np.max([np.max(np.abs(eigV[i][j]))/np.min(np.abs(eigV[i][j])) for j in range(M)]) for i in range(len(Tailles))])
print(rapports)


P = np.polyfit(Tailles,rapports,1)
plt.figure(1)
plt.clf()
plt.plot(Tailles, rapports, 'o')
plt.plot(Tailles,P[1]+Tailles*P[0], label = "Regression; pente = {:8f}".format(P[0]))
plt.legend()
plt.title("rapports en fonction de N; s = " +str(s))


## Dépendance en s
M = 107
N = 48
S = np.array([2,3,4,6])
eigV = [debruitage(M,N,s)[0] for s in S]
rapports = np.array([np.max([np.max(np.abs(eigV[i][j]))/np.min(np.abs(eigV[i][j])) for j in range(M)]) for i in range(len(S))])
print(rapports)


P = np.polyfit(np.log10(S),np.log10(rapports),1)
plt.figure(2)
plt.clf()
plt.loglog(S, rapports, 'o')
plt.loglog(S,10**P[1]*S**P[0], label = "Regression; pente = {:8f}".format(P[0]))
plt.legend()
plt.title("rapports en fonction de s; N = " +str(N))

## Affichage des vp pour N,M,s donné
s = 4
N = 40
M = 80
eigV = debruitage(M,N,s)[0]
I = np.arange(len(eigV))

rd = lambda x: np.random.rand()*x


plt.figure(3)
plt.clf()
for i in I:
    if np.all(np.abs(eigV[i]) < 1000):
        plt.plot(np.real(eigV[i])+rd(0),np.imag(eigV[i])+rd(0),'o', label = str(i))
plt.legend()


#eigV[n] ~= eigV[n + M*s/N]

# eigV[0,0] = eigV[0,1]
eigVUnique = np.array([np.unique(eigV[i]) for i in I])
print(eigVUnique.shape)

##
s = 4
N = 40
M = 80
i = 13
eigV = np.sort(np.abs(debruitage(M,N,s)[0][i]))


plt.figure(2)
plt.clf()
plt.plot(eigV,'o')
#plt.hist(eigV, bins = 150, label = "Valeurs propres (val. absolues)")
plt.title("Histogramme de vp de chaque matrice, i=" + str(i))
plt.legend()

## Histogramme des vp
s = 2
N = 22
M = 45
eigV = np.abs(debruitage(M,N,s)[0])

valMax = np.array([np.max(eig) for eig in eigV])
eigV = np.ravel(eigV)

plt.figure(2)
plt.clf()
plt.hist(valMax, bins = 150, label = "Valeurs propres (val. absolues)")
plt.title("Histogramme des vp max de chaque matrice")
plt.legend()

## Histogramme des rapports
s = 3
N = 30
M = 51
eigV = np.abs(debruitage(M,N,s)[0])

valMax = np.array([np.max(eig) for eig in eigV])
rapports = np.array([np.max(eigV[j]) / np.min(eigV[j]) for j in range(M)])
eigV = np.ravel(eigV)

plt.figure(2)
plt.clf()
plt.hist(rapports, bins = 90, label = "Valeurs propres (val. absolues)")
plt.title("Histogramme des rapports max de chaque matrice")
plt.legend()


## Animation
s = 4
N = 24
M = 51

Tailles = np.arange(16,52,4)
S = [1,2,3,4,6,8]


plt.figure(2)
for s in S:
# for N in Tailles:
    eigV = np.abs(debruitage(M,N,s)[0])
    valMax = np.array([np.max(eig) for eig in eigV])
    rapports = np.array([np.max(eigV[j]) / np.min(eigV[j]) for j in range(M)])
    eigV = np.ravel(eigV)
    
    plt.clf()
    plt.hist(rapports, bins = s*20, label = "Valeurs propres (val. absolues)")
    plt.title("Rapports max de chaque matrice, s="+str(s))
    # plt.title("Rapports max de chaque matrice, N="+str(N))
    plt.legend()
    plt.pause(1)










## Matrices égales ?
s = 2
N = 24
M = 240
listeG = debruitage(M,N,s,matricesOnly=True)






