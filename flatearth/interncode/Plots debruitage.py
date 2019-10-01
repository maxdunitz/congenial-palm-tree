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
from numpy.fft import fft2, ifft2
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
    image=imageio.imread("map_humidity_bw.png")[50:50+N, 900:900+M]
    image=np.concatenate((image, image), axis = 1) 
    F = lambda x,y: np.max((0,1 - 4*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)))**(3/2)
    v=1
    
    
    def clicheBruitdft2(i,s):
        u0 = np.copy(image[:, i:i+N]).astype(float)
        v0 = np.array([[u0[k,l]*F(k,l) for l in range(N)] for k in range(N)])
        vtilde = fft2(v0)/N
        vtildeS = np.zeros((N,N), dtype = complex)
        
        Noise = np.random.normal(scale = sigma/np.sqrt(2), size = (N,N)).astype(complex)
        Noise += 1j*np.random.normal(scale = sigma/np.sqrt(2), size = (N,N))
        for l in range(0,N//2+1,s):
            vtildeS[:,l] = vtilde[:,l] + Noise[:,l]
            vtildeS[:,-l] = vtilde[:,-l] + Noise[:,-l]
            
        v2 = ifft2(vtildeS)*N
        return v2
    
    
    # Fonction qui construit la matrice et renvoie sa pseudo-inverse
    def PInv(xi,s):
        Mat = np.zeros((N,2*s-1))
        for j in range(s):
            for k in range(j*N//s, (j+1)*N//s):
                for l in range(j,j+s):
                    Mat[k,l] = F(xi, N-k-1 +(l-s+1)*N//s)/s
        return np.linalg.pinv(Mat, rcond = 10**-9)
    
    #pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse
    sol=np.zeros((N,M), dtype = complex)
    listCliche = []
    for j in range(M):
        listCliche.append(clicheBruitdft2(j,s))
    
    for xi in range(0,N):
        A = PInv(xi,s)
        for eta in range(0,M):
            vecteur_cliche=np.zeros(N, dtype = complex)
            for j in range(0,N):
                vecteur_cliche[j]=listCliche[eta - (N-1) + j][xi,-j-1]
            S=np.dot(A,vecteur_cliche)
            sol[xi,eta]=S[s-1]
    
    def PInvGlobal(xi,s):
        Mat = np.zeros((M*N//s,M))
        for j in range(M):
            for k in range(j*N//s, (j+1)*N//s):
                for l in range(k+j-j*N//s, k+j-j*N//s + N, N//s):
                    Mat[k,l%M] = F(xi, l-j)/s
        return np.linalg.pinv(Mat)
    
    solGlobale = np.zeros((N,M), dtype = complex)
    for xi in range(N):
        A = PInvGlobal(xi,s)
        vecteur_cliche=np.zeros(M*N//s, dtype = complex)
        for j in range(M):
            for k in range(j*N//s, (j+1)*N//s):
                vecteur_cliche[k]=listCliche[j][xi,k%(N//s)]
        S = np.dot(A, vecteur_cliche)
        solGlobale[xi,:] = S


    print("Cas s = "+str(s), "N = " + str(N), "sigma = " + str(sigma))
    print("rmse          = "+str(rmse(image[:,:M],sol)))
    print("")
    return rmse(image[:,:M],sol)
    #return rmse(image[2:-2,:M],sol[2:-2])


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


print("\n\npente*sqrt(N) = {:5f}".format(P[0]*np.sqrt(N)))

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

##

'''
# On peut en conclure la fonction suivante pour la rmse:
#
# rmse = k * sigma*s**(2.8)/sqrt(N)
#
# Avec k une constante
#
'''

## Tableau deux entrées s et sigma: rapide
N = 24
S = np.array([2,3,4,6])
Sigma = np.arange(5,16)
k = np.zeros(S.shape)

print("N = 24: k = rmse/sigma*np.sqrt(N)/s**(2.8)")
print("Sigma      s = 2      s = 3      s = 4      s = 6")
for sigma in Sigma:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**(2.8)
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3]))



## Tableau deux entrées s et sigma: lent
N = 60
S = np.array([2,3,4,5,6])
Sigma = np.arange(5,16)
k = np.zeros(S.shape)

print("N = 60: k = rmse/sigma*np.sqrt(N)/s**(2.8)")
print("Sigma      s = 2      s = 3      s = 4      s = 5      s = 6")
for sigma in Sigma:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**(2.8)
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))


## Tableau deux entrées s et N
Tailles = np.arange(12,61,6)
S = np.array([2,3,6])
sigma = 10
k = np.zeros(S.shape)

print("sigma = 10: k = rmse/sigma*np.sqrt(N)/s**(2.8)")
print("    N      s = 2      s = 3      s = 6")
for N in Tailles:
    for i,s in enumerate(S):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**(2.8)
    print("{:5.1f}   {:4f}   {:4f}   {:4f}".format(N,k[0],k[1],k[2]))


## Tableau deux entrées sigma et N, s = 2
Tailles = np.array([12, 18, 22, 26, 30])
s = 2
Sigma = np.arange(5,16)
k = np.zeros(Tailles.shape)

print("s = 2: k = rmse/sigma*np.sqrt(N)/s**(2.8)")
print("Sigma     N = 12     N = 18     N = 22     N = 26     N = 30")
for sigma in Sigma:
    for i,N in enumerate(Tailles):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**(2.8)
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))

## Tableau deux entrées sigma et N, s = 3
Tailles = np.array([12, 18, 21, 27, 30])
s = 3
Sigma = np.arange(5,16)
k = np.zeros(Tailles.shape)

print("s = 3: k = rmse/sigma*np.sqrt(N)/s**(2.8)")
print("Sigma     N = 12     N = 18     N = 21     N = 27     N = 30")
for sigma in Sigma:
    for i,N in enumerate(Tailles):
        k[i] = debruitage(sigma,N,s)/sigma*np.sqrt(N)/s**(2.8)
    print("{:5.1f}   {:4f}   {:4f}   {:4f}   {:4f}   {:4f}".format(sigma,k[0],k[1],k[2],k[3], k[4]))


