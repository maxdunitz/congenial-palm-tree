#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
# Stage L3
#
# Reconstruction et débruitage de l'image par pseudo-inverse
#
'''


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
import imageio
plt.ion()

## Initialisation et bruitage

N = 48
M = 3*N+1
sigma = 4.6
s = 3




image=imageio.imread("PlanisphereSatellite_bw.png")[100:100+N, 450:450+M]
image=np.concatenate((image, image), axis = 1) 
# On concatene deux fois la même image pour simuler la bande (on n'affichera 
# que la premiere moitié)
#F = lambda x,y: 1 - 2*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)
F = lambda x,y: np.max((0,1 - 4*(((x-(N-1)/2)/N)**2 + ((y-(N-1)/2)/N)**2)))**(3/2)

v=1

def snr(u,u0):
    """ snr - calcule le signal-to-noise ratio
        v = snr( u, u0 )
        u est l'image modifie'
        u0 est l'image original de reference
    """
    return -10 * np.log10( np.mean((u-u0)**2) / np.mean(u0**2) )
    
def rmse(u,u0):
    return np.sqrt(np.mean((u-u0)**2))

def noise_matrix(vis_DC, size):
    B = 20e6
    tau = 1
    sigma = vis_DC.real / np.sqrt(2*B*tau)
    sigma_per_component = np.sqrt(sigma**2 / 2)
    print(sigma_per_component)
    M = np.random.normal(loc=0.0, scale=sigma_per_component, size=size).astype(np.complex)
    M += 1j * np.random.normal(loc=0.0, scale=sigma_per_component, size=size)
    return M


## On se place avec s quelconque

#création de la fonction qui renvoie le i-ème cliché

def clicheBruitdft2(i,s): #celui qu'on utilise
    u0 = np.copy(image[:, i:i+N]).astype(float)
    v0 = np.array([[u0[k,l]*F(k,l) for l in range(N)] for k in range(N)])
    vtilde = fft2(v0)/N
    vtildeS = np.zeros((N,N), dtype = complex)
    
    Noise = np.random.normal(scale = sigma/np.sqrt(2), size = (N,N)).astype(complex)
    Noise += 1j*np.random.normal(scale = sigma/np.sqrt(2), size = (N,N))
    for l in range(0,N//2+1,s):
        vtildeS[:,l] = vtilde[:,l] + Noise[:,l]
        vtildeS[:,-l] = vtilde[:,-l] + Noise[:,-l]
        
    v2 = np.real(ifft2(vtildeS))*N
    return v2


# Fonction qui construit la matrice et renvoie sa pseudo-inverse
def PInv(xi,s):
    Mat = np.zeros((N,2*s-1))
    for j in range(s):
        for k in range(j*N//s, (j+1)*N//s):
            for l in range(j,j+s):
                Mat[k,l] = F(xi, N-k-1 +(l-s+1)*N//s)/s
    return np.linalg.pinv(Mat, rcond = 10**-10)

        

#pour chaque couple (xi,eta) on trouve T(xi,eta) par pseudo-inverse
sol=np.zeros((N,M))
listCliche = []
for j in range(M):
    listCliche.append(clicheBruitdft2(j,s))
    
for xi in range(0,N):
    A = PInv(xi,s)
    for eta in range(0,M):
        vecteur_cliche=np.zeros(N)
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

solGlobale = np.zeros((N,M))
for xi in range(N):
    A = PInvGlobal(xi,s)
    vecteur_cliche=np.zeros(M*N//s)
    for j in range(M):
        for k in range(j*N//s, (j+1)*N//s):
            vecteur_cliche[k]=listCliche[j][xi,k%(N//s)]
    S = np.dot(A, vecteur_cliche)
    solGlobale[xi,:] = S

        


plt.figure(1)
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
plt.suptitle("Cas s=" + str(s) + ", sigma=" + str(sigma))


print("Cas s = "+str(s), "N = " + str(N), "sigma = " + str(sigma))
print("snr depart = " + str(snr(image[:,:M] + np.random.normal(scale = sigma, size = (N,M)),image[:,:M])))
print("snr        = " + str(snr(sol, image[:,:M])))
print("rmse          = "+str(rmse(image[:,:M],sol)))
print("sigma/sqrt(N) = " + str(sigma/np.sqrt(N)))


plt.figure(2)
plt.clf()
plt.subplot(3,1,1)
plt.imshow(image[:,:M], cmap = "gray")
plt.title("Image originale")
plt.subplot(3,1,2)
plt.imshow(solGlobale, cmap='gray')
plt.title("Image reconstruite")
plt.subplot(3,1,3)
plt.imshow(np.abs(image[:,:M] - solGlobale), cmap = "gray")
plt.title("Différence")
plt.suptitle("Cas s=" + str(s) + ", sigma=" + str(sigma))


print("Cas s = "+str(s), "N = " + str(N), "sigma = " + str(sigma))
print("rmse          = "+str(rmse(image[:,:M],solGlobale)))

##

"""

def clicheBruitdft2SansF(i,s):
    u0 = np.copy(image[:, i:i+N]).astype(float)
    v0 = np.array([[u0[k,l] for l in range(N)] for k in range(N)])
    vtilde = fft2(v0)/N
    vtildeS = np.zeros((N,N), dtype = complex)
    
    Noise = np.random.normal(scale = sigma/np.sqrt(2), size = (N,N)).astype(complex)
    Noise += 1j*np.random.normal(scale = sigma/np.sqrt(2), size = (N,N))
    for l in range(0,N//2+1,s):
        vtildeS[:,l] = vtilde[:,l] + Noise[:,l]
        vtildeS[:,-l] = vtilde[:,-l] + Noise[:,-l]
        
    v2 = np.real(ifft2(vtildeS))*N
    return v2

uSansF = clicheBruitdft2SansF(0,2)
uAvecF = clicheBruitdft2(0,2)

minimum = np.min(uAvecF)
maximum = np.max(uSansF)

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(uSansF, cmap = 'gray', vmin = minimum, vmax = maximum)
plt.title("Sans fonction F, s=2")
plt.subplot(1,2,2)
plt.imshow(uAvecF, cmap = 'gray', vmin = minimum, vmax = maximum)
plt.title("Avec fonction F, s=2")



"""






