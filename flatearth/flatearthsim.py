#maison des idiots
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import generic_filter as gf


def F(xi, eta, N):
    ## taken from page 9 of Hugo and Clement's report
    xi_term =  (xi/N - 1.0/2 + 1.0/(2*N))**2
    eta_term = (eta/N - 1.0/2 + 1.0/(2*N))**2
    return 1-2*(xi_term + eta_term)

#def ant_pattern(xi, eta):
    

if __name__ == "__main__":

    # Geometric parameters
    R = 6371 # radius of spherical earth (km)
    h = 687 # altitude of satellite (km)
    Rorbit = R + h
    
    direction_cosines_ = [(x,0) for x in np.linspace(-0.99, 0.99, 200) if is_visible(x,0)]

    img=mpimg.imread('bluemarble1.jpg')
    Nlat, Nlong, rgb = img.shape
    pixels_per_radian = Nlat/np.pi ## pi vertically, 2pi horizontally
    assert(abs(Nlat/np.pi - Nlong/(2*np.pi)))

    print(Nlat, Nlong)

    for i in range(Nlong):
        snapshot = get_snapshot(img, i)


