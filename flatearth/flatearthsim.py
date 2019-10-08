#maison des idiots
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import generic_filter as gf



def apply_to_each_color(f, im):
    # apply f to each color channel of MxNx3 array im

    M,N,r = im.shape
    assert r==3

    # read out the channels
    im_r = np.squeeze(im[:,:,0])
    im_g = np.squeeze(im[:,:,1])
    im_b = np.squeeze(im[:,:,2])

    # apply the function to each channel
    out_r = f(im_r)
    out_g = f(im_g)
    out_b = f(im_b)

    # write the channels to the output
    out = np.zeros((M,N,r))
    out[:,:,0] = out_r
    out[:,:,1] = out_g
    out[:,:,2] = out_b

    return out

def F(xi, eta, N):
    ## taken from page 9 of Hugo and Clement's report
    xi_term =  (xi/N - 1.0/2 + 1.0/(2*N))**2
    eta_term = (eta/N - 1.0/2 + 1.0/(2*N))**2
    return 1-2*(xi_term + eta_term)

#def ant_pattern(xi, eta):

def sample(s):
    def lambda_(im):
        print("in lambda_, shape", im.shape)
        for j in range(1,s):
            print("j",j)
            im[:,j::s] = 0
        return im
    return lambda_

def get_snapshot(img,N,j,s):
    img_frame = img[:,j:j+N,:]
    grayscale = np.mean(img_frame, axis=2)
    plt.figure(1)
    plt.subplot(3,3,1)
    plt.imshow(grayscale,cmap='gray')
    plt.subplot(3,3,2)
    transform = np.fft.fft2(grayscale)
    transform[:,1::s] = 0
    plt.imshow(np.real(np.fft.ifft2(transform)).astype('uint8'),cmap='gray')
    
    image_grayrep = np.zeros(img_frame.shape)
    image_grayrep[:,:,0] = np.copy(grayscale)
    image_grayrep[:,:,1] = np.copy(grayscale)
    image_grayrep[:,:,2] = np.copy(grayscale)
    undersamp = apply_to_each_color(np.fft.fft2, image_grayrep)
    undersamp = apply_to_each_color(sample(s), undersamp)
    undersamp = apply_to_each_color(np.fft.ifft2, undersamp)
    plt.subplot(3,3,3)
    plt.imshow(np.real(undersamp).astype('uint8'))

    # undersample in fourier domain
    fft_img = apply_to_each_color(np.fft.fft2, img_frame)
    undersampled = apply_to_each_color(sample(s), fft_img)

    # invert for result
    snapshot = apply_to_each_color(np.fft.ifft2, undersampled)
    plt.subplot(3,3,4)
    plt.imshow(np.real(snapshot).astype('uint8'))
    plt.subplot(3,3,5)
    plt.imshow(np.real(snapshot[:,:,0]).astype('uint8'), cmap='gray')
    plt.subplot(3,3,6)
    plt.imshow(np.real(snapshot[:,:,1]).astype('uint8'), cmap='gray')
    

    # ignore the color axis
    img_3d = np.copy(img[:,j:j+N,:])
    fft_img = np.fft.fft2(img_3d)
    fft_img[:,1::s] = 0
    ifft = np.fft.ifft2(fft_img)
    plt.subplot(3,3,7)
    plt.imshow(np.real(ifft).astype('uint8'))
    
    plt.show()
    return ifft


if __name__ == "__main__":

    # Geometric parameters
    R = 6371 # radius of spherical earth (km)
    h = 687 # altitude of satellite (km)
    Rorbit = R + h
    s = 2
    

    img=mpimg.imread('bluemarble1.jpg')
    Nlat, Nlong, rgb = img.shape
    pixels_per_radian = Nlat/np.pi ## pi vertically, 2pi horizontally
    assert(np.isclose(Nlat/np.pi, Nlong/(2*np.pi)))

    print(Nlat, Nlong, rgb)

    # take the periodic extension of the image
    imgrep = np.zeros((Nlat, 2*Nlong, rgb))
    imgrep[:,0:Nlong,:] = img
    imgrep[:,Nlong:2*Nlong,:] = img
    for i in range(0,Nlong,100):
        snapshot = get_snapshot(imgrep, Nlat, i, s)
        assert np.allclose(snapshot*0, np.imag(snapshot))
        plt.imshow(np.real(snapshot).astype('uint8'))
        plt.show()
