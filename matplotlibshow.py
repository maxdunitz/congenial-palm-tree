import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mpl_sphere(image_file):
    img_big = plt.imread(image_file)

    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img_big.shape[0])
    phi = np.linspace(0, 2*np.pi, img_big.shape[1])

    count = 180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img_big.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img_big.shape[1] - 1, count).round().astype(int)
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    img = img_big[np.ix_(theta_inds, phi_inds)]

    theta,phi = np.meshgrid(theta, phi)
    R = 1

    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1) # we've already pruned ourselves
    plt.axis('off')
    # make the plot more spherical
    ax.axis('scaled')
    return (x,y,z,img,img_big,ax)

if __name__ == "__main__":
    image_file = 'equirectangular.jpg'
    (x,y,z,img,img_big,ax) = mpl_sphere(image_file)
    for i in range(1):
        print(i)
        print(img.shape)
        ax.view_init(elev=0, azim=i*18)
        plt.savefig("movie%d.png" % i)
    #$%plt.show()
