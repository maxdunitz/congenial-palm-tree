#maison des idiots
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import generic_filter as gf


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
    R = 6371

    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # create 3d Axes
    fig9 = plt.figure()
    ax9 = fig9.add_subplot(111, projection='3d')
    ax9.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1) # we've already pruned ourselves
    plt.axis('off')
    #make the plot more spherical
    ax9.axis('scaled')
    return (x,y,z,img,img_big,ax9)


## helper functions ##

def xi_eta_to_satellite_centered_spherical(xi, eta):
    # compute theta
    u = np.arcsin(np.sqrt(eta**2 + xi**2))
    theta = np.pi - u
    visible = is_visible_(theta)
    
    # compute phi
    if xi == 0 and eta == 0: # SSP, phi undefined
        phi = np.pi/2
    elif xi == 0: # xi = 0 --> on trace, pi/2 if in front, 3pi/2 if behind
        phi = np.pi/2 if eta > 0 else 3*np.pi/2
    elif eta == 0: #eta = 0 --> on line perpindicular to trace that passes through SSP, phi = pi if northern hemisphere (xi < 0) and 0 if southern (xi > 0)
        phi = np.pi if xi < 0 else 0
    else: # neither SSP, on trace, or along line perpendicular to trace through SSP
        phi = np.arctan(eta/xi) # well-defined, in [-pi/2, pi/2]
        if xi < 0 and eta < 0: # northern hemisphere, behind
            # computed phi is positive
            phi = np.pi + phi
        elif xi < 0 and eta > 0: # northern hemisphere, in front
            # computed phi is negative
            phi = np.pi + phi
        elif xi > 0 and eta < 0: # southern hemisphere, behind
            # computed phi is negative
            phi = 2*np.pi + phi
        elif xi > 0 and eta > 0: # southern hemisphere, in front
            # computed phi is positive and should be
            phi = phi
    
    # compute rho
    rho = -(R+h)*np.cos(theta) - np.sqrt(R**2 - (np.sin(theta)*(R+h))**2)
# (R+h)sin\theta is R when u points to horizon...when theta is bigger (u is smaller), (R+h)sin\theta is less than R...when u is 0, (R+h)sin\theta is 0
    assert( (R+h)*np.sin(theta) <= R and (R+h)*np.sin(theta) >= 0 ) #"sanity check fail: 0 <= (R+h)sin theta <= R", (R+h)*np.sin(theta) <= R, (R+h)*np.sin(theta) >= 0)
    assert( np.sqrt(h**2 + 2*R*h) >= rho and rho >= 0 ) # print("sanity check fail: 0 <= rho <= sqrt(h^2 + 2Rh)", 
    assert( np.abs(xi - np.sin(theta)*np.cos(phi)) <= 10e-4 ) # print("sanity check fail: xi = sin theta cos phi)", n
    assert( np.abs(eta - np.sin(theta)*np.sin(phi)) <= 10e-4 ) #print("sanity check fail: eta = sin theta sin phi)", np.abs(eta - np.sin(theta)*np.sin(phi)) <= 10e-4)
# (R+h) cos \theta is negative...cos \theta = - cos u
# (R+h) cos u is rho when u is at the horizon....when u is smaller, (R+h)cosu is bigger....when u is 0, (R+h)cos u is (R+h)

    return (rho, theta, phi, visible)

def xi_eta_to_geocentric_Cartesian(xi, eta):
    rho, theta, phi, visible = xi_eta_to_satellite_centered_spherical(xi, eta)
    print(rho, theta, phi, visible)
    u = np.pi - theta
    d_Px_SAT = rho*np.cos(u)
    xctr = R + h - d_Px_SAT
    sqrt_dy_sq_plus_dz_sq = rho*np.sin(u)
    if xi == 0 and eta == 0: # SSP, phi undefined
        return (R,0,0)
    elif xi == 0: # xi = 0 --> on trace, pi/2 if in front, 3pi/2 if behind
        phi_prime = 0
    elif eta == 0: #eta = 0 --> on line perpindicular to trace that passes through SSP, phi = pi if northern hemisphere (xi < 0) and 0 if southern (xi > 0)
        phi_prime = np.pi/2
    elif xi < 0 and eta < 0: # northern hemishpere, behind
        phi_prime = 3.0*np.pi/2 - phi
    elif xi < 0 and eta > 0: #northern hemisphere, in front
        phi_prime = phi - np.pi/2
    elif xi > 0 and eta < 0: #southern hemisphere, behind
        phi_prime = phi - 3.0*np.pi/2
    elif xi > 0 and eta > 0: #southern hemisphere, in front
        phi_prime = np.pi/2 - phi
    abs_dy = sqrt_dy_sq_plus_dz_sq * np.cos(phi_prime)
    abs_dz = sqrt_dy_sq_plus_dz_sq * np.sin(phi_prime)
    yctr = np.sign(eta)*abs_dy
    zctr = -np.sign(xi)*abs_dz
    assert(xctr < R)
    assert(np.abs(np.sqrt(xctr**2 + yctr**2 + zctr**2) - R) < 10e-2)
    return (xctr, yctr, zctr)


'''def eta_xi_to_satellite_centered_Cartesian(xi, eta):
    ## WRONG?? ##
    
    zeta = np.sqrt(1-xi**2-eta**2)
    second_term = np.sqrt(R**2-(R+h)**2*(xi**2+eta**2))
    common_term = (R+h)*zeta - second_term
    x = xi * common_term
    y = eta * common_term
    z = -1.0*zeta*common_term
    return (x,y,z)'''

def geocentric_Cartesian_to_satellite_centered_spherical(x,y,z):
    # moving in positive x-direction when Earth-centered Cartesian coordinates is moving in positive z-direction in satellite-centered Cartesian
    # moving in positive z-direction when Earth-centered is moving in negative x-direction in satellite-centered Cartesian
    # y direction still the same
    r = np.sqrt(y**2+z**2)
    negative_z_coord = h + R - x
    rho = np.sqrt(negative_z_coord**2 + r**2)
    u = np.arcsin(r/rho)
    theta = np.pi-u
    vec_1 = np.array([x,0,-r])
    vec_2 = np.array([x,y,z])
    phi = np.arccos(np.dot(vec_1, vec_2)/(R**2)) # for positive y (ie, Eastern hemisphere when satellite is orbiting West to East)
    if y < 0:
        phi = 2.0*np.pi - phi
    return (rho, theta, phi)


def xi_eta_to_x_y(xi, eta):
    rho,theta,phi = xi_eta_to_satellite_centered_spherical(xi, eta)
    xsat, ysat, zsat = xi_eta_to_satellite_centered_Cartesian(xi,eta)
    xgeo, ygeo, zgeo = (R*np.cos(np.arcsin(np.sqrt(xsat**2+ysat**2)/R)), ysat, -xsat)
    xang = np.arcsin(zgeo/R)
    yang = np.arctan(ygeo/xgeo)
    return (R*xang, R*yang)


def geocentric_obs_and_geocentric_sat_to_everything(dir_1, dir_2, M, minutes):
    t = (2*np.pi)/period * minutes
    sat_loc_geocentric = Rorbit * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
    obs_loc_geocentric = obs_point ## relative to an arbitrary prime meridian, not the CTR-SAT line.
    
    # Rotate satellite orbit into xy-plane
    sat_loc_geocentric = np.dot(M, sat_loc_geocentric)
    obs_loc_geocentric = np.dot(M, obs_loc_geocentric)

    # compute rho using Euclidean distance
    rho = np.sqrt(sum(x**2 for x in (sat_loc_geocentric-obs_loc_geocentric)))
            
    # compute theta using law of cosines
    # a^2 = b^2 + c^2 - 2bc cos a
    #R**2 = rho**2 + (R+h)**2 - 2*rho*(R+H)*cos u
    #2*rho*(R+h)*cos u = rho**2 + (R+h)**2 - R**2
    u = np.arccos((rho**2 + (R+h)**2 - R**2)/(2.0*rho*(R+h)))
    theta = np.pi - u

    # find gamma using law of sines, assuming the observed point is observable
    #sin u / R = sin gamma/rho
    gamma = np.arcsin(rho * np.sin(u) / R) ## assumes observed point is observable
    # check that gamma is the same as computed using dot product
    gamma_2 = np.arccos(np.dot(obs_loc_geocentric, sat_loc_geocentric)/(R*(R+h)))


    # find length CTR-P_x
    ctr_px = R * np.cos(gamma)

    # find dx
    dx = R - ctr_px

    # find sqrt(dy**2+d_z**2)
    dy_dz_hyp = R*np.sin(gamma)

    # find P_xy, knowing satellite orbit is in xy-plane
    P_xy = np.array([obs_loc_geocentric[0], obs_loc_geocentric[1], 0])

    # find d_y
    dy = obs_loc_geocentric[1] - sat_loc_geocentric[1]

    # find d_z, knowing satellite orbit is in xy-plane
    dz = obs_loc_geocentric[2]

    #compute phi
    phi_prime = np.arccos(np.abs(dy)/dy_dz_hyp)

    # if dy > 0, phi in [0, pi), else, phi in (pi,2pi)
    # if dz > 0, phi in (pi/2,3pi/2) else phi in (3pi/2,2pi) U (0,pi/2)

    if dy > 0 and dz > 0:
        phi = np.pi/2 + phi_prime
    elif dy > 0 and dz < 0:
        phi = np.pi/2 - phi_prime
    elif dy < 0 and dz > 0:
        phi = 3*np.pi/2 - phi_prime
    elif dy < 0 and dz < 0:
        phi = 3*np.pi/2 + phi_prime

    # is the observed point observable?
    visible_flag = is_visible_(theta)

    # get direction cosines
    xi, eta = spherical_to_eta_xi(rho, theta, phi)

    # compute x and y coordinates
    x_ang = np.arcsin(np.abs(dz)/R)
    y_ang = np.arcsin(np.abs(dy)/(np.sqrt(dy**2 + ctr_px**2)))

    x = np.sign(dz)*R*x_ang
    y = np.sign(dy)*R*y_ang

    return [eta, xi, x, y, rho, theta, phi, visible_flag]


def cartesian_to_eta_xi(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    xi = x / rho
    eta = y / rho
    return (xi, eta)

def spherical_to_eta_xi(rho, theta, phi):
    eta = np.sin(theta)*np.cos(phi)
    xi = np.sin(theta)*np.sin(phi)
    return (xi, eta)

def cartesian_to_spherical(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rho)
    phi = np.arctan2(y,x)
    return (rho, theta, phi)

def spherical_to_cartesian(rho, theta, phi):
    x = rho * np.sin(theta)*np.cos(phi)
    y = rho * np.sin(theta)*np.sin(phi)
    z = rho * np.cos(theta)
    return (x,y,z)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    x = rho * np.sin(theta)*np.cos(phi)
    y = rho * np.sin(theta)*np.sin(phi)
    z = rho * np.cos(theta)
    return (x,y,z)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)

def spherical_to_cartesian(rho, theta, phi):
    x = rho * np.sin(theta)*np.cos(phi)
    y = rho * np.sin(theta)*np.sin(phi)
    z = rho * np.cos(theta)
    return (x,y,z)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rodrigues_rotation(v, axis, theta):
    return np.dot(rotation_matrix(axis, theta), v)


def get_rotation_e3_to_new_orthonormal_basis(u,v,w):
    M = np.zeros((3,3))
    M[:,0] = u
    M[:,1] = v
    M[:,2] = w
    return M

def get_rotation_on_basis_to_e3(u,v,w):
    return get_rotation_e3_to_new_orthonormal_basis(u,v,w).T

def is_visible_(theta):
    return theta >= (np.pi - np.arcsin(R/(R+h)))

if __name__ == "__main__":

    # Geometric parameters
    R = 6371 # radius of spherical earth (km)
    h = 687 # altitude of satellite (km)
    Rorbit = R + h
    period = 100.2 # period of satellite orbit (minutes)
    
    # two random points at distance Rorbit from the origin in GEOCENTRIC CARTESIAN COORDINATES, plus a random point to observe
    rand_1 = np.random.randn(3)
    dir_1 = rand_1 *1.0/np.linalg.norm(rand_1)
    point_1 = Rorbit*dir_1

    rand_2 = np.random.randn(3)
    dir_2 = rand_2 * 1.0/np.linalg.norm(rand_2)
    point_2 = Rorbit*dir_2

    rand_3 = np.random.randn(3)
    dir_3 = rand_3 * 1.0/np.linalg.norm(rand_3)
    obs_point = R * dir_3

    midpoint_dir =  np.array([dir_1[0] + dir_2[0], dir_1[1] + dir_2[1], dir_1[2] + dir_2[2]])
    midpoint_dir = midpoint_dir / np.linalg.norm(midpoint_dir)
    point_midtrace = R * midpoint_dir
    other_midpoint_dir =  np.array([-dir_1[0] - dir_2[0], -dir_1[1] - dir_2[1], -dir_1[2] - dir_2[2]])
    other_midpoint_dir = other_midpoint_dir / np.linalg.norm(other_midpoint_dir)
    other_point_midtrace = R * other_midpoint_dir
    
    # get orthonormal basis for plane defined by these two points (after rotation, the y-z plane) and the rotation matrix
    toward_north = np.cross(dir_1, dir_2)
    toward_north = toward_north / np.linalg.norm(toward_north)
    dir_2 = np.cross(toward_north, dir_1) #make dir_2 orthogonal
    M = get_rotation_on_basis_to_e3(dir_1, dir_2, toward_north)

    xs_to_probe_deg = np.array([50, 60, 70, 80, 100, 110, 120, 130])
    xs_to_probe = xs_to_probe_deg * (2*np.pi/360.0)
    rotations_to_probe = np.pi/2.0 - xs_to_probe
    Nrot = len(rotations_to_probe)
    

    second_disk = False
    mapping = False
    show_midpoints = False
    show_echoes = False
    animate_trace = False

    # Set figures
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.view_init(azim=45,elev=30)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)

    figplan = plt.figure()
    img=mpimg.imread('bluemarble1.jpg')
    Nlat, Nlong, rgb = img.shape
    pixels_per_radian = Nlat/np.pi ## pi vertically, 2pi horizontally
    equator = int(Nlat/2)
    assert(np.abs(pixels_per_radian - Nlong/(2*np.pi)) < 10e-5)
    radius_in_radians = np.pi/2-np.arcsin(R/(R+h))
    radius_in_pixels = radius_in_radians * pixels_per_radian
    radius_in_pixels = int(radius_in_pixels)
    imglan = plt.imshow(img)
    
    # initialization function: plot the background of each frame
    def init():
        imglan.set_data(img)
        return [imglan]

    def unit_circle_to_arbitrary_circle(xi, eta, ppr):
        x,y,z = xi_eta_to_geocentric_Cartesian(xi, eta)
        rho, theta, phi = cartesian_to_spherical(x,y,z)
        yang = phi
        xang = np.pi/2 - theta 
        print("XANG, YANG", xang, yang)
        assert(-1 <= xi and xi <= 1)
        assert(-1 <= eta and eta <= 1)
        return (int(xang*ppr), int(yang*ppr))

    def is_visible(eta, xi):
        print("eta", eta, "xi", xi, "argument_to_arcsin", np.sqrt(eta**2+xi**2))
        theta = np.pi - np.arcsin(np.sqrt(eta**2+xi**2))
        return is_visible_(theta)
    if animate_trace:
        direction_cosines = [(x,y) for x in [0.01*p for p in range(-100,101)] for y in [0.02*q for q in range(-50,51)] if is_visible(x,y)]
        # animation function.  This is called sequentially
        def animate(i):
            long_ctr = int((i/period) * Nlong)
            kernel = np.zeros((2*radius_in_pixels+1, 2*radius_in_pixels+1))
            y,x = np.ogrid[-equator:Nlat-equator, -long_ctr:Nlong-long_ctr]
            mask = x**2 + y**2 <= radius_in_pixels**2
            im = img[:,:,:]#imglan.get_array()
            img_prime_0 = np.zeros((Nlat,Nlong))
            img_prime_0[:,:] = im[:,:,0]
            img_prime_1 = np.zeros((Nlat,Nlong))
            img_prime_1[:,:] = im[:,:,1]
            img_prime_2 = np.zeros((Nlat,Nlong))
            img_prime_2[:,:] = im[:,:,2]
            img_prime_0[mask] = 255
            img_prime_1[mask] = 255
            img_prime_2[mask] = 255
            for xi_, eta_ in direction_cosines:
                d_xi_idx, d_eta_idx = unit_circle_to_arbitrary_circle(xi_, eta_, pixels_per_radian)
                img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = img[(equator-d_xi_idx)%Nlat,(long_ctr+d_eta_idx)%Nlong,0] 
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = img[(equator-d_xi_idx)%Nlat,(long_ctr+d_eta_idx)%Nlong,1]
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = img[(equator-d_xi_idx)%Nlat,(long_ctr+d_eta_idx)%Nlong,2]
                '''img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0 
                img_prime_0[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0 
                img_prime_0[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0 
                img_prime_0[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0 
                img_prime_0[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_0[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_1[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx-2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+1)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-2)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx-1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+1)%Nlong] = 0
                img_prime_2[(equator-d_xi_idx+2)%Nlat, (long_ctr+d_eta_idx+2)%Nlong] = 0'''
                img_prime = np.zeros((Nlat,Nlong,3)).astype('uint8')
                img_prime[:,:,0] = img_prime_0.astype('uint8')
                img_prime[:,:,1] = img_prime_1.astype('uint8')
            img_prime[:,:,2] = img_prime_2.astype('uint8')
            imgo = imglan.set_array(img_prime)
            ##circular_mean = gf(data, np.mean, footprint=kernel)
            return [imgo]
            
            anima = FuncAnimation(figplan, animate, init_func=init,
                                   frames=int(period), blit=False)

            anima.save('planimation.gif', writer='imagemagick', fps=10)
            anima.save('basic_animation_1.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    direction_cosines_ = [(x,0) for x in np.linspace(-0.99, 0.99, 200) if is_visible(x,0)]
    print("direction_cosines_", direction_cosines_)
    im = np.zeros(img.shape).astype('uint8')
    img.setflags(write=1)
    im_0 = img[:,:,0].astype('uint8')
    im_1 = img[:,:,1].astype('uint8')
    im_2 = img[:,:,2].astype('uint8')
    im_0[equator-radius_in_pixels:equator+radius_in_pixels,:] = 255
    im_1[equator-radius_in_pixels:equator+radius_in_pixels,:] = 255
    im_2[equator-radius_in_pixels:equator+radius_in_pixels,:] = 255
    for xi_, eta_ in direction_cosines_:
        print("XI, ETA", xi_, eta_)
        d_xi_idx, d_eta_idx = unit_circle_to_arbitrary_circle(xi_, eta_, pixels_per_radian)
        print("PIXEL VALUE", img[equator-d_xi_idx, 0, :])
        print("RADIUS_IN_PIXELS", radius_in_pixels, "DXI", d_xi_idx)
        im_0[equator-d_xi_idx, :] = img[equator-d_xi_idx, :, 0].astype('uint8')
        im_1[equator-d_xi_idx, :] = img[equator-d_xi_idx, :, 1].astype('uint8')
        im_2[equator-d_xi_idx, :] = img[equator-d_xi_idx, :, 2].astype('uint8')
    im[:,:,0] = im_0
    im[:,:,1] = im_1
    im[:,:,2] = im_2
    plt.imshow(im)
    plt.show()

    if second_disk:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(2,1,1, projection="3d")
        ax4 = fig3.add_subplot(2,1,2)
    
    if mapping:
        fig4 = plt.figure()
        ax5 = fig4.add_subplot(2,1,1)
        ax6 = fig4.add_subplot(2,1,2)

    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1, projection="3d")

    figxy = plt.figure()
    axxy = figxy.add_subplot(1,1,1)

    figrho = plt.figure()
    axrho = figrho.add_subplot(1,1,1)

    # Animated data generation
    def gen(period): # actual orbit in Cartesian coordinates
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            point = Rorbit * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            yield point
            minutes += 1

    def gen2(period): # get all coordinates of orbit
        minutes = 0
        while minutes < period:
            yield np.array(geocentric_obs_and_geocentric_sat_to_everything(dir_1, dir_2, M, minutes))
            minutes += 1

    def gen3(period): # y in the xi, eta plane along the orbit
        minutes = 0
        max_ = 2*np.pi
        while minutes < period:
            yield minutes/period * max_

    def gen4(period, angle): # xi, eta along arc (0.5 radian, y(t))
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = R * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            raxis = np.cross(pt, toward_north)
            rotated_pt = rodrigues_rotation(pt, raxis, angle)
            xi, eta = cartesian_to_eta_xi(*rotated_pt)
            yield np.array([xi, eta])
            minutes += 1

    def gen5(period): # rotated orbit in Cartesian coordinates
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            point = Rorbit * np.dot(M, np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]]).reshape((3,1))).reshape((3,))
            yield point
            minutes += 1

    def gen6(period, angle): # trace with varying colatitude x (angle is the complement to the colatitude, ie the latitude pi/2 - x)
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = R * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            raxis = np.cross(pt, toward_north)
            rot = rodrigues_rotation(pt, raxis, angle)
            yield rot
            minutes += 1

    def gen7(period, angle): # rotated trace with varying colatitude x (angle is the complement to the colatitude, ie the latitude pi/2 - x)
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = R * np.dot(M, np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]]).reshape((3,1))).reshape((3,))
            raxis = np.cross(pt, toward_north)
            rot = rodrigues_rotation(pt, raxis, angle)
            yield rot
            minutes += 1

    # Update functions
    def update(num, data, line): # for the orbit
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    def update2(num, data, line): # for the orbit in the xi-eta plane
        line.set_data(data[:2, :num])
        line.axes.axis([-1, 1, -1, 1])
        return line,

    def updatexy(num, data, line):
        line.set_data(data[2:4, :num])
        return line,

    def updaterhotheta(num, data, line):
        line.set_data(data[4:6, :num])
        return line,

    def update3(num, data, line): # for the orbit echoes (varying x)
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])
        line.set_color('k')
        line.set_alpha(0.2)
        #return line,

    def update4(num, data, line): # for orbit echoes in the xi-eta plane 
        line.set_data(data[:2, :num])
        line.set_color('k')
        line.set_alpha(0.2)
        return line,


    # Compute animation
    # orbit around sphere
    data = np.array(list(gen(period))).T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], color='r')
    
    if show_echoes:
        # orbit echoes around sphere
        lines3d = []
        datas3d = []
        for r in rotations_to_probe:
            d = np.array(list(gen6(period, r))).T
            l, = ax.plot(d[0, 0:1], d[1, 0:1], d[2, 0:1], color='r')
            lines3d.append(l)
            datas3d.append(d)

    # orbit around rotated sphere
    data7 = np.array(list(gen5(period))).T
    line7, = ax7.plot(data7[0, 0:1], data7[1, 0:1], data7[2, 0:1], color='r')

    if show_echoes:
        # orbit echoes around sphere
        lines3d_7 = []
        datas3d_7 = []
        for r in rotations_to_probe:
            d = np.array(list(gen7(period, r))).T
            l, = ax.plot(d[0, 0:1], d[1, 0:1], d[2, 0:1])
            lines3d_7.append(l)
            datas3d_7.append(d)


    # orbit in xi-eta plane
    data2 = np.array(list(gen2(period))).T ##[eta, xi, x, y, rho, theta, phi, visible_flag]
    [nmbr,nfeat] = data2.T.shape
    visible_mask = visible_flag = data2[7,:].reshape((nmbr,))
    visible_mask = visible_mask.reshape((nmbr,1))
    eta_data = data2[0,:].reshape((nmbr,1))
    xi_data = data2[1,:].reshape((nmbr,1))
    x_data = data2[2,:].reshape((nmbr,1))
    y_data = data2[3,:].reshape((nmbr,1))
    rho_data = data2[4,:].reshape((nmbr,1))
    theta_data = data2[5,:].reshape((nmbr,1))
    phi_data = data2[6,:].reshape((nmbr,1))
    eta_data = np.multiply(eta_data,visible_mask).reshape((nmbr,))
    xi_data = np.multiply(xi_data,visible_mask).reshape((nmbr,))
    x_data = np.multiply(x_data,visible_mask).reshape((nmbr,))
    y_data = np.multiply(y_data,visible_mask).reshape((nmbr,))
    rho_data = np.multiply(rho_data,visible_mask).reshape((nmbr,))
    theta_data = np.multiply(theta_data,visible_mask).reshape((nmbr,))
    phi_data = np.multiply(phi_data,visible_mask).reshape((nmbr,))


    #eta_data = data2[0,visible_mask]
    #xi_data = data2[1,visible_mask]
    #x_data = data2[2,visible_mask]
    #y_data = data2[3,visible_mask]
    #rho_data = data2[4,visible_mask]
    #theta_data = data2[5,visible_mask]
    #phi_data = data2[6,visible_mask]

    line2, = ax2.plot(xi_data, eta_data)
    linexy, = axxy.plot(x_data, y_data)
    linerho, = axrho.plot(rho_data, theta_data)



    if show_echoes:
        # orbit echoes in xi-eta plane
        lines2d = []
        datas2d = []
        for r in rotations_to_probe:
            d = np.array(list(gen4(period, r))).T
            l, = ax2.plot(d[0,:], d[1,:])
            lines2d.append(l)
            datas2d.append(d)
            
    # Plot static info
    # sphere
    ##u, v = np.mgrid[0:2*np.pi:960j, 0:np.pi:480j]
    ##x = R * np.cos(u)*np.sin(v)
    ##y = R * np.sin(u)*np.sin(v)
    ##z = R * np.cos(v)
    
    image_file = 'bluemarble1.jpg'#'equirectangular.jpg'
    (x,y,z,img,img_big,ax9) = mpl_sphere(image_file)
    ##ax.plot_wireframe(x.T, y.T, z.T, color="r", alpha=0.05)
    print(x.shape, y.shape, z.shape, img.shape)
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1)

    # rotated sphere
    ax7.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1)
    ax7.view_init(azim=45, elev=30)
    ##ax7.plot_wireframe(x, y, z, color="r", alpha=0.05)
    
    # second sphere
    if second_disk:
        u, v = np.mgrid[0:2*np.pi:240j, 0:np.pi:120j]
        x = R * np.cos(u)*np.sin(v)
        y = R * np.sin(u)*np.sin(v)
        z = R * np.cos(v)
        ax.plot_wireframe(x, y, z, color="r", alpha=0.05)
        ax3.scatter(x[np.logical_and(z>=0, y>=0)], y[np.logical_and(z>=0, y>=0)], z[np.logical_and(z>=0, y>=0)], color='r', alpha=0.05)
        ax3.scatter(x[np.logical_and(z<0, y>=0)], y[np.logical_and(z<0, y>=0)], z[np.logical_and(z<0, y>=0)], color='b', alpha=0.05)
        ttt = np.logical_and(y < 0, np.logical_and(x>=0, z>=0))
        uuu = np.logical_and(y < 0, np.logical_and(x<0, z>=0))
        vvv = np.logical_and(y < 0, np.logical_and(x<0, z<0))
        ax3.scatter(x[ttt], y[ttt], z[ttt], color='k', alpha=0.05)
        ax3.scatter(x[uuu], y[uuu], z[uuu], color='m', alpha=0.05)
        ax3.scatter(x[vvv], y[vvv], z[vvv], color='y', alpha=0.05)
    
    # fixed points on sphere/orbit 
    ax.scatter([0], [0], [0], color="b", s=10) #origin 
    if show_midpoints:
        ax.scatter([point_midtrace[0]], [point_midtrace[1]], [point_midtrace[2]], color='k', s=2) # midpoint of trace
        ax.scatter([other_point_midtrace[0]], [other_point_midtrace[1]], [other_point_midtrace[2]], color='k', s=2) # other midpoint of trace
    ax.scatter([point_1[0]], [point_1[1]], [point_1[2]], color="g", s=8) # random point 1
    ax.scatter([point_2[0]], [point_2[1]], [point_2[2]], color="g", s=8) # random point 2
    ax.scatter([obs_point[0]], [obs_point[1]], [obs_point[2]], color='m', s=8) # observed point

    # fixed points on rotated sphere/orbit
    rotated_p1 = np.dot(M, point_1.reshape((3,1))).reshape((3,))
    rotated_p2 = np.dot(M, point_2.reshape((3,1))).reshape((3,))
    rotated_obs = np.dot(M, obs_point.reshape((3,1)).reshape((3,)))
    ax7.scatter([0], [0], [0], color="b", s=10) #origin
    ax7.scatter([rotated_p1[0]], [rotated_p1[1]], [rotated_p1[2]], color="g", s=8) # random point 1
    ax7.scatter([rotated_p2[0]], [rotated_p2[1]], [rotated_p2[2]], color="g", s=8) # random point 2
    ax7.scatter([rotated_obs[0]], [rotated_obs[1]], [rotated_obs[2]], color="m", s=8) # obs point

    # unit disk in eta/xi plane
    tt = np.linspace(0, 2*np.pi, 1000)
    ax2.plot(np.cos(tt), np.sin(tt), linewidth=1, alpha = 0.33)

    # axes in x/y plane
    tt = np.linspace(-R, R, 1000)
    axxy.plot(np.zeros((1000,)), tt)
    axxy.plot(tt, np.zeros((1000,)))

    tt = np.linspace(-R, R, 1000)
    axrho.plot(np.zeros((1000,)), tt)
    axrho.plot(tt, np.zeros((1000,)))

    # second disk
    if second_disk:
        ax4.plot(np.cos(tt), np.sin(tt), linewidth=1, alpha = 0.33)
        etas = np.zeros(len(x[np.logical_and(z>=0, y>=0)]),)
        xis = np.zeros(len(x[np.logical_and(z>=0, y>=0)]),)
        i=0
        for xx, yy, zz in zip(x[np.logical_and(z>=0, y>=0)],y[np.logical_and(z>=0, y>=0)],z[np.logical_and(z>=0, y>=0)]):
            (xi, eta) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='r', alpha=0.05)
        etas = np.zeros(len(x[np.logical_and(z<0, y>=0)]),)
        xis = np.zeros(len(x[np.logical_and(z<0, y>=0)]),)
        i=0
        for xx, yy, zz in zip(x[np.logical_and(z<0, y>=0)],y[np.logical_and(z<0, y>=0)],z[np.logical_and(z<0, y>=0)]):
            (xi, eta) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='b', alpha=0.05)
        etas = np.zeros(len(x[ttt]),)
        xis = np.zeros(len(x[ttt]),)
        i=0
        for xx, yy, zz in zip(x[ttt],y[ttt],z[ttt]):
            (xi, eta) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='k', alpha=0.05)
        etas = np.zeros(len(x[uuu]),)
        xis = np.zeros(len(x[uuu]),)
        i=0
        for xx, yy, zz in zip(x[uuu],y[uuu],z[uuu]):
            (xi, eta) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='m', alpha=0.05)
        etas = np.zeros(len(x[vvv]),)
        xis = np.zeros(len(x[vvv]),)
        i=0
        for xx, yy, zz in zip(x[vvv],y[vvv],z[vvv]):
            (xi, eta) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='y', alpha=0.05)
    
    if mapping:
        # second disk plot
        nx, ny = (50, 50)
        xx = np.linspace(-.99, .99, nx)
        yy = np.linspace(-.99, .99, ny)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.flatten()
        yy = yy.flatten()
        ax5.scatter(xx, yy, color = 'b', alpha = 0.5)

        yvals = np.arctan(np.divide(yy, xx))
        zvals = np.arccos(np.sqrt(np.power(yy, 2) + np.power(xx,2)))
        ax6.scatter(zvals, yvals, color='b', alpha=0.5)

    # Set the axes properties
    ax.set_xlabel('X')
    ax.set_xlim([-1.10*Rorbit, 1.10*Rorbit])
    ax.set_ylabel('Y')
    ax.set_ylim([-1.10*Rorbit, 1.10*Rorbit])
    ax.set_zlabel('Z')
    ax.set_zlim([-1.10*Rorbit, 1.10*Rorbit])

    ax2.set_xlabel('eta')
    ax2.set_ylabel('xi')
   
    axxy.set_xlabel('x')
    axxy.set_ylabel('y')

    axrho.set_xlabel('rho')
    axrho.set_ylabel('theta')

    if second_disk:
        ax4.set_xlabel('eta')
        ax4.set_ylabel('xi')

    a1 = FuncAnimation(fig5, update, int(period), fargs=(data7, line7), blit=False)
    a1.save('a1.gif', writer='imagemagick',fps=10)
    if show_echoes:
        a_0 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[0], lines3d_7[0]), blit=False)
        a_1 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[1], lines3d_7[1]), blit=False)
        a_2 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[2], lines3d_7[2]), blit=False)
        a_3 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[3], lines3d_7[3]), blit=False)
        a_4 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[4], lines3d_7[4]), blit=False)
        a_5 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[5], lines3d_7[5]), blit=False)
        a_6 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[6], lines3d_7[6]), blit=False)
        a_7 = FuncAnimation(fig5, update3, int(period), fargs=(datas3d_7[7], lines3d_7[7]), blit=False)

    # disk
    ani2 = FuncAnimation(fig2, update2, int(period), fargs=(data2, line2), blit=False)
    ani2.save('ani2.gif', writer='imagemagick', fps=10)
    if show_echoes:
        an_0 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[0], lines2d[0]), blit=False)
        an_1 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[1], lines2d[1]), blit=False)
        an_2 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[2], lines2d[2]), blit=False)
        an_3 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[3], lines2d[3]), blit=False)
        an_4 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[4], lines2d[4]), blit=False)
        an_5 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[5], lines2d[5]), blit=False)
        an_6 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[6], lines2d[6]), blit=False)
        an_7 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[7], lines2d[7]), blit=False)
    
    #ani.save('matplot003.gif', writer='imagemagick')
   
    # xy
    anixy = FuncAnimation(figxy, updatexy, int(period), fargs=(data2, linexy), blit=False)

    anixy.save('anixy.gif', writer='imagemagick', fps=10)

    anirho = FuncAnimation(figrho, updaterhotheta, int(period), fargs=(data2, linerho), blit=False)
    anirho.save('anirho.gif', writer='imagemagick', fps=10)

    #plt.show() 
