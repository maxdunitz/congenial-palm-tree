#maison des idiots
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


## helper functions ##

def eta_xi_to_satellite_centered_spherical(eta, xi):
    # compute theta
    theta = np.arcsin(np.sqrt(eta**2 + xi**2))
    theta = np.pi - theta
    
    # compute phi
    phi = np.arctan(eta/xi)
    if eta < 0: # northern hemisphere
        phi = phi + np.pi
    if phi < 0:
        phi = phi + 2*np.pi

    # compute rho
    rho = np.abs((R+h)*np.cos(theta) + np.sqrt(R**2 - (np.sin(theta)*(R+h))**2))
    
    return (rho, theta, phi)
    
def eta_xi_to_satellite_centered_Cartesian(eta, xi):
    zeta = np.sqrt(1-xi**2-eta**2)
    second_term = np.sqrt(R**2-(R+h)**2*(xi**2+eta**2))
    common_term = (R+h)*zeta - second_term
    x = xi * common_term
    y = eta * common_term
    z = -1.0*zeta*common_term
    return (x,y,z)

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

def eta_xi_to_x_y(eta, xi):
    rho,theta,phi = eta_xi_to_satellite_centered_spherical(eta, xi)
    xsat, ysat, zsat = eta_xi_to_satellite_centered_Cartesian(eta,xi)
    xgeo, ygeo, zgeo = (R*np.cos(np.arcsin(np.sqrt(xsat**2+ysat**2)/R)), ysat, -xsat)
    xang = np.arcsin(zgeo/R)
    yang = np.arctan(ygeo/xgeo)
    return (R*xang, R*yang)

#def eta_xi_to_x_y_lookangle(eta, xi):
    #ztr = np.sqrt(R**2 - np.sin(theta)**2 * np.sin(phi)**2 * ((R+h)*np.cos(theta) + np.sqrt(R**2-(R+h)**2*np.sin(theta)**2))**2)
    #xobs,yobs,zobs = eta_xi_to_satellite_centered_Cartesian(eta, xi)
    #CTR_SSP = np.array([0,0,R])
    #CTR_OBS = np.array([xobs,yobs,zobs+R+h])
    #CTR_TR = np.array([0,yobs,ztr])
    #xangular = np.arccos(np.dot(CTR_SSP,CTR_TR)/(R**2))
    #yangular = np.arccos(np.dot(CTR_TR,CTR_OBS)/(R**2))
    #gamma = np.arccos(np.cos(xangular)*np.cos(yangular))
    #lookangle = np.arccos((R+h)/rho * np.sin(gamma))
    #return (R*xangular, R*yangular, lookangle, rho)

def cartesian_to_eta_xi(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    xi = x / rho
    eta = y / rho
    return (eta, xi)

def spherical_to_eta_xi(rho, theta, phi):
    eta = np.sin(theta)*np.cos(phi)
    xi = np.sin(theta)*np.sin(phi)
    return (eta, xi)

def cartesian_to_spherical(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rho)
    phi = np.atan2(y,x)
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
    print(M)
    print(u.shape)
    M[:,0] = u
    M[:,1] = v
    M[:,2] = w
    return M

def get_rotation_on_basis_to_e3(u,v,w):
    return get_rotation_e3_to_new_orthonormal_basis(u,v,w).T

def is_visible(theta):
    print(R, h, "Seuil", np.pi - np.arcsin(R/(R+h)))
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

    # Set figures
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)

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


    # Animated data generation
    def gen(period): # actual orbit in Cartesian coordinates
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            point = Rorbit * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            yield point
            minutes += 1

    def gen2(period): # eta, xi
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            sat_loc_geocentric = np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            sat_loc_geocentric_after_rot = np.dot(M,sat_loc_geocentric)
            obs_point_after_rot = np.dot(M,obs_point.reshape((3,1)))
            dir_to_sat = sat_loc_geocentric_after_rot
            ## rotate OBS and SAT so that SAT is along the positive x-axis
            # get rotation angle
            rot_angle = np.arccos(dir_to_sat[0]) if dir_to_sat[1] >= 0 else 2.0*np.pi-np.arccos(dir_to_sat[0])
            rot_mat = np.array([[np.cos(rot_angle),-1.0*np.sin(rot_angle),0],[np.sin(rot_angle),np.cos(rot_angle),0],[0,0,1]])
            rotated_sat = np.dot(rot_mat,Rorbit*dir_to_sat.reshape((3,1))).reshape((3,))
            rotated_obs = np.dot(rot_mat,obs_point_after_rot.reshape((3,1))).reshape((3,))
            print(rotated_sat) #should be Rorbit,0,0
            rho,theta,phi = geocentric_Cartesian_to_satellite_centered_spherical(*rotated_obs)
            eta, xi = spherical_to_eta_xi(rho, theta, phi)
            x,y = eta_xi_to_x_y(eta, xi)
            #print("ERROR:", abs(rr-rho))
            yield np.array([eta, xi, x, y])#[eta, xi, rho, theta, phi, x, y])
            minutes += 1

    def gen3(period): # y in the eta, xi plane along the orbit
        minutes = 0
        max_ = 2*np.pi
        while minutes < period:
            yield minutes/period * max_

    def gen4(period, angle): # eta, xi along arc (0.5 radian, y(t))
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = R * np.array([np.cos(t)*dir_1[0] + np.sin(t)*dir_2[0], np.cos(t)*dir_1[1] + np.sin(t)*dir_2[1], np.cos(t)*dir_1[2] + np.sin(t)*dir_2[2]])
            raxis = np.cross(pt, toward_north)
            rotated_pt = rodrigues_rotation(pt, raxis, angle)
            eta, xi = cartesian_to_eta_xi(*rotated_pt)
            yield np.array([eta, xi])
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
        line.set_data(data[2:, :num])
        return line

    def update3(num, data, line): # for the orbit echoes (varying x)
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])
        line.set_color('k')
        line.set_alpha(0.2)

    def update4(num, data, line): # for orbit echoes in the xi-eta plane 
        line.set_data(data[:2, :num])
        line.set_color('k')
        line.set_alpha(0.2)
        return line,


    # Compute animation
    # orbit around sphere
    data = np.array(list(gen(period))).T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    
    if show_echoes:
        # orbit echoes around sphere
        lines3d = []
        datas3d = []
        for r in rotations_to_probe:
            d = np.array(list(gen6(period, r))).T
            l, = ax.plot(d[0, 0:1], d[1, 0:1], d[2, 0:1])
            lines3d.append(l)
            datas3d.append(d)

    # orbit around rotated sphere
    data7 = np.array(list(gen5(period))).T
    line7, = ax7.plot(data7[0, 0:1], data7[1, 0:1], data7[2, 0:1])

    if show_echoes:
        # orbit echoes around sphere
        lines3d_7 = []
        datas3d_7 = []
        for r in rotations_to_probe:
            d = np.array(list(gen7(period, r))).T
            print(d)
            l, = ax.plot(d[0, 0:1], d[1, 0:1], d[2, 0:1])
            lines3d_7.append(l)
            datas3d_7.append(d)


    # orbit in xi-eta plane
    data2 = np.array(list(gen2(period))).T
    #visible_mask = is_visible(data2[3,:])
    #nmbr = len(visible_mask)
    #print("MASK",sum(visible_mask)/nmbr)
    #visible_mask = visible_mask.reshape((nmbr,1))
    eta_data = data2[0,:]#.reshape((nmbr,1))
    xi_data = data2[1,:]#.reshape((nmbr,1))
    x_data = data2[2,:]
    y_data = data2[3,:]
    line2, = ax2.plot(eta_data, xi_data)
    linexy, = axxy.plot(x_data, y_data)
    #line2, = ax2.plot(np.multiply(visible_mask,eta_data), np.multiply(visible_mask,xi_data))



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
    u, v = np.mgrid[0:2*np.pi:960j, 0:np.pi:480j]
    x = R * np.cos(u)*np.sin(v)
    y = R * np.sin(u)*np.sin(v)
    z = R * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.05)
    
    # rotated sphere
    ax7.plot_wireframe(x, y, z, color="r", alpha=0.05)
    
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

    # second disk
    if second_disk:
        ax4.plot(np.cos(tt), np.sin(tt), linewidth=1, alpha = 0.33)
        etas = np.zeros(len(x[np.logical_and(z>=0, y>=0)]),)
        xis = np.zeros(len(x[np.logical_and(z>=0, y>=0)]),)
        i=0
        for xx, yy, zz in zip(x[np.logical_and(z>=0, y>=0)],y[np.logical_and(z>=0, y>=0)],z[np.logical_and(z>=0, y>=0)]):
            (eta, xi) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='r', alpha=0.05)
        etas = np.zeros(len(x[np.logical_and(z<0, y>=0)]),)
        xis = np.zeros(len(x[np.logical_and(z<0, y>=0)]),)
        i=0
        for xx, yy, zz in zip(x[np.logical_and(z<0, y>=0)],y[np.logical_and(z<0, y>=0)],z[np.logical_and(z<0, y>=0)]):
            (eta, xi) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='b', alpha=0.05)
        etas = np.zeros(len(x[ttt]),)
        xis = np.zeros(len(x[ttt]),)
        i=0
        for xx, yy, zz in zip(x[ttt],y[ttt],z[ttt]):
            (eta, xi) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='k', alpha=0.05)
        etas = np.zeros(len(x[uuu]),)
        xis = np.zeros(len(x[uuu]),)
        i=0
        for xx, yy, zz in zip(x[uuu],y[uuu],z[uuu]):
            (eta, xi) = cartesian_to_eta_xi(xx,yy,zz)
            etas[i] = eta
            xis[i] = xi
            i+=1
        ax4.scatter(etas, xis, color='m', alpha=0.05)
        etas = np.zeros(len(x[vvv]),)
        xis = np.zeros(len(x[vvv]),)
        i=0
        for xx, yy, zz in zip(x[vvv],y[vvv],z[vvv]):
            (eta, xi) = cartesian_to_eta_xi(xx,yy,zz)
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

    if second_disk:
        ax4.set_xlabel('eta')
        ax4.set_ylabel('xi')

    ax7.set_xlabel('X')
    ax7.set_xlim([-1.10*Rorbit, 1.10*Rorbit])
    ax7.set_ylabel('Y')
    ax7.set_ylim([-1.10*Rorbit, 1.10*Rorbit])
    ax7.set_zlabel('Z')
    ax7.set_zlim([-1.10*Rorbit, 1.10*Rorbit])


    # Animate
    # first sphere
    ani1 = FuncAnimation(fig, update, int(period), fargs=(data, line), blit=False)
    if show_echoes:
        ani_0 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[0], lines3d[0]), blit=False)
        ani_1 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[1], lines3d[1]), blit=False)
        ani_2 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[2], lines3d[2]), blit=False)
        ani_3 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[3], lines3d[3]), blit=False)
        ani_4 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[4], lines3d[4]), blit=False)
        ani_5 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[5], lines3d[5]), blit=False)
        ani_6 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[6], lines3d[6]), blit=False)
        ani_7 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[7], lines3d[7]), blit=False)
    
    # rotated sphere 
    a1 = FuncAnimation(fig5, update, int(period), fargs=(data7, line7), blit=False)
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

    plt.show() 

