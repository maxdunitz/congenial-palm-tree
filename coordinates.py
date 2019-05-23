import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


## helper functions ##

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


if __name__ == "__main__":
    # Geometric parameters
    Rearth = 6371 # radius of spherical earth (km)
    Alt = 755 # altitude of satellite (km)
    Rorbit = Rearth + Alt
    period = 100.2 # period of satellite orbit (minutes)
    rand_1 = np.random.randn(3)
    dir_1 = rand_1 *1.0/np.linalg.norm(rand_1)
    rand_2 = np.random.randn(3)
    dir_2 = rand_2 * 1.0/np.linalg.norm(rand_2)
    point_1 = Rorbit*dir_1
    point_2 = Rorbit*dir_2
    midpoint_dir =  np.array([dir_1[0] + dir_2[0], dir_1[1] + dir_2[1], dir_1[2] + dir_2[2]])
    midpoint_dir = midpoint_dir / np.linalg.norm(midpoint_dir)
    point_midtrace = Rearth * midpoint_dir
    other_midpoint_dir =  np.array([-dir_1[0] - dir_2[0], -dir_1[1] - dir_2[1], -dir_1[2] - dir_2[2]])
    other_midpoint_dir = other_midpoint_dir / np.linalg.norm(other_midpoint_dir)
    other_point_midtrace = Rearth * other_midpoint_dir
    toward_north = np.cross(dir_1, dir_2)
    
    xs_to_probe_deg = np.array([50, 60, 70, 80, 100, 110, 120, 130])
    xs_to_probe = xs_to_probe_deg * (2*np.pi/360.0)
    rotations_to_probe = np.pi/2.0 - xs_to_probe
    Nrot = len(rotations_to_probe)
    
    # Set figures
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)


    # Animated data generation
    def gen(period): # actual orbit in Cartesian coordinates
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            point = np.array([np.cos(t)*point_1[0] + np.sin(t)*point_2[0], np.cos(t)*point_1[1] + np.sin(t)*point_2[1], np.cos(t)*point_1[2] + np.sin(t)*point_2[2]])
            print("ORBIT", Rorbit, np.linalg.norm(point))
            yield point
            minutes += 1

    def gen2(period): # eta, xi
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = np.array([np.cos(t)*point_1[0] + np.sin(t)*point_2[0], np.cos(t)*point_1[1] + np.sin(t)*point_2[1], np.cos(t)*point_1[2] + np.sin(t)*point_2[2]])
            eta, xi = cartesian_to_eta_xi(*pt)
            yield np.array([eta, xi])
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
            pt = (Rearth/Rorbit) * np.array([np.cos(t)*point_1[0] + np.sin(t)*point_2[0], np.cos(t)*point_1[1] + np.sin(t)*point_2[1], np.cos(t)*point_1[2] + np.sin(t)*point_2[2]])
            raxis = np.cross(pt, toward_north)
            rotated_pt = rodrigues_rotation(pt, raxis, angle)
            eta, xi = cartesian_to_eta_xi(*rotated_pt)
            yield np.array([eta, xi])
            minutes += 1

    def gen6(period, angle): # trace with varying colatitude x (angle is the complement to the colatitude, ie the latitude pi/2 - x)
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            pt = (Rearth/Rorbit) * np.array([np.cos(t)*point_1[0] + np.sin(t)*point_2[0], np.cos(t)*point_1[1] + np.sin(t)*point_2[1], np.cos(t)*point_1[2] + np.sin(t)*point_2[2]])
            raxis = np.cross(pt, toward_north)
            rot = rodrigues_rotation(pt, raxis, angle)
            print(np.linalg.norm(rot), np.linalg.norm(pt), Rorbit)
            print(rot)
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
    
    # orbit echoes around sphere
    lines3d = []
    datas3d = []
    for r in rotations_to_probe:
        d = np.array(list(gen6(period, r))).T
        l, = ax.plot(d[0, 0:1], d[1, 0:1], d[2, 0:1])
        lines3d.append(l)
        datas3d.append(d)

    # orbit in xi-eta plane
    data2 = np.array(list(gen2(period))).T
    line2, = ax2.plot(data2[0,:], data2[1,:])

    # orbit echoes in xi-eta plane
    lines2d = []
    datas2d = []
    for r in rotations_to_probe:
        d = np.array(list(gen4(period, r))).T
        l, = ax.plot(d[0,:], d[1,:])
        lines2d.append(l)
        datas2d.append(d)
        
    # Plot static info
    # sphere
    u, v = np.mgrid[0:2*np.pi:960j, 0:np.pi:480j]
    x = Rearth * np.cos(u)*np.sin(v)
    y = Rearth * np.sin(u)*np.sin(v)
    z = Rearth * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.05)

    # fixed points on sphere/orbit 
    ax.scatter([0], [0], [0], color="b", s=10) #origin 
    ax.scatter([point_midtrace[0]], [point_midtrace[1]], [point_midtrace[2]], color='k', s=2) # midpoint of trace
    ax.scatter([other_point_midtrace[0]], [other_point_midtrace[1]], [other_point_midtrace[2]], color='k', s=2) # other midpoint of trace
    ax.scatter([point_1[0]], [point_1[1]], [point_1[2]], color="g", s=8) # random point 1
    ax.scatter([point_2[0]], [point_2[1]], [point_2[2]], color="g", s=8) # random point 2
    
    # unit disk in eta/xi plane
    tt = np.linspace(0, 2*np.pi, 1000)
    ax2.plot(np.cos(tt), np.sin(tt), linewidth=1, alpha = 0.33)
    
    # fixed points in eta/xi plane
    eta, xi = cartesian_to_eta_xi(*point_1) # random point 1
    ax2.scatter([eta], [xi], color='g', s=8)
    eta, xi = cartesian_to_eta_xi(*point_2) # random point 2
    ax2.scatter([eta], [xi], color='g', s=8)
    eta, xi = cartesian_to_eta_xi(*point_midtrace) # midtrace point 
    ax2.scatter([eta], [xi], color='k', s=2)
    eta, xi = cartesian_to_eta_xi(*other_point_midtrace) # midtrace point 
    ax2.scatter([eta], [xi], color='k', s=2)


    # Set the axes properties
    ax.set_xlabel('X')
    ax.set_xlim([-1.10*Rorbit, 1.10*Rorbit])
    ax.set_ylabel('Y')
    ax.set_ylim([-1.10*Rorbit, 1.10*Rorbit])
    ax.set_zlabel('Z')
    ax.set_zlim([-1.10*Rorbit, 1.10*Rorbit])

    ax2.set_xlabel('eta')
    ax2.set_ylabel('xi')


    # Animate
    ani1 = FuncAnimation(fig, update, int(period), fargs=(data, line), blit=False)
    ani_0 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[0], lines3d[0]), blit=False)
    ani_1 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[1], lines3d[1]), blit=False)
    ani_2 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[2], lines3d[2]), blit=False)
    ani_3 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[3], lines3d[3]), blit=False)
    ani_4 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[4], lines3d[4]), blit=False)
    ani_5 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[5], lines3d[5]), blit=False)
    ani_6 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[6], lines3d[6]), blit=False)
    ani_7 = FuncAnimation(fig, update3, int(period), fargs=(datas3d[7], lines3d[7]), blit=False)
    #for d, l in zip(datas3d, lines3d):
    #    ani_ = FuncAnimation(fig, update3, int(period), fargs=(d, l), blit=False)
    ani2 = FuncAnimation(fig2, update2, int(period), fargs=(data2, line2), blit=False)
    an_0 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[0], lines2d[0]), blit=False)
    an_1 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[1], lines2d[1]), blit=False)
    an_2 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[2], lines2d[2]), blit=False)
    an_3 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[3], lines2d[3]), blit=False)
    an_4 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[4], lines2d[4]), blit=False)
    an_5 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[5], lines2d[5]), blit=False)
    an_6 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[6], lines2d[6]), blit=False)
    an_7 = FuncAnimation(fig2, update4, int(period), fargs=(datas2d[7], lines2d[7]), blit=False)
    #for d, l in zip(datas2d, lines2d):
    #    ani_ = FuncAnimation(fig2, update4, int(period), fargs=(d, l), blit=False)
    #ani.save('matplot003.gif', writer='imagemagick')
    plt.show() 

