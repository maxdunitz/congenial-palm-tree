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


    # Set figures
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)


    # Animated data generation
    def gen(period): # orbit in Cartesian coordinates
        minutes = 0
        while minutes < period:
            t = (2*np.pi)/period * minutes
            yield np.array([np.cos(t)*point_1[0] + np.sin(t)*point_2[0], np.cos(t)*point_1[1] + np.sin(t)*point_2[1], np.cos(t)*point_1[2] + np.sin(t)*point_2[2]])
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

    def gen4(period): # eta, xi along arc (1 degree, y(t))
        pass

    def gen5(period): # eta, xi along arc (-1 degree, y(t))
        pass

    # Update functions
    def update(num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    def update2(num, data, line):
        line.set_data(data[:2, :num])
        line.axes.axis([-1, 1, -1, 1])
        return line,


    # Compute animation
    data = np.array(list(gen(period))).T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    data2 = np.array(list(gen2(period))).T
    line2, = ax2.plot(data2[0,:], data2[1,:])


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
    ani = FuncAnimation(fig, update, int(period), fargs=(data, line), blit=False)
    ani2 = FuncAnimation(fig2, update2, int(period), fargs=(data2, line2), blit=False)
    
    #ani.save('matplot003.gif', writer='imagemagick')
    plt.show() 

