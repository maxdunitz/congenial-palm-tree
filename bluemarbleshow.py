import time
from mayavi import mlab
from tvtk.api import tvtk # python wrappers for the C++ vtk ecosystem

def auto_sphere(image_file):
    # create a figure window (and scene)
    fig = mlab.figure(size=(600, 600))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 1
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,
                                       phi_resolution=Nrad)

    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)
    return fig, sphere_actor


if __name__ == "__main__":
    image_file = 'equirectangular.jpg' #sphere.jpg'
    fig,sa = auto_sphere(image_file)
    mlab.view(azimuth=0,elevation=90)
    for i in range(36):
        fig.scene.camera.azimuth(10)
        fig.scene.reset_zoom()
        fig.scene.save_png('anim%d.png'%i) 
    mlab.show()

