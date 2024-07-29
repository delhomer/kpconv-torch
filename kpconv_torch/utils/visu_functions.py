"""
Visualization functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, C0103, E0401

from mayavi import mlab
import numpy as np


def show_modelnet_models(all_points):
    """
    docstring to do
    """

    # Interactive visualization
    # Create figure for features
    fig1 = mlab.figure("Models", bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = all_points[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        mlab.points3d(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            points[:, 2],
            scale_factor=3.0,
            scale_mode="none",
            figure=fig1,
        )

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = "<--- (press g for previous)" + 50 * " " + "(press h for next) --->"
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ["g", "G"]:

            file_i = (file_i - 1) % len(all_points)
            update_scene()

        elif vtk_obj.GetKeyCode() in ["h", "H"]:

            file_i = (file_i + 1) % len(all_points)
            update_scene()

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer("KeyPressEvent", keyboard_callback)
    mlab.show()
