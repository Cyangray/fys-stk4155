from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_3d(x, y, z, an_x, an_y, an_z, plot_type):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #surf = ax.plot_surface(x_array, y_array, z_tilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    print("analytical", np.shape(an_x), np.shape(an_y), np.shape(an_z))
    print("num.", np.shape(x), np.shape(y), np.shape(z))
    surf = ax.plot_surface(an_x[0], an_y[0], an_z[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)


    for i in range(len(z)):
        surf_2 = ax.scatter(x[i], y[i], z[i])
        """if plot_type[i] == "surface":
            surf = ax.plot_surface(x[i], y[i], Z[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        elif plot_type[i] == "scatter":
            surf_2 = ax.scatter(x[i], y[i], Z[i])
        else:
            AssertionError("error")"""
    
    
    #print(np.shape(x_array), np.shape(y_array), np.shape(z_tilde))


    # Customize the z axis.
    ax.set_zlim(-0.30, 2.40)#(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
