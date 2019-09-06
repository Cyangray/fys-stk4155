from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_3d(x, y, z, Z): #, plot_type):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #surf = ax.plot_surface(x_array, y_array, z_tilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    print(np.shape(Z))

    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    for i in range(len(Z)):
        surf_2 = ax.scatter(x[i], y[i], Z[i])
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
