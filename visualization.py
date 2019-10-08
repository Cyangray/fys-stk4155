from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_3d(x, y, z, an_x, an_y, an_z, plot_type):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("The franke function model and analytical solution.", fontsize=22)

    #surf = ax.plot_surface(x_array, y_array, z_tilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Surface of analytical solution.
    surf = ax.plot_surface(an_x, an_y, an_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


    surf_2 = ax.scatter(x, y, z)
    """if plot_type[i] == "surface":
        surf = ax.plot_surface(x[i], y[i], Z[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    elif plot_type[i] == "scatter":
        surf_2 = ax.scatter(x[i], y[i], Z[i])
    else:
        AssertionError("error")"""
    # Customize the z axis.
    ax.set_zlim(-0.30, 2.40)#(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
        
        
def plot_3d_terrain(x, y, z, x_map, y_map, z_map):
    """ Plots 3d terrain with trisurf"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf1 = ax.plot_trisurf(x_map, y_map, z_map, cmap=cm.coolwarm, alpha=0.2)
    
    surf2 = ax.scatter(x, y, z)
    
    # Customize the z axis.
    #ax.set_zlim(-0.30, 2.40)#(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_cmap(x, y, z, x_map, y_map, z_map):
    f, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].tripcolor(x,y,z)
    ax[1].tricontourf(x,y,z, 20) # choose 20 contour levels, just to show how good its interpolation is
    ax[1].plot(x,y, 'ko ')
    ax[0].plot(x,y, 'ko ')
    plt.show()
    plt.savefig('test.png')

def plot_bias_var_tradeoff(deg, mse):
    """ Plots bias-variance tradeoff for different polynoial degrees of models. """
    plt.title("Bias-variance tradeoff for different complexity of models")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Prediction error")
    plt.plot(deg, mse)
    plt.grid('on')
    plt.show()

def plot_mse_vs_complexity(deg, mse_test, mse_train):
    """ Plots mse vs. polynomial degree of matrix. """
    fig, ax = plt.subplots()
    ax.set_title("Bias-variance tradeoff for different complexity of models")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Prediction error")
    ax.plot(deg, mse_test, 'r-', label = 'Test sample')
    ax.plot(deg, mse_train, 'b-', label = 'Training sample')
    plt.grid('on')
    plt.legend()
    plt.show()

def plot_bias_variance_vs_complexity(deg, bias, variance):
    """ Plots bias-variance vs. polynomial degree of matrix. """
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax1.set_title("Bias-variance tradeoff for different complexity of models")
    #ax1 = plt.subplot(211)
    ax1.set_ylabel("Bias values")
    ax1.plot(deg, bias, 'r-', label = 'Bias')
    ax1.legend()
    ax1.grid('on')
    
    #ax2 = plt.subplot(212, sharex = ax1)
    ax2.set_xlabel("Polynomial degree")
    ax2.set_ylabel("Variance values")
    ax2.plot(deg, variance, 'b-', label = 'Variance')
    ax2.grid('on')
    ax2.legend()
    plt.show()

    

