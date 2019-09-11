import numpy as np 
import matplotlib.pyplot as plt 

from functions import franke_function as franke
from design_matrix import create_design_matrix
from fit import fit_design_matrix_numpy
from statistics import calc_MSE, calc_R2_score
from visualization import plot_3d

import sys

n = 120  # no. of x and y coordinates
deg = 5 #degree of polynomial
noise = True #True/False, add random noise.

no_datasets = 4

x_mesh = np.zeros((no_datasets, n, n))
y_mesh = np.zeros((no_datasets, n, n))
z = np.zeros((no_datasets, n, n))
z_model = np.zeros((no_datasets, n, n))
design_matrix = np.zeros((no_datasets, n*n, 21))


for i in range(no_datasets):
    x_array = np.sort(np.random.uniform(0, 1, n)) #np.linspace((0,1,n, n)) #
    y_array = np.sort(np.random.uniform(0, 1, n)) # np.linspace(0,1,n) #np.sort(np.random.uniform(0, 1, n))

    x_mesh[i], y_mesh[i] = np.meshgrid(x_array,y_array)
    z[i] = franke(x_mesh[i],y_mesh[i])

    x_1d = np.ravel(x_mesh[i])
    y_1d = np.ravel(y_mesh[i])
    z_1d = np.ravel(z[i])

    #print(np.shape(z_1d))

    if noise:
        z_1d += np.random.randn(n*n) * 0.07





    design_matrix[i] = create_design_matrix(x_1d,y_1d,n=deg)
    z_tilde = fit_design_matrix_numpy(design_matrix[i], z_1d)
    z_model[i] = z_tilde.reshape(n,n)
    


plot_3d(x_mesh, y_mesh, z, z_model)#, ["surface", "scatter"])


#print(np.shape(z_1d), np.shape(z_tilde))

#print(calc_MSE(z_1d, np.ravel(z_tilde))) #mean sq error

#print(calc_R2_score(z_1d, np.ravel(z_tilde)))


