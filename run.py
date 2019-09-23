import numpy as np 
import matplotlib.pyplot as plt 
import sys

#OLD
#from functions import franke_function as franke
#from design_matrix import create_design_matrix
#from fit import fit_design_matrix_numpy
from statistics import calc_MSE, calc_R2_score

#NEW
from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d



n = 120         # no. of x and y coordinates
deg = 5         #degree of polynomial
noise = True    #True/False, add random noise.

no_datasets = 4 # Number of datasets

# Generate data from franke function
inst = data_generate(no_datasets, n, noise)
inst.generate_franke()

# Fit design matrix
design_matrix = fit(inst)
design_matrix.create_design_matrix(deg)
design_matrix.fit_design_matrix_numpy()
z_model = design_matrix.y_tilde

# Generate analytical solution for plotting purposes
analytical = data_generate(no_datasets=1, n=n, noise=False)
analytical.generate_franke()

# Plot solutions and analytical
plot_3d(inst.x_mesh, inst.y_mesh, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])




"""
#Statistics - need an update.
#print(np.shape(z_1d), np.shape(z_tilde))
#print(calc_MSE(z_1d, np.ravel(z_tilde))) #mean sq error
#print(calc_R2_score(z_1d, np.ravel(z_tilde)))
"""