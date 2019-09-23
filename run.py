import numpy as np 
import matplotlib.pyplot as plt 
import sys

#OLD
from statistics import calc_MSE, calc_R2_score

#NEW
from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d



n = 120         # no. of x and y coordinates
deg = 5         #degree of polynomial
noise = True    #True/False, add random noise.

no_datasets = 11# Number of datasets

# Generate data from franke function

inst.sort_trainingdata(0.7)

# Fit design matrix
FittedModel = fit(Dataset)
liste = [FittedModel]
FittedModel.create_design_matrix(deg)
FittedModel.fit_design_matrix_numpy()
#z_model = FittedModel.y_tilde

# Generate analytical solution for plotting purposes
analytical = data_generate(no_datasets=1, n=n, noise=False)
analytical.generate_franke()

# Plot solutions and analytical
plot_3d(Dataset.x_mesh, Dataset.y_mesh, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

# Generate statistics on the fit

#FittedModel.save_statistics()
#print(FittedModel.mse)
#print(FittedModel.R2score)




"""
#Statistics - need an update.
#print(np.shape(z_1d), np.shape(z_tilde))
#print(calc_MSE(z_1d, np.ravel(z_tilde))) #mean sq error
#print(calc_R2_score(z_1d, np.ravel(z_tilde)))
"""