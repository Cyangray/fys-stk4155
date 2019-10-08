import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task a) of the project.
Generating dataset from Franke function with background noise
for standard least square regression w/polynomials up to the
fifth order. Also adding MSE and R^2 score."""


n = 150                 # no. of x and y coordinates
deg = 5                 # degree of polynomial
noise = 0.05            # if zero, no contribution. Otherwise scaling the noise.

# Load dataset and generate Franke function
dataset = data_generate()
liste1 = [dataset] #M: Do you still need this, F?
dataset.generate_franke(n, noise)

# Normalize the dataset
dataset.normalize_dataset()

# Fit design matrix
fitted_model = fit(dataset)
liste2 = [fitted_model] #M: Do you still need this, F?

# Ordinary least square fitting
fitted_model.create_design_matrix(deg)
z_model_norm, beta = fitted_model.fit_design_matrix_numpy()

# Scale back the dataset
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
x_model = rescaled_dataset[0]
y_model = rescaled_dataset[1]
z_model = rescaled_dataset[2]

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(n, noise=0)

# Statistical evaluation
mse, calc_r2 = statistics.calc_statistics(dataset.z_1d, z_model)
print("Mean square error: ", mse, "\n", "R2 score: ", calc_r2)

# Plot solutions and analytical for comparison
plot_3d(dataset.x_unscaled, dataset.y_unscaled, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])


try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")