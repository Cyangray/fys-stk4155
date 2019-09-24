import numpy as np 
import matplotlib.pyplot as plt 
import sys


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistics


n = 120                 # no. of x and y coordinates
deg = 5                 #degree of polynomial
noise = 0.05            #if zero, no contribution. Otherwise scaling the noise.

#If you want to sort, approx fraction of data to be used for training. The rest is for testing. OBS! Statistical vs. random!
training_data_fraction = 0.9

no_datasets = 10     # Number of datasets

# Generate data
dataset = data_generate(no_datasets, n, noise)
dataset.generate_franke()
#dataset.sort_trainingdata_random(training_data_fraction)
dataset.sort_trainingdata_statistical(training_data_fraction)
dataset.fill_array_test_training()

# Fit design matrix
fitted_model = fit(dataset)
liste = [fitted_model]
fitted_model.create_design_matrix(deg)
fitted_model.fit_design_matrix_numpy()


# Generate analytical solution for plotting purposes
analytical = data_generate(no_datasets=1, n=n, noise=False)
analytical.generate_franke()

# Plot solutions and analytical
z_model = fitted_model.y_tilde
#plot_3d(dataset.x_mesh, dataset.y_mesh, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

mse, calc_r2 = statistics.calc_statistics(dataset.no_datasets, dataset.z_1d, fitted_model.y_tilde)
print("Mean square error: ", mse, "\n", "R2: ", calc_r2)
print("Averages: ", np.average(mse), np.average(calc_r2))
