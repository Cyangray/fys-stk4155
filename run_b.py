import numpy as np 
import matplotlib.pyplot as plt 
import sys


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task b) of the project.
Using k-fold cross validation for running data,
and evaluating MSE and R^2 for this sampling method."""



n = 300                 # no. of x and y coordinates
deg = 5                 #degree of polynomial
noise = 0.05            #if zero, no contribution. Otherwise scaling the noise.

# k batches for k-fold.
k = 5
method = "least squares" # "least squares", "ridge" or "lasso"

# Generate data
dataset = data_generate(n, noise)
liste1 = [dataset] #M: Trenger du denne fremdeles, F?
dataset.generate_franke()
dataset.sort_in_k_batches(k)

#Run k-fold algorithm and fit models.
sample = sampling(dataset)
sample.kfold_cross_validation(k, method)

print("Batches: k = ", k)
print("Average mse: ", np.average(sample.mse))#, "\n",
print("Best mse: ",np.min(sample.mse[np.argmin(np.abs(np.array(sample.mse)))]))
print("Average R2: ", np.average(sample.R2))#, "\n",
print("Best R2: ", np.min(sample.R2[np.argmin(np.abs(np.array(sample.R2)))]))

# Plotting the best fit/best beta with the lowest mse.
dataset.reload_data()
fitted = fit(dataset)
fitted.create_design_matrix()
z_model = fitted.test_design_matrix(sample.best_predicting_beta)

# Generate analytical solution for plotting purposes
analytical = data_generate(n, noise=0)
analytical.generate_franke()

# Plot
plot_3d(dataset.x_1d, dataset.y_1d, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

