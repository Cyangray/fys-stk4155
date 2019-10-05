import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task d) of the project.
Ridge regression and dependence on lambda.

Need to do:
Q: Var. deg. of polynomials?
Q: Exactly what to run for what.. :) 

A: runfile not done at all."""



n = 300                 # no. of x and y coordinates
deg = 5                 #degree of polynomial
noise = 0.05            #if zero, no contribution. Otherwise scaling the noise.
no_lambdas = 6          # the number of labdas you want to test

# k batches for k-fold.
k = 5
method = "least squares" # "least squares", "ridge" or "lasso"
lambdas = np.linspace(0, 4, no_lambdas)

for i in range(no_lambdas):
    # Generate data
    dataset = data_generate(n, noise)
    liste1 = [dataset] #M: Trenger du denne fremdeles, F?
    dataset.generate_franke()
    dataset.sort_in_k_batches(k)

    #Run k-fold algorithm and fit models.
    sample = sampling(dataset)
    sample.kfold_cross_validation(k, method, lambd=lambdas[i])

    print("Batches: k = ", k, " Lambda = ", lambdas[i])
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)

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


try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")