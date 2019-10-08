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


#Parameters for the simulation
n = 300                 # no. of x and y coordinates
deg = 5                 #degree of polynomial
noise = 0.05            #if zero, no contribution. Otherwise scaling the noise.
no_lambdas = 6          # the number of labdas you want to test
k = 5                   # k batches for k-fold.
method = "ridge"        # "least squares", "ridge" or "lasso"
lambdas = [10**(-no_lambdas + i) for i in range(no_lambdas)]


# Load dataset and Franke function
dataset = data_generate()
liste1 = [dataset] #M: Trenger du denne fremdeles, F?
dataset.generate_franke(n, noise)

# Normalize the dataset and divide in samples
dataset.normalize_dataset()
dataset.sort_in_k_batches(k)

for i in range(no_lambdas):

    #Run k-fold algorithm and fit models.
    sample = sampling(dataset)
    sample.kfold_cross_validation(k, method, lambd=lambdas[i])
    
    print("\n"+"Batches: k = ", k, " Lambda = ", lambdas[i])
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)

    # Plotting the best fit/best beta with the lowest mse.
    dataset.reload_data()
    fitted = fit(dataset)
    liste2 = [fitted] #M: Trenger du denne fremdeles, F?
    liste2 = [sample] #M: Trenger du denne fremdeles, F?
    fitted.create_design_matrix()
    z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(n, noise=0)

#rescale for plotting:
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
z_model = rescaled_dataset[2]

# Plot
plot_3d(dataset.x_unscaled, dataset.y_unscaled, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])


try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")