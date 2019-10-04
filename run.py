import numpy as np 
import matplotlib.pyplot as plt 
import sys


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling


n = 150                 # no. of x and y coordinates
deg = 5                 #degree of polynomial
noise = 0.05            #if zero, no contribution. Otherwise scaling the noise.

#How many sets do you want to share the dataset in? In a train-test situation, 
#one of these will be picked as test, the others as train
k = 5
method = "ridge"

# running k-fold algorithm:

# Generate data
dataset = data_generate(n, noise)
liste1 = [dataset] #M: Trenger du denne fremdeles, F?
dataset.generate_franke()
dataset.sort_in_k_batches(k)

#Run k-fold algorithm and fit models.

sample = sampling(dataset)
sample.kfold_cross_validation(k, method)

#best_predicting_beta, test_index = fitted_model.kfold_cross_validation()


# Generate analytical solution for plotting purposes
analytical = data_generate(n, noise=0)
analytical.generate_franke()

#z_model = fitted_model.X @ sampling.beta[np.argmin(sampling.mse)]


#plot_3d(dataset.x_1d, dataset.y_1d, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

#mse, calc_r2 = statistics.calc_statistics(dataset.z_1d, fitted_model.y_tilde)
#print("Mean square error: ", mse, "\n", "R2: ", calc_r2)
#print("Averages: ", np.average(mse), np.average(calc_r2))



"""
# Generate data
dataset = data_generate(n, noise)
liste1 = [dataset]
dataset.generate_franke()
dataset.sort_in_k_batches(k)


# Fit design matrix
fitted_model = fit(dataset)#,deg)
liste2 = [fitted_model]

#Ordinary fitting (exercise 1)
fitted_model.X = fitted_model.create_design_matrix()
fitted_model.y_tilde, fitted_model.beta = fitted_model.fit_design_matrix_numpy()

#K-fold cross-validation (exercise 2)
dataset.sort_in_k_batches(k)
best_predicting_beta, test_index = fitted_model.kfold_cross_validation()


# Generate analytical solution for plotting purposes
analytical = data_generate(n, noise=False)
analytical.generate_franke()

# Plot solutions and analytical
#z_model = fitted_model.y_tilde
z_model = fitted_model.X @ best_predicting_beta


plot_3d(dataset.x_1d, dataset.y_1d, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

mse, calc_r2 = statistics.calc_statistics(dataset.z_1d, fitted_model.y_tilde)
print("Mean square error: ", mse, "\n", "R2: ", calc_r2)
#print("Averages: ", np.average(mse), np.average(calc_r2))




dataset = data_generate(n, noise)
liste1 = [dataset] #M: Do you still need this, F?
dataset.generate_franke()

# Fit design matrix
fitted_model = fit(dataset)
liste2 = [fitted_model] #M: Do you still need this, F?

#Ordinary fitting
fitted_model.X = fitted_model.create_design_matrix(deg)
z_model, beta = fitted_model.fit_design_matrix_numpy()

# Generate analytical solution for plotting purposes
analytical = data_generate(n, noise=0)
analytical.generate_franke()

# Statistical evaluation
mse, calc_r2 = statistics.calc_statistics(dataset.z_1d, z_model)
print("Mean square error: ", mse, "\n", "R2 score: ", calc_r2)

# Plot solutions and analytical for comparison
plot_3d(dataset.x_1d, dataset.y_1d, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])


"""