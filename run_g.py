import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import reduce4

from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_3d_terrain, plot_bias_var_tradeoff, plot_mse_vs_complexity, plot_cmap
import statistical_functions as statistics
from sampling_methods import sampling



def rung(CV, k, method, lambd, pol_deg):
    #Load the dataset
    dataset = data_generate()
    dataset.load_data()
    shape = dataset.shape
    
    #Normalize and sample the dataset
    dataset.normalize_dataset()
    #Run k-fold algorithm and fit models.
    if CV:
        sample = sampling(dataset)
        dataset.sort_in_k_batches(k)
        if method == "least squares":
            sample.kfold_cross_validation(k, method, deg = pol_deg)
        else:
            sample.kfold_cross_validation(k, method, deg = pol_deg, lambd = lambd)
    
        #Print statistics if CV
        if method == "least squares":
            print("\n" + "Run for k = ", k, " and deg = ", pol_deg)
        else:
            print("\n"+"Batches: k = ", k, " Lambda = ", lambd, " and deg = ", pol_deg) 
        
        bias = np.average(sample.bias)
        variance = np.average(sample.variance)
        mse = np.average(sample.mse)
        R2 = np.average(sample.R2)
        statistics.print_mse(sample.mse)
        statistics.print_R2(sample.R2)
    else:
        print('\n' + 'Run for deg = ', pol_deg)
    
    
    
    # Plotting the best fit/best beta with the lowest mse.
    fitted = fit(dataset)
    fitted.create_design_matrix(deg = pol_deg)
    if CV:
        z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)
    else:
        if method == "least squares":
            z_model_norm, beta = fitted.fit_design_matrix_numpy()
        elif method == "ridge":
            z_model_norm, beta = fitted.fit_design_matrix_ridge()
        else:
            z_model_norm, beta = fitted.fit_design_matrix_lasso()
        
        #Print statistics if not CV
        mse, R2 = statistics.calc_statistics(dataset.z_1d, z_model_norm)
        print("Mean square error: ", mse, "\n", "R2 score: ", R2)
    
    rescaled_dataset = dataset.rescale_back(z = z_model_norm)
    z_model = rescaled_dataset[2]
    
    z_matrix = np.empty(shape)
    z_matrix[:] = np.nan
    for i, z_value in enumerate(z_model):
        z_matrix[int(rescaled_dataset[0,i]), int(rescaled_dataset[1,i])] = z_value
    
    return z_matrix, mse, R2, bias, variance



#Regression parameters
CV = True                   #Use Cross Validation?
k=5                         #k-fold?
method = 'lasso'    #Method
no_lambdas = 6
lambdas = [10**(-no_lambdas + 3 + i) for i in range(no_lambdas)]
degs = [3,5,7,9,11,13]               #degree of the polynomial
#pol_deg = 5
lambd = 1e-2

ind_var = degs
ind_var_text = "deg"


z_matrices = []
mses = []
R2s = []
biases = []
variances = []
for pol_deg in ind_var:
    current_z_matrix, current_mse, current_R2, current_bias, current_variance = rung(CV, k, method, lambd, pol_deg)
    z_matrices.append(current_z_matrix)
    mses.append(current_mse)
    R2s.append(current_R2)
    biases.append(current_bias)
    variances.append(current_variance)


if CV:
    CV_text = "w/"
else:
    CV_text = "without"


# Plot terrains
fig, axs = plt.subplots(nrows = 1, ncols = len(ind_var), sharey = True)
xlabels = [ind_var_text + " = " + str(i) for i in ind_var]
axs[2].set_title("Model of map for " + method + ", " + CV_text + " cross validation")
for i, ax in enumerate(axs):
    ax.imshow(z_matrices[i], cmap = cm.coolwarm)
    ax.set_xlabel(xlabels[i])
plt.show()

# Plot errors
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
ax1.set_title("MSE and R2 score of map for " + method + ", " + CV_text + " cross validation")
ax1.plot(ind_var,mses,'r-',label = "MSE")
if ind_var == lambdas:
    ax1.set_xscale('log')
ax1.grid('on')
ax1.legend()

ax2.set_xlabel(ind_var_text)
ax2.plot(ind_var,R2s,'b-',label = "R2 score")
if ind_var == lambdas:
    ax2.set_xscale('log')
ax2.grid('on')
ax2.legend()
plt.show()

#Plot bias and variance
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
ax1.set_title("Bias and variance of map for " + method + ", " + CV_text + " cross validation")
ax1.plot(ind_var,biases,'r-',label = "Bias")
if ind_var == lambdas:
    ax1.set_xscale('log')
ax1.grid('on')
ax1.legend()

ax2.set_xlabel(ind_var_text)
ax2.plot(ind_var,variances,'b-',label = "Variance")
if ind_var == lambdas:
    ax2.set_xscale('log')
ax2.grid('on')
ax2.legend()
plt.show()




