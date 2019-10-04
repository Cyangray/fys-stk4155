import numpy as np 
import matplotlib.pyplot as plt 
import sys


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_bias_var_tradeoff
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task c) of the project.
Running additional variance-bias tradeoff calculations
for different degrees of polynomials of OLS. 

Q: k-fold??? Or just a single test/training set?
Q: Just a single, "random" run of each?
Hmmmmmm...  
Q: Plotting the bias-variance tradeoff on y-axis? Or just mse etc?

Need to do: 
- Correct sampling
- Bias-var tradeoff correct
- Plor correct thing. """

n = 400                 # no. of x and y coordinates
deg = range(2,9)        # degree of polynomial
noise = 0.01            # if zero, no contribution. Otherwise scaling the noise.

# k batches for k-fold.
k = 8
method = "least squares" # "least squares", "ridge" or "lasso"

best_mse = []

for pol_deg in deg:
    #If best of k-fold for the plotting:
    # Generate data
    dataset = data_generate(n, noise)
    liste1 = [dataset] #M: Trenger du denne fremdeles, F?
    dataset.generate_franke()
    dataset.sort_in_k_batches(k)

    #Run k-fold algorithm and fit models.
    sample = sampling(dataset)
    sample.kfold_cross_validation(k, method, pol_deg)

    best_mse.append(np.min(sample.mse[np.argmin(np.abs(np.array(sample.mse)))]))

    print("Run for k = ", k, " and deg = ", pol_deg)
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)

plot_bias_var_tradeoff(deg, best_mse)
