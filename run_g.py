import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import reduce4

from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_3d_terrain, plot_bias_var_tradeoff, plot_mse_vs_complexity
import statistical_functions as statistics
from sampling_methods import sampling

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

terrain1 = reduce4(terrain1)
terrain1 = reduce4(terrain1)
dataset = data_generate()
dataset.load_terrain_data(terrain1)
k=5
method = 'ridge'
lambd = 1e-2

liste1 = [dataset] #M: Trenger du denne fremdeles, F?
dataset.normalize_dataset()
dataset.sort_in_k_batches(k)

#Run k-fold algorithm and fit models.
sample = sampling(dataset)
sample.kfold_cross_validation(k, method, lambd=lambd)

print("\n"+"Batches: k = ", k, " Lambda = ", lambd)
statistics.print_mse(sample.mse)
statistics.print_R2(sample.R2)

# Plotting the best fit/best beta with the lowest mse.
fitted = fit(dataset)
liste2 = [fitted] #M: Trenger du denne fremdeles, F?
liste2 = [sample] #M: Trenger du denne fremdeles, F?
fitted.create_design_matrix()
z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
z_model = rescaled_dataset[2]

#plot_3d_terrain(dataset.x_unscaled, dataset.y_unscaled, z_model)
plot_3d_terrain(rescaled_dataset[0], rescaled_dataset[1], rescaled_dataset[2])





