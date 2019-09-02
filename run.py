import numpy as np 
import matplotlib.pyplot as plt 

from functions import franke_function as franke
from design_matrix import create_design_matrix
from fit import fit_design_matrix_numpy
from statistics import calc_MSE, calc_R2_score

n_x = 120  # no. of points
deg = 5 #degree of polynomial
noise = True #True/False, add random noise.

x_vanilla = np.linspace(0,1,n_x) #np.sort(np.random.uniform(0, 1, n_x))
y_vanilla = np.linspace(0,1,n_x) #np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x_vanilla,y_vanilla)
z = franke(x,y)

x_1d=np.ravel(x)
y_1d=np.ravel(y)
n=int(len(x_1d))

z_1d=np.ravel(z)

if noise:
    z_1d += np.random.randn(n) * 0.07


X = create_design_matrix(x_1d,y_1d,n=deg)
#assume rekkefolge er riktig 

z_tilde = fit_design_matrix_numpy(X, z_1d)

z_tilde = z_tilde.reshape(n_x,n_x)
#print(np.shape(x_1d))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

#surf = ax.plot_surface(x_vanilla, y_vanilla, z_tilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

surf_2 = ax.scatter(x, y, z_tilde)
#print(np.shape(x_vanilla), np.shape(y_vanilla), np.shape(z_tilde))


# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



#print(np.shape(z_1d), np.shape(z_tilde))

print(calc_MSE(z_1d, np.ravel(z_tilde))) #mean sq error

print(calc_R2_score(z_1d, np.ravel(z_tilde)))


