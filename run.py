import numpy as np 
import matplotlib.pyplot as plt 

from functions import franke_function as franke
from design_matrix import create_design_matrix


n_x = 10  # no. of points
deg = 3 #degree of polynomial
noise = True

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z = franke(x,y)

x_1d=np.ravel(x)
y_1d=np.ravel(y)
n=int(len(x_1d))


if noise == True:
    z_1d=np.ravel(z)+ np.random.random(n) * 1
else: 
    z_1d = np.ravel(z)


X= create_design_matrix(x_1d,y_1d,n=deg)

print(X)


