import numpy as np
#from statistics import calc_MSE, calc_R2_score, calc_R2_score_sklearn
import statistical_functions

class fit():
    def __init__(self, inst):
        self.inst = inst

    def create_design_matrix(self, deg=5): 
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit.
        """

        N = self.inst.N
        x = self.inst.x_1d
        y = self.inst.y_1d

        l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) + y**k
                    
        #Design matrix
        self.X = X
        
        
    
    def fit_design_matrix_numpy(self):
        """Method that uses the design matrix to find the coefficients beta, and
        thus the prediction y_tilde"""
        X = self.X
        z = self.inst.z_1d

        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        self.y_tilde = X @ self.beta
        