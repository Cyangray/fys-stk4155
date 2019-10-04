import numpy as np
#from statistics import calc_MSE, calc_R2_score, calc_R2_score_sklearn
import statistical_functions as statistics
import sys

class fit():
    def __init__(self, inst):#, deg=5):
        self.inst = inst
        #self.deg = deg

    def create_design_matrix(self, x=0, y=0, z=0, N=0, deg=5):#, data='none', deg=5): 
        if type(x) == int:
            x = self.inst.x_1d
            y = self.inst.y_1d
            z = self.inst.z_1d
            N = self.inst.N
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit.
        """
        
        """if data=='none':
            N = self.inst.N
            x = self.inst.x_1d
            y = self.inst.y_1d
        elif data=='train':
            N = self.inst.N_training
            x = self.inst.train_x_1d
            y = self.inst.train_y_1d
        elif data=='test':
            N = self.inst.N_testing
            x = self.inst.test_x_1d
            y = self.inst.test_y_1d
        deg = self.deg"""

        #self.N = N
        self.x = x
        self.y = y
        self.z = z

        self.l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, self.l))

        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) + y**k
                    
        #Design matrix
        self.X = X
        return X
        
    def fit_design_matrix_numpy(self):
        """Method that uses the design matrix to find the coefficients beta, and
        thus the prediction y_tilde"""
        X = self.X
        z = self.z

        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_ridge(self):
        """Method that uses the design matrix to find the coefficients beta with 
        the ridge method, and thus the prediction y_tilde"""
        X = self.X
        z = self.z
        l = 0.3 #arbitrary for now.

        beta = np.linalg.inv(X.T.dot(X) + l*np.identity(self.l)).dot(X.T).dot(z)
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_lasso(self):
        """Here, the lasso algorithm will be implemented."""
        X = self.X
        z = self.z

        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        y_tilde = X @ beta
        return y_tilde, beta

    def test_design_matrix(self, beta):
        """Testing the design matrix"""
        y_tilde = self.X @ beta
        return y_tilde

        

     