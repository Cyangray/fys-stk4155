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

        l = int((deg+1)*(deg+2)/2)		# Number of elements in beta
        Xs = np.ones((self.inst.no_datasets, N, l))

        for j in range(self.inst.no_datasets):
            for i in range(1,deg+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                    Xs[j, :,q+k] = x[j]**(i-k) + y[j]**k
                    
        #Design matrices for each dataset.
        self.Xs = Xs
        
        #All the design matrices, stacked together.
        self.X = np.concatenate(self.Xs)
        

    def fit_design_matrix_numpy(self):
        Xs = self.Xs
        z = self.inst.z_1d
        self.y_tilde = np.zeros((self.inst.no_datasets, len(z[0,:]) ))
        self.betas = np.zeros((self.inst.no_datasets, len(Xs[0,0,:]) ))

        for j in range(self.inst.no_datasets): 
            self.betas[j] = np.linalg.inv(Xs[j].T.dot(Xs[j])).dot(Xs[j].T).dot(z[j])
            self.y_tilde[j] = Xs[j] @ self.betas[j]
        
    
    def fit_design_matrix_numpy_all(self):
        """alternative for a more general fit of all the no_datasets"""
        X = self.X
        z = self.inst.z_1d

        longz = np.concatenate(z)
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(longz)
        self.y_tilde = X @ beta
        #FP: må gjøres: del y_tilde tilbake i datasets, slik at den kan plottes ordentlig mtp meshgrid osv (?) 
        #M: Maa sjekke ut
        