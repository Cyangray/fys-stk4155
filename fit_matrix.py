import numpy as np
from statistics import calc_MSE, calc_R2_score, calc_R2_score_sklearn

class fit():
    def __init__(self, inst):
        self.inst = inst

    def create_design_matrix(self, deg=5): 
        #FP: er ikke dette det samme som sklearn.preprocessing.PolynomialFeatures ? 
        #M: Jepp, men dette er vaar egen versjon :)

        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit.
        """

        N = self.inst.n**2
        n = deg
        x = self.inst.x_1d
        y = self.inst.y_1d

        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X = np.ones((self.inst.no_datasets, N, l))

        for j in range(self.inst.no_datasets):
            for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                    X[j, :,q+k] = x[j]**(i-k) + y[j]**k
                    
        #Design matrices for each dataset.
        self.X = X
        
        #All the design matrices, stacked together.
        self.X2D = np.concatenate(self.X)


    def fit_design_matrix_numpy(self):
        X = self.X
        z = self.inst.z_1d
        self.y_tilde = np.zeros((self.inst.no_datasets, len(z[0,:]) ))

        for j in range(self.inst.no_datasets): 
            beta = np.linalg.inv(X[j].T.dot(X[j])).dot(X[j].T).dot(z[j])
            self.y_tilde[j] = X[j] @ beta
        
    
    def fit_design_matrix_numpy_all(self)
        """alternative for a more general fit of all the no_datasets"""
        X2D = self.X2D
        z = self.inst.z_1d

        longz = np.concatenate(z)
        beta = np.linalg.inv(X2D.T.dot(X2D)).dot(X2D.T).dot(longz)
        self.y_tilde = X2D @ beta
        #FP: må gjøres: del y_tilde tilbake i datasets, slik at den kan plottes ordentlig mtp meshgrid osv (?) 
        #M: Maa sjekke ut
        