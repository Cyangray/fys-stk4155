import numpy as np
from statistics import calc_MSE, calc_R2_score, calc_R2_score_sklearn

class fit():
    def __init__(self, inst):
        self.inst = inst

    def create_design_matrix(self, deg=5): #FP: er ikke dette det samme som sklearn.preprocessing.PolynomialFeatures ?
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
                    
        #self.X contains a design matrix for each dataset
        self.X = X
        
        #self.X2D are all the design matrices, stacked together, and describe as
        #such the whole sample
        self.X2D = np.concatenate(self.X)


    def fit_design_matrix_numpy(self):
        X = self.X
        z = self.inst.z_1d
        
        self.y_tildes = np.zeros((self.inst.no_datasets, len(z[0,:]) ))
        self.betas = np.zeros((self.inst.no_datasets, len(X[0,0,:])))
        #FP: Forsiktig! Nå dannes beta for hvert dataset, og ikke en generell 
        #beta for alle datasettene til sammen! Som følge, tilhører hver y_tilde
        #kun ett datasett, og ikke hele ensamble. Jeg laga et alternativ under,
        #slik at mse og R2score som regnes ut tar utgangspunkt i alle datapunktene
        #(og ikke bare ett av de fire datasettene)
        for j in range(self.inst.no_datasets): 
            self.betas[j] = np.linalg.inv(X[j].T.dot(X[j])).dot(X[j].T).dot(z[j])
            self.y_tildes[j] = X[j] @ self.betas[j]
            #print(np.shape(y_tilde))
            
        #FP: alternative for a more general fit of all the no_datasets
        X2D = self.X2D
        longz = np.concatenate(z)
        self.beta = np.linalg.inv(X2D.T.dot(X2D)).dot(X2D.T).dot(longz)
        self.y_tilde = X2D @ self.beta
        #FP: må gjøres: del y_tilde tilbake i datasets, slik at den kan plottes ordentlig mtp meshgrid osv (?)
        
    def save_statistics(self): #FP: rotete, I know
        y = np.concatenate(self.inst.z_1d)
        #y_tilde = np.concatenate(self.y_tilde)
        y_tilde = self.y_tilde
        self.mse = calc_MSE(y, y_tilde)
        self.R2score = calc_R2_score(y, y_tilde)