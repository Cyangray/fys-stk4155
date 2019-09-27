import numpy as np
#from statistics import calc_MSE, calc_R2_score, calc_R2_score_sklearn
import statistical_functions as statistics

class fit():
    def __init__(self, inst, deg=5):
        self.inst = inst
        self.deg = deg

    def create_design_matrix(self, data='none'): 
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit.
        """
        if data=='none':
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
        deg = self.deg

        l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) + y**k
                    
        #Design matrix
        return X
        
    def fit_design_matrix_numpy(self):
        """Method that uses the design matrix to find the coefficients beta, and
        thus the prediction y_tilde"""
        X = self.X
        z = self.inst.z_1d

        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        y_tilde = X @ beta
        return y_tilde, beta
        
    def evaluate_test(self, z_predicted, z_observed):
        """Method that tests how good the model fits the test set"""
        mse, calc_r2 = statistics.calc_statistics(z_predicted, z_observed)
        return mse, calc_r2
        
    def kfold_cross_validation(self, method='none'):
        """Method that implements the k-fold cross-validation algorithm. It takes
        as input the method we want to use. if "none" an ordinary OLS will be evaulated.
        if "ridge" then the ridge method will be used, and respectively the same for "LASSO"."""
        
        if method == 'none':
            lowest_mse = 1.0e99
            highestR2score = 0.
            best_predicting_beta = self.beta
            test_index = 0
            
            for i in range(self.inst.k):
                #pick the i-th set as test
                self.inst.sort_training_test(i)
                self.inst.fill_array_test_training()
                
                #Make design matrices for both the new train and test sets
                Xtrain = self.create_design_matrix('train')
                Xtest = self.create_design_matrix('test')
                
                #Find out which values get predicted by the training set
                beta_train = np.linalg.inv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(self.inst.train_z_1d)
                z_pred = Xtest @ beta_train
                
                #evaluate the training set
                mse, calc_r2 = self.evaluate_test(z_pred, self.inst.test_z_1d)
                if mse < lowest_mse:
                    lowest_mse = mse
                    best_predicting_beta = beta_train
                    test_index = i
                
        elif method == 'ridge': #To implement
            return 0
        elif method == 'LASSO': #To implement
            return 0
        else:
            print('method unknown')
            
        return best_predicting_beta, test_index
        