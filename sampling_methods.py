import numpy as np 

import statistical_functions as statistics
from fit_matrix import fit

class sampling():
    def __init__(self, inst):
        self.inst = inst

    def kfold_cross_validation(self, k, method):
        """Method that implements the k-fold cross-validation algorithm. It takes
        as input the method we want to use. if "none" an ordinary OLS will be evaulated.
        if "ridge" then the ridge method will be used, and respectively the same for "LASSO"."""

        #self.inst.sort_in_k_batches()
        #for i in range(self.inst.k):
        #    self.inst.k_idxs[i]


        #lowest_mse = 1.0e99
        #highestR2score = 0.for i in range(self.inst.k):
        #best_predicting_beta = self.beta
        #test_index = 0
        inst = self.inst

        self.mse = np.ones(k)
        self.calc_r2 = np.ones(k)
        #self.beta = np.zeros((k, ))
        design_matrix = fit(inst)
        
        for i in range(self.inst.k):
            #pick the i-th set as test
            inst.sort_training_test(i)
            inst.fill_array_test_training()

            design_matrix.create_design_matrix()
            if method == None:
                z_pred, beta_pred = design_matrix.fit_design_matrix_numpy()
            elif method == "ridge":
                z_pred, beta_pred = design_matrix.fit_design_matrix_ridge()
            elif method == "lasso":
                z_pred, beta_pred = design_matrix.fit_design_matrix_lasso()
            else:
                sys.exit("Wrongly designated method: ", method)
            #print(np.shape(beta_pred))
                
            #Make design matrices for both the new train and test sets
            #Xtrain = self.create_design_matrix('train')
            #Xtest = self.create_design_matrix('test')

            #Find out which values get predicted by the training set
            #beta_train = np.linalg.inv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(self.inst.train_z_1d)
            X_test = design_matrix.create_design_matrix(x=inst.test_x_1d, y=inst.test_y_1d, z=inst.test_z_1d, N=inst.N_testing)
            z_test = design_matrix.test_design_matrix(beta_pred)
                
            #evaluate the training set
            #self.mse, self.calc_r2 = statistics.calc_statistics(z_pred, z_test)
            #print(self.mse, self.calc_r2)
            #if mse < lowest_mse:
            #    lowest_mse = mse
            #    best_predicting_beta = beta_train
            #    test_index = i
             


"""
        if method == None:
            lowest_mse = 1.0e99
            highestR2score = 0.for i in range(self.inst.k):
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
                mse, calc_r2 = statistics.calc_statistics(z_predicted, z_observed)
                if mse < lowest_mse:
                    lowest_mse = mse
                    best_predicting_beta = beta_train
                    test_index = i
             
        elif method == 'ridge': #To implement
            return 0
        elif method == 'LASSO': #To implement
            return 0
        else:
            print('method unknown')"""
            
#return best_predicting_beta, test_index