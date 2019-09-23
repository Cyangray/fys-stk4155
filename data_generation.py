import numpy as np
import sys


from functions import franke_function

class data_generate():
    def __init__(self, no_datasets, n, noise ):
        self.no_datasets = no_datasets
        self.n = n
        self.noise = noise
        self.resort = int(0)
        

    def generate_franke(self):
        """ Generate franke-data """
        no_datasets = self.no_datasets
        n = self.n
        noise = self.noise

        self.x = np.zeros((no_datasets, n))
        self.y = np.zeros((no_datasets, n))
        
        self.x_mesh = np.zeros((no_datasets, n, n))
        self.y_mesh = np.zeros((no_datasets, n, n))
        self.z_mesh = np.zeros((no_datasets, n, n))

        self.x_1d = np.zeros((no_datasets, n*n))
        self.y_1d = np.zeros((no_datasets, n*n))
        self.z_1d = np.zeros((no_datasets, n*n))

        for i in range(no_datasets):
            self.x[i] = np.sort(np.random.uniform(0, 1, n))
            self.y[i] = np.sort(np.random.uniform(0, 1, n))

            self.x_mesh[i], self.y_mesh[i] = np.meshgrid(self.x[i],self.y[i])
            self.z_mesh[i] = franke_function(self.x_mesh[i],self.y_mesh[i])

            self.x_1d[i] = np.ravel(self.x_mesh[i])
            self.y_1d[i] = np.ravel(self.y_mesh[i])
            self.z_1d[i] = np.ravel(self.z_mesh[i])

            if self.noise:
                self.z_1d[i] += (np.random.randn(n*n)-0.5) * 0.07

        
    def sort_trainingdata(self, fractions_trainingdata):
        """Approx fraction of data to be used as training data (number from 0 to 1)
        or a list of fraction for training data and test data. The last fraction will be extra data. 
        Manual/self written version."""

        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        if self.resort < 1:
            np.savez("backup_data", self.no_datasets, self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        else: # self.resort > 0:
            np.load("backup_data")
        i = 0
        n = self.n
        training_indicies = [] ; test_indicies = []
        while i < self.no_datasets:    
            if np.random.rand() > fractions_trainingdata:
                training_indicies.append(i)
            else:
                test_indicies.append(i)
            i += 1
        self.resort += 1



    def load_terrain_data(self):
        self.data()
        return 1. 

    def fill_array_test_training(self, test, training):

        self.test_x = self.x[testing]
        self.test_y = self.y[i]
            
        self.test_x_mesh = self.x_mesh[testing]
        self.test_y_mesh = self.y_mesh[testing]
        self.test_z_mesh = self.z_mesh[testing]

        self.test_x_1d = self.x_1d[testing]
        self.test_y_1d = self.y_1d[testing]
        self.test_z_1d = self.z_1d[testing]


        #Rename training data to "normal" data to avoid confusion w/ other functions.
        self.x = self.x[training]
        self.y = self.y[training]
            
        self.x_mesh = self.x_mesh[training]
        self.y_mesh = self.y_mesh[training]
        self.z_mesh = self.z_mesh[training]

        self.x_1d = self.x_1d[training]
        self.y_1d = self.y_1d[training]
        self.z_1d = self.z_1d[training]

        # Redefine number of datasets for training and testing.
        self.no_datasets = len(training)
        self.no_datasets_testing = len(testing)
