import numpy as np
import sys


from functions import franke_function

class data_generate():
    def __init__(self, no_datasets, n, noise ):
        self.no_datasets = no_datasets
        self.n = n
        self.noise = noise
        self.dumm = 3
        

    def generate_franke(self):
        """ Generate franke-data """
        no_datasets = self.no_datasets
        n = self.n
        noise = self.noise

        self.x = np.zeros((no_datasets, n))
        self.y = np.zeros((no_datasets, n))
        #z = np.zeros((no_datasets, n))
        
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

        

    def load_terrain_data():
        self.data()
        return 1. 