import numpy as np
from sklearn import preprocessing
import sys


from functions import franke_function

class data_generate():
    def __init__(self):
        self.resort = int(0)
        self.normalized = False
        
    def generate_franke(self, n, noise ):
        """ Generate franke-data """
        self.n = n
        self.noise = noise
        self.N = n*n #Number of datapoints (in a square meshgrid)

        self.x = np.zeros((n))
        self.y = np.zeros((n))
        
        self.x_mesh = np.zeros((n, n))
        self.y_mesh = np.zeros((n, n))
        self.z_mesh = np.zeros((n, n))

        self.x_1d = np.zeros((n*n))
        self.y_1d = np.zeros((n*n))
        self.z_1d = np.zeros((n*n))


        self.x = np.sort(np.random.uniform(0, 1, n))
        self.y = np.sort(np.random.uniform(0, 1, n))

        self.x_mesh, self.y_mesh = np.meshgrid(self.x,self.y)
        self.z_mesh = franke_function(self.x_mesh,self.y_mesh)

        self.x_1d = np.ravel(self.x_mesh)
        self.y_1d = np.ravel(self.y_mesh)
        self.z_1d = np.ravel(self.z_mesh)
        
        if self.noise != 0:
            self.z_1d += np.random.randn(n*n) * self.noise

    
    def normalize_dataset(self):
        self.normalized = True
        self.x_unscaled = self.x_1d.copy()
        self.y_unscaled = self.y_1d.copy()
        self.z_unscaled = self.z_1d.copy()
        dataset_matrix = np.stack((self.x_1d, self.y_1d, self.z_1d)).T
        self.scaler = preprocessing.StandardScaler().fit(dataset_matrix)
        [self.x_1d, self.y_1d, self.z_1d] = self.scaler.transform(dataset_matrix).T
        #scaled_matrix = self.scaler.transform(dataset_matrix)
        #self.x_1d, self.y_1d, self.z_1d = scaled_matrix[:,0].T, scaled_matrix[:,1].T, scaled_matrix[:,2].T
            
    def rescale_back(self, x=0, y=0, z=0):
        self.normalized = False
        if isinstance(x, int):
            x = self.x_1d
        if isinstance(y, int):
            y = self.y_1d
        if isinstance(z, int):
            z = self.z_1d
        dataset_matrix = np.stack((x, y, z))
        rescaled_matrix = self.scaler.inverse_transform(dataset_matrix.T)
        return rescaled_matrix.T
    
    def load_terrain_data(self,terrain):
        self.N = np.size(terrain)
        nrows = range(np.size(terrain,0))
        ncolumns = range(np.size(terrain,1))
        self.x_mesh, self.y_mesh = np.meshgrid(nrows,ncolumns)
        self.z_mesh = terrain

        self.x_1d = np.ravel(self.x_mesh)
        self.y_1d = np.ravel(self.y_mesh)
        self.z_1d = np.ravel(self.z_mesh)
    

    def sort_in_k_batches(self, k, random=True):
        """ Sorts the data into k batches, i.e. prepares the data for k-fold cross
        validation. Recommended numbers are k = 3, 4 or 5. "random" sorts the
        dataset randomly. if random==False, it sorts them statistically"""
            
        self.k = k
        idx = 0
        N = self.N
        
        self.k_idxs = [[] for i in range(k)]
        limits = [i/k for i in range(k+1)]
        
        if random:
            #Loop all indexes, Generate a random number, see where it lies in k 
            #evenly spaced intervals, use that to determine in which set to put
            #each index
            while idx < N:
                random_number = np.random.rand()
                for i in range(k):
                    if limits[i] <= random_number < limits[i+1]:
                        self.k_idxs[i].append(idx)
                idx += 1
            
        else: #Statistical sorting
            # Lists int values, shuffles randomly and splits into k pieces.
            split = np.arange(N)
            np.random.shuffle(split)
            limits = [int(limits[i]*N) for i in range(limits)]
            for i in range(k):
                self.k_idxs[i].append( split[limits[i] : limits[i+1]] )
                
    
    def sort_training_test_kfold(self, i):
        """After sorting the dataset into k batches, pick one of them and this one 
        will play the part of the test set, while the rest will end up being 
        the training set. the input i should be an integer between 0 and k-1, and it
        picks the test set. """
        self.test_indices = self.k_idxs[i]
        self.training_indices = []
        for idx in range(self.k):
            if idx != i:
                self.training_indices += self.k_idxs[idx]

    def sort_training_test_random(self, fractions_trainingdata):
        """ RANDOM! Does not give you the fraction, but the fraction 
        is the probability of being training data. Generates lists for 
        sorting training data and test data."""

        # Since training data are renamed other places, make a copy for it to be able to resort later. 
        #self.reload_data()
            
        i = 0
        n = self.n
        self.training_indices = [] ; self.test_indices = []

        while i < self.no_datasets:    
            if np.random.rand() > fractions_trainingdata:
                self.training_indices.append(i)
            else:
                self.test_indices.append(i)
            i += 1


    def sort_training_test_statistical(self, fractions_trainingdata):
        """ STATISTICAL! Does give you the fraction as close as possible.
        Generates lists for sorting training data and test data."""

        # Since training data are renamed other places, make a copy for it to be able to resort later. 
        #self.reload_data()

        no_training_set = int(self.no_datasets*fractions_trainingdata)
        no_test_set = self.no_datasets - no_training_set

        # Lists int values, shuffles randomly and splits into two pieces.
        split = np.arange(self.N)
        np.random.shuffle(split)

        self.training_indices = list(split[:no_training_set])
        self.test_indices = list(split[no_training_set:])


    def fill_array_test_training(self):
        testing = self.test_indices ; training = self.training_indices

        self.reload_data()

        self.test_x_1d = np.take(self.x_1d, testing)
        self.test_y_1d = np.take(self.y_1d, testing)
        self.test_z_1d = np.take(self.z_1d, testing)
        
        self.x_1d = np.take(self.x_1d,training)
        self.y_1d = np.take(self.y_1d,training)
        self.z_1d = np.take(self.z_1d,training)
        
        # Redefine lengths for training and testing.
        self.N = len(training)
        self.N_testing = len(testing)

    def reload_data(self):
        if self.resort < 1:
            np.savez("backup_data", N=self.N, x=self.x_1d, y=self.y_1d, z=self.z_1d)
        else: # self.resort >= 1:
            data = np.load("backup_data.npz")
            self.N = data["N"]
            self.x_1d = data["x"]
            self.y_1d = data["y"]
            self.z_1d = data["z"]
        self.resort = 10
        
        
    def save_data(self):
        np.savez("pregen_dataset", N=self.N, x=self.x_1d, y=self.y_1d, z=self.z_1d)

        
    def load_data(self):
        data = np.load("pregen_dataset.npz")
        self.N = data["N"]
        self.x_1d = data["x"]
        self.y_1d = data["y"]
        self.z_1d = data["z"]





