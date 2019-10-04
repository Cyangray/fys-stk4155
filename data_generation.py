import numpy as np
import sys


from functions import franke_function

class data_generate():
    def __init__(self, n, noise ):
        self.n = n
        self.noise = noise
        self.resort = int(0)
        

    def generate_franke(self):
        """ Generate franke-data """
        n = self.n
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
        
        if self.noise != 0: #0.5 for centering from [0,1] to [-0.5,0.5]
            self.z_1d += (np.random.randn(n*n)-0.5) * self.noise

    
    def sort_in_k_batches(self, k, random=True):
        """ Sorts the data into k batches, i.e. prepares the data for k-fold cross
        validation. Recommended numbers are k = 3, 4 or 5. "random" sorts the
        dataset randomly. if random==False, it sorts them statistically"""
        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        #if self.resort < 1:
        #    np.savez("backup_data", self.n, self.N, self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        #else: # self.resort > 0:
        #    np.load("backup_data")
            
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
                
        #self.resort += 1
    
    def sort_training_test(self, i):
        """After soring the dataset into k batches, pick one of them and this one 
        will play the part of the test set, while the rest will end up being 
        the training set. the input i should be an integer between 0 and k-1, and it
        picks the test set."""
        self.test_indices = self.k_idxs[i]
        self.training_indices = []
        for idx in range(self.k):
            if idx != i:
                self.training_indices += self.k_idxs[idx]
        
    

        

    def load_terrain_data(self):
        self.data()
        return 1. 

    def fill_array_test_training(self):
        testing = self.test_indices ; training = self.training_indices

        if self.resort < 1:
            np.savez("backup_data", N=self.N, x=self.x_1d, y=self.y_1d, z=self.z_1d) # self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        else: # self.resort > 0:
            data = np.load("backup_data.npz")
            self.N = data["N"]
            self.x_1d = data["x"]
            self.y_1d = data["y"]
            self.z_1d = data["z"]

        print(self.resort)
        print(len(self.x_1d))
        self.resort += 1

        #ntest = len(testing)
        #ntraining = len(training)

        #self.test_x_1d = np.zeros((ntest,))
        #self.test_y_1d = np.zeros((ntest,))
        ##self.test_z_1d = np.zeros((ntest,))
        #self.train_x_1d = np.zeros((ntraining,))
        #self.train_y_1d = np.zeros((ntraining,))
        #self.train_z_1d = np.zeros((ntraining,))

        self.test_x_1d = np.take(self.x_1d, testing)
        self.test_y_1d = np.take(self.y_1d, testing)
        self.test_z_1d = np.take(self.z_1d, testing)
        
        self.x_1d = np.take(self.x_1d,training)
        self.y_1d = np.take(self.y_1d,training)
        self.z_1d = np.take(self.z_1d,training)
        
        #Rename training data to "normal" data to avoid confusion w/ other functions.
        #self.x_1d = self.train_x_1d
        #self.y_1d = self.train_y_1d
        #self.z_1d = self.train_z_1d
        
        # Redefine lengths for training and testing.
        self.N = len(training)
        self.N_testing = len(testing)

        




        

    def sort_trainingdata_random(self, fractions_trainingdata): #OBSOLETE
        """ RANDOM! Does not give you the fraction, but the fraction is the probability of being training data.
        Generates lists for sorting training data and test data."""

        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        if self.resort < 1:
            np.savez("backup_data", self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        else: # self.resort > 0:
            np.load("backup_data")
        i = 0
        n = self.n
        self.training_indices = [] ; self.test_indices = []
        # FP: Careful, here it should have probably been self.n instead of self.no_datasets...?
        while i < self.no_datasets:    
            if np.random.rand() > fractions_trainingdata:
                self.training_indices.append(i)
            else:
                self.test_indices.append(i)
            i += 1
        self.resort += 1


    def sort_trainingdata_statistical(self, fractions_trainingdata): #OBSOLETE
        """ STATISTICAL! Does give you the fraction as close as possible.
        Generates lists for sorting training data and test data."""

        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        if self.resort < 1:
            np.savez("backup_data", self.no_datasets, self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        else: # self.resort > 0:
            np.load("backup_data")
        
        
        
        # FP: Careful, here it should have probably been self.n instead of self.no_datasets...?
        no_training_set = int(self.no_datasets*fractions_trainingdata)
        no_test_set = self.no_datasets - no_training_set

        # Lists int values, shuffles randomly and splits into two pieces.
        split = np.arange(self.no_datasets)
        np.random.shuffle(split)
        
        self.training_indices = list(split[:no_training_set])
        self.test_indices = list(split[no_training_set:])

        self.resort += 1
