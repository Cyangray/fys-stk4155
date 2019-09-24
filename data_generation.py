import numpy as np
import sys


from functions import franke_function

class data_generate():
    def __init__(self, no_datasets, n, noise ):
        self.no_datasets = no_datasets
        #FP: vurdere å lagre n**2 i stedet for n, dersom man bruker n til å lage en meshgrid. 
        #Evt overskrive det når man genererer datasettet fra Franke-funksjonen. Sjekk etterpå at
        #Funksjonene er konsekvente
        self.n = n
        self.noise = noise
        self.resort = int(0)
        

    def generate_franke(self):
        """ Generate franke-data """
        no_datasets = self.no_datasets
        n = self.n
        self.N = n*n

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
            
            if self.noise != 0: #0.5 for centering from [0,1] to [-0.5,0.5]
                self.z_1d[i] += (np.random.randn(n*n)-0.5) * self.noise

    
    def sort_in_k_batches(self, k, random=True):
        """ Sorts the data into k batches, i.e. prepares the data for k-fold cross
        validation. Recommended numbers are k = 3, 4 or 5. "random" sorts the
        dataset randomly. if random==False, it sorts them statistically"""
        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        if self.resort < 1:
            np.savez("backup_data", self.n, self.no_datasets, self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
        else: # self.resort > 0:
            np.load("backup_data")
            
        self.k = k
        idx = 0
        
        #FP: Are we sorting all datasets, or just one? If all datasets:
        #n = self.N * self.no_datasets
        #FP: If only ony dataset:
        n = self.N
        
        self.k_idxs = [[] for i in range(k)]
        limits = [i/k for i in range(k+1)]
        
        if random:
            while idx < n:
                random_number = np.random.rand()
                for i in range(k):
                    if limits[i] <= random_number < limits[i+1]:
                        self.k_idxs[i].append(idx)
                idx += 1
            
        else: #Statistical sorting
            # Lists int values, shuffles randomly and splits into k pieces.
            split = np.arange(n)
            np.random.shuffle(split)
            limits = [int(limits[i]*n) for i in range(limits)]
            for i in range(k):
                self.k_idxs[i].append( split[limits[i] : limits[i+1]] )
                
        self.resort += 1
    
    def sort_training_test(self, i):
        """After soring the dataset into k batches, pick one of them and this one 
        will play the part of the test batch, while the rest will end up being 
        the training batch. the input i should be an integer between 0 and k-1"""
        self.test_indices = self.k_idxs[i]
        self.training_indices = []
        for idx in range(self.k):
            if idx != i:
                self.training_indices += self.k_idxs[idx]
        
    
    def sort_trainingdata_random(self, fractions_trainingdata):
        """ RANDOM! Does not give you the fraction, but the fraction is the probability of being training data.
        Generates lists for sorting training data and test data."""

        # Since training data are renamed further down, make a copy for it to be able to resort later. 
        if self.resort < 1:
            np.savez("backup_data", self.no_datasets, self.x, self.y, self.x_mesh, self.y_mesh, self.z_mesh, self.x_1d, self.y_1d, self.z_1d)
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


    def sort_trainingdata_statistical(self, fractions_trainingdata):
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
        

    def load_terrain_data(self):
        self.data()
        return 1. 

    def fill_array_test_training(self):
        testing = self.test_indices ; training = self.training_indices
        
        #FP: I just want to use the first dataset, just to see how it works when
        #not using datasets at all. The "dataset" version is restored by setting FP = False
        FP = True
        
        if FP:
            no_datasets = self.no_datasets
            ntest = len(testing)
            ntraining = len(training)
    
            self.test_x_1d = np.zeros((no_datasets, ntest))
            self.test_y_1d = np.zeros((no_datasets, ntest))
            self.test_z_1d = np.zeros((no_datasets, ntest))
            self.train_x_1d = np.zeros((no_datasets, ntraining))
            self.train_y_1d = np.zeros((no_datasets, ntraining))
            self.train_z_1d = np.zeros((no_datasets, ntraining))
            for j in range(self.no_datasets):
                self.test_x_1d[j] = np.take(self.x_1d[j,:],testing)
                self.test_y_1d[j] = np.take(self.y_1d[j,:],testing)
                self.test_z_1d[j] = np.take(self.z_1d[j,:],testing)
    
    
            #Rename training data to "normal" data to avoid confusion w/ other functions.
            #self.x = self.x[0,training]
            #self.y = self.y[0,training]
                
            #self.x_mesh = self.x_mesh[0,training]
            #self.y_mesh = self.y_mesh[0,training]
            #self.z_mesh = self.z_mesh[0,training]
    
                self.train_x_1d[j] = np.take(self.x_1d[j,:],training)
                self.train_y_1d[j] = np.take(self.y_1d[j,:],training)
                self.train_z_1d[j] = np.take(self.z_1d[j,:],training)
    
            # Redefine lengths for training and testing.
            self.x_1d = self.train_x_1d
            self.y_1d = self.train_y_1d
            self.z_1d = self.train_z_1d
            self.N = len(training)
            self.N_testing = len(testing)
            
        else:
            self.test_x = self.x[testing]
            self.test_y = self.y[testing]
                
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
