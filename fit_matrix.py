import numpy as np

class fit():
	def __init__(self, inst):
		self.inst = inst

	def create_design_matrix(self, deg=5):
		"""
		Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
		Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
		"""

		N = self.inst.n**2
		n = deg
		x = self.inst.x_1d
		y = self.inst.y_1d

		l = int((n+1)*(n+2)/2)		# Number of elements in beta
		X = np.ones((self.inst.no_datasets, N,l))

		for j in range(self.inst.no_datasets):
			for i in range(1,n+1):
				q = int((i)*(i+1)/2)
				for k in range(i+1):
					X[j, :,q+k] = x[j]**(i-k) + y[j]**k
		self.X = X


	def fit_design_matrix_numpy(self):
		X = self.X
		z = self.inst.z_1d

		y_tilde = np.zeros((self.inst.no_datasets, ))
		for j in range(self.inst.no_datasets):
			beta = np.linalg.inv(X[j].T.dot(X[j])).dot(X[j].T).dot(z[j])
			y_tilde[j] = X @ beta
			#print(np.shape(y_tilde))

		self.y_tilde = y_tilde