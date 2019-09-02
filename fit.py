import numpy as np 

def fit_design_matrix_numpy(X,z):

    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)

    y_tilde = X @ beta

    return y_tilde