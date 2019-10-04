import numpy as np
from sklearn.metrics import r2_score as r2


def calc_MSE(z, z_tilde):
    mse = 0
    n=len(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
    return mse/n


def calc_R2_score(z, z_tilde):
    mse = 0
    ms_avg = 0
    n=len(z)
    mean_z = np.mean(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
        ms_avg += (z[i] - mean_z)**2
    return 1. - mse/ms_avg

def calc_R2_score_sklearn(z, z_tilde):
    return r2(z, z_tilde)


def calc_statistics(z, z_tilde):
    mse = calc_MSE(z, z_tilde)
    calc_r2 = calc_R2_score(z, z_tilde)
    return mse, calc_r2

def bias_variance_tradeoff(self, z, z_tilde, n, sigma):
    """ Calculate cost function of bias-variance tradeoff """
    # NOT FINISHED - NEED MATHS AND STUFF
    E = 2.3
    a = 1/n * np.sum((z - E*z_tilde)**2)
    b = 1/n * np.sum((z_tilde - E*z_tilde)**2 + sigma**2)
    return a + b

def print_mse(mse):
    print("Average mse: ", np.average(mse))
    print("Best mse: ", np.min(mse[np.argmin(np.abs(np.array(mse)))]))

def print_R2(R2):
    print("Average R2: ", np.average(R2))
    print("Best R2: ", R2[np.argmax(np.array(R2))])


