import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def relu(x):
    return np.array([max(0, a) for a in x])

def unit_step_func(x):
    return np.array([1 if a > 0 else 0 for a in x])

def dtanh(x):
    tmp = np.tanh(x)
    return 1 - tmp * tmp

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def dsigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)

def d_tanh_sigmoid(x):
    y = sigmoid(x)
    d = dtanh(y) * y * (1 - y)
    return d

def dtanh_sigmoid(x):
    y = sigmoid(x)
    return dtanh(y)

def gaussion(x, mean, sigma):
    return 1./(math.sqrt(2*math.pi) * sigma) * np.exp(-0.5*(x-mean)*(x-mean) / (sigma * sigma))
 
def uniform_noise():
    U = np.random.rand()
    return U

def logistic_noise():
    U = uniform_noise()
    return np.log(U) - np.log(1-U)

def gumbel_noise():
    U = uniform_noise()
    return -np.log(-np.log(U))

def bernoulli_noise(p):
    alpha = p / (1 - p)
    log_alpha = math.log(alpha)
    logit = log_alpha + logistic_noise()
    sample = 1 if logit >0 else 0
    return sample
    
def bin_concrete_sample(p, beta):
    alpha = p / (1 - p)
    log_alpha = math.log(alpha)
    logit = log_alpha + logistic_noise()
    return sigmoid(logit / beta)

def hard_concrete_sample(p, beta, gamma, zeta):
    s = bin_concrete_sample(p, beta)
    s = s * (zeta - gamma) + gamma
    s = np.clip(s, 0, 1)
    return s


