#Activation functions for mlmodule

#Imports
import numpy as np
import pandas as pd

def sigmoid(Z, derivative=False):
    if derivative:
        return sigmoid(Z) * (1 - sigmoid(Z))
    return 1 / (1 + np.exp(-Z))

def relu(Z, derivative=False):
    if derivative:
        return np.where(Z <= 0, 0, 1)
    return np.maximum(0, Z)

def leaky_relu(Z, derivative=False):
    if derivative:
        return np.where(Z <= 0, 0.01, 1)
    return np.maximum(0.01 * Z, Z)

def tanh(Z, derivative=False):
    if derivative:
        return 1 - np.square(np.tanh(Z))
    return np.tanh(Z)