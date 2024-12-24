#Losses module for mlmodule

#Imports
import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    if derivative:
        return y_pred - y_true
    return np.mean(np.square(y_pred - y_true))

def binary_crossentropy(y_true, y_pred, derivative=False):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return -y_true/y_pred + (1-y_true)/(1-y_pred)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, derivative=False):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return (y_pred - y_true) / y_pred.shape[0]
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
