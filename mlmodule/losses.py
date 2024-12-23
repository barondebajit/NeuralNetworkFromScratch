#Losses module for mlmodule

#Imports
import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred, derivative=False):
    if derivative:
        return y_pred - y_true
    return np.mean(np.square(y_pred - y_true))

def binary_crossentropy(y_true, y_pred, derivative=False):
    if derivative:
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, derivative=False):
    if derivative:
        return (y_pred - y_true) / y_pred.shape[0]
    return -np.mean(y_true * np.log(y_pred))