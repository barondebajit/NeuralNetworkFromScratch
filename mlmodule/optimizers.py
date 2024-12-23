#Optimizers module for mlmodule

#Imports
import numpy as np
import pandas as pd

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, w, dw, regularizer=None, reg_lambda=0.0):
        if regularizer is not None:
            dw += regularizer.gradient(w, reg_lambda)
        return w - self.learning_rate * dw

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update_weights(self, w, dw, regularizer=None, reg_lambda=0.0):
        print("Inside update_weights")
        print("dw: ", dw)
        '''self.t += 1
        if self.m is None:
            self.m = np.zeros_like(dw)
            self.v = np.zeros_like(dw)
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        dw = m_hat / (np.sqrt(v_hat) + self.epsilon)
        if regularizer is not None:
            dw += regularizer.gradient(w, reg_lambda)'''
        return w - self.learning_rate * dw
    
    def update_bias(self, b, db):
        print("Inside update_bias")
        print("db: ", db)
        return b - self.learning_rate * db