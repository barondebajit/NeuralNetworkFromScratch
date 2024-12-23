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
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update_weights(self, w, dw, regularizer=None, reg_lambda=0.0):
        """Update function for weights."""
        self.t += 1
        if self.m_w is None:
            self.m_w = np.zeros_like(dw)
            self.v_w = np.zeros_like(dw)
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        m_hat_w = self.m_w / (1 - self.beta1 ** self.t)
        v_hat_w = self.v_w / (1 - self.beta2 ** self.t)
        dw = m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        if regularizer is not None:
            dw += regularizer.gradient(w, reg_lambda)
        return w - self.learning_rate * dw

    def update_biases(self, b, db, regularizer=None, reg_lambda=0.0):
        """Update function for biases."""
        if self.m_b is None:
            self.m_b = np.zeros_like(db)
            self.v_b = np.zeros_like(db)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
        print("db shape in update_biases before assignment: ", db.shape)
        db = m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        print("db shape in update_biases after assignment: ", db.shape)
        if regularizer is not None:
            db += regularizer.gradient(b, reg_lambda)
        return b - self.learning_rate * db
