#Optimizers module for mlmodule

#Imports
import numpy as np
import pandas as pd

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t_w= 0
        self.t_b = 0

    def update_weights(self, w, dw, m_W, v_W, regularizer=None, reg_lambda=0.0):
        self.t_w += 1

        m_W = self.beta1 * m_W + (1 - self.beta1) * dw
        v_W = self.beta2 * v_W + (1 - self.beta2) * (dw ** 2)

        m_hat_w = m_W / (1 - self.beta1 ** self.t_w)
        v_hat_w = v_W / (1 - self.beta2 ** self.t_w)
        dw = m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        if regularizer is not None:
            dw += regularizer.gradient(w, reg_lambda)
        return w - self.learning_rate * dw, m_W, v_W

    def update_biases(self, b, db, m_b, v_b, regularizer=None, reg_lambda=0.0):
        self.t_b += 1

        m_b = self.beta1 * m_b + (1 - self.beta1) * db
        v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)

        m_hat_b = m_b / (1 - self.beta1 ** self.t_b)
        v_hat_b = v_b / (1 - self.beta2 ** self.t_b)

        db = m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        if regularizer is not None:
            db += regularizer.gradient(b, reg_lambda)
        
        return b - self.learning_rate * db, m_b, v_b
