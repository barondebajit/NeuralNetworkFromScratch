#Layers module for mlmodule

#Imports
import numpy as np
import pandas as pd

class Dense:
    def __init__(self, units, input_shape=None, activation=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.W = None
        self.b = None
        self.X = None
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None
        self.dX = None
        self.dZ = None
        self.dA = None
        self.optimizer = None
        self.regularizer = None
        self.reg_lambda = 0.0
        self.initialized = False

    def build(self, input_shape):
        if not self.initialized:
            if isinstance(input_shape, tuple):
                input_dim = input_shape[-1]
            elif isinstance(input_shape, (int, np.integer)):
                input_dim = input_shape
            else:
                raise ValueError(f"Unexpected input shape: {input_shape}")
            
            self.input_shape = input_dim
            self.W = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
            self.b = np.zeros((1, self.units))
            print("b shape in build: ", self.b.shape)
            self.initialized = True

    def forward(self, X: np.ndarray):
        # Reshape input if necessary
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])
        
        if len(X.shape) > 2:
            X = X.reshape(-1, X.shape[-1])
        
        # Build layer if not initialized
        if not self.initialized:
            self.build(X.shape[-1])
            
        assert X.shape[-1] == self.W.shape[0], f"Input shape {X.shape} is not compatible with weight shape {self.W.shape}"
        self.X = X
        self.Z = np.dot(X, self.W)
        self.Z += self.b
        self.A = self.activation(self.Z) if self.activation is not None else self.Z
        return self.A

    def backward(self, dA):
        if len(dA.shape) == 1:
            dA = dA.reshape(-1, dA.shape[0])
            
        self.dA = dA
        if self.activation is not None:
            self.dZ = dA * self.activation(self.Z, derivative=True)
        else:
            self.dZ = dA
            
        self.dW = np.dot(self.X.T, self.dZ)
        try:
            print("db shape before assignment in backward: ", self.db.shape)
        except:
            print("db shape before assignment in backward: None")
        self.db = np.sum(self.dZ, axis=0, keepdims=True)
        try:
            print("db shape after assignment in backward: ", self.db.shape)
        except:
            print("db shape after assignment in backward: None")
        self.dX = np.dot(self.dZ, self.W.T)
        return self.dX

    def update(self):
        if self.optimizer is not None:
            self.W = self.optimizer.update_weights(self.W, self.dW, self.regularizer, self.reg_lambda)
            print("b shape in update before update_biases call: ", self.b.shape)
            self.b = self.optimizer.update_biases(self.b, self.db, self.regularizer, self.reg_lambda).reshape(1,-1)
            print("b shape in update after update_biases call: ", self.b.shape)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_regularizer(self, regularizer, reg_lambda=0.0):
        self.regularizer = regularizer
        self.reg_lambda = reg_lambda