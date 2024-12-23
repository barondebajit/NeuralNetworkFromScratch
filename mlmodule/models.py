#Models module for mlmodel

#Imports
import numpy as np
import pandas as pd

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.loss = None
        self.optimizer = None
        self.regularizer = None
        self.reg_lambda = 0.0
        self.initialized = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, regularizer=None, reg_lambda=0.0):
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_lambda = reg_lambda
        
        # Set optimizer and regularizer for each layer
        for layer in self.layers:
            layer.set_optimizer(optimizer)
            if regularizer is not None:
                layer.set_regularizer(regularizer, reg_lambda)

    def _validate_input(self, X, y=None):
        """Validate and normalize input data."""
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
            
        # Reshape 1D inputs
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if y is not None and len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        return X, y

    def fit(self, X, y, epochs=10, batch_size=32, verbose=1):
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                self.backward(y_batch)
                
                # Update weights
                self.update()
                
                # Calculate loss
                batch_loss = self.loss(y_batch, y_pred)
                losses.append(batch_loss)
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(losses):.4f}")

    def forward(self, X):
        i = 0
        X, _ = self._validate_input(X)
        for layer in self.layers:
            X = layer.forward(X)
            i+=1
        return X

    def backward(self, y):
        i = 0
        dA = self.loss(y, self.layers[-1].A, derivative=True)
        for layer in reversed(self.layers):
            print("Layer", i+1)
            dA = layer.backward(dA)
            i+=1

    def update(self):
        i = 0
        for layer in self.layers:
            layer.update()
            i+=1

    def predict(self, X):
        X, _ = self._validate_input(X)
        return self.forward(X)

    def evaluate(self, X, y):
        X, y = self._validate_input(X, y)
        y_pred = self.predict(X)
        return self.loss(y, y_pred)