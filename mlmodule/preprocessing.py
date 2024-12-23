#Preprocessing module for mlmodule

#Imports
import numpy as np
import pandas as pd

class LabelEncoder:
    def __init__(self):
        pass

    def fit(self, X):
        self.classes_ = np.unique(X)
        return self
    
    def transform(self, X):
        encoded = np.searchsorted(self.classes_, X)
        if np.any(encoded >= len(self.classes_)):
            raise ValueError("Found unseen labels during transform.")
        return encoded
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        return self.classes_[X]
    
class StandardScaler:
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        if np.any(self.std_ == 0):
            raise ValueError("Standard deviation is zero for one or more features.")
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X):
        return X * self.std_ + self.mean_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class MinMaxScaler:
    def __init__(self):
        pass
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        if np.any(self.max_ == self.min_):
            raise ValueError("All values are equal for one or more features.")
        return self
    
    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)
    
    def inverse_transform(self, X):
        return X * (self.max_ - self.min_) + self.min_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)