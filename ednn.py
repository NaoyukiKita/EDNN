import sys
sys.path.append("./layer")

import numpy as np
from numpy.random import default_rng

from modules.activ import Sigmoid
from modules.connection import Synapse
from layer import Layer1, Layer2

class EDNN1():
    def __init__(self, in_dim, hidden_dims, bias=True, lr=0.8, decay=0.9, max_iter=100, rng=default_rng()):
        self.max_iter = max_iter
        self.layers = []

        self.layers.append(Layer1(in_dim, hidden_dims[0], bias, lr, decay)) # input layer
        for la in range(len(hidden_dims)-1): # hidden layers
            self.layers.append(Layer1(hidden_dims[la], hidden_dims[la+1], bias, lr, decay))
        self.layers.append(Layer1(hidden_dims[-1], 1, bias, lr, decay=1.0)) # output layer
    
    def forward(self, X):
        excit, inhib = X.copy(), X.copy()
        for layer in self.layers:
            excit, inhib = layer.forward(excit, inhib)
        excit_out = excit.reshape(-1,) # shape (N, 1) -> (N,)
        return excit_out
    
    def update(self, residual):
        for layer in self.layers:
            layer.update(residual)
        
        return
    
    def fit(self, X, y):
        if len(y.shape) == 2:
            y = y.reshape(-1,)

        self.losses = []
        for iter in range(self.max_iter):
            y_hat = self.forward(X)
            residual = y - y_hat
            self.update(residual)

            loss = np.sum(residual ** 2) / 2
            self.losses.append(loss)
        
        return self

class EDNN2():
    def __init__(self, in_dim, hidden_dims, bias=True, lr=0.8, decay=0.9, max_iter=100, rng=default_rng()):
        self.max_iter = max_iter
        self.layers = []

        # [!] `in_features` of the first layer is 2*`in_dim`
        self.layers.append(Layer2(2*in_dim, hidden_dims[0], bias, lr, decay))
        for la in range(len(hidden_dims)-1): # hidden layers
            self.layers.append(Layer2(hidden_dims[la], hidden_dims[la+1], bias, lr, decay))
        self.layers.append(Layer2(hidden_dims[-1], 1, bias, lr, decay=1.0)) # output layer
    
    def forward(self, X: np.array):
        X = np.hstack([X, X])
        for layer in self.layers:
            X = layer.forward(X)
        out = X.reshape(-1,) # shape (N, 1) -> (N,)

        return out
    
    def update(self, residual):
        for layer in self.layers:
            layer.update(residual)
        
        return

    def fit(self, X, y):
        if len(y.shape) == 2:
            y = y.reshape(-1,)

        self.losses = []
        for iter in range(self.max_iter):
            y_hat = self.forward(X)
            residual = y - y_hat
            self.update(residual)

            loss = np.sum(residual ** 2) / 2
            self.losses.append(loss)
        
        return self
