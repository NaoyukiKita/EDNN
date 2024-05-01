import numpy as np
from numpy.random import default_rng

class Linear():
    def __init__(self, in_features, out_features, sign, bias=True, rng=default_rng()):
        self.weight = rng.random([in_features, out_features]) # ~ Uni(0,1)
        if sign == "negative":
            self.weight *= -1
        
        self.bias = None
        if bias:
            self.bias = rng.random([out_features]) # ~ Uni(0,1)
            if sign == "negative":
                self.bias *= -1
        
        return
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        out = X @ self.weight

        if self.bias is not None:
            out += np.vstack([self.bias for _ in range(X.shape[0])])

        return out

class Synapse():
    def __init__(self, in_features, out_features, bias=True, rng=default_rng()):
        self.weight = rng.random([in_features, out_features]) # ~ Uni(0,1)

        # force the number of excitatory neurons to be one of inhibitory neurons or more
        self.in_exc_fs = in_features - in_features // 2 # Ceil(in_features / 2)
        self.out_exc_fs = out_features - out_features // 2 # Ceil(out_features / 2)

        self.weight[:self.in_exc_fs, self.out_exc_fs:] *= -1
        self.weight[self.in_exc_fs:, :self.out_exc_fs] *= -1
        
        self.bias = None
        if bias:
            self.bias = rng.random([2, out_features]) # ~ Uni(0,1)
            self.bias[0, self.out_exc_fs:] *= -1
            self.bias[1, :self.out_exc_fs] *= -1

        return
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        out = X @ self.weight

        if self.bias is not None:
            out += np.vstack([self.bias.sum(0) for _ in range(X.shape[0])])

        return out
