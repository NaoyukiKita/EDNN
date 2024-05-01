import numpy as np
from numpy.random import default_rng

from modules.activ import Sigmoid
from modules.connection import Synapse, Linear

class Layer1():
    def __init__(self, in_features, out_features, bias=True, lr=0.8, decay=0.9, rng=default_rng()):
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias

        self.lr = lr
        self.decay = decay
        
        # excitatory synapses, which connect homogeneous neurons
        self.excit2excit = Linear(in_features, out_features, "positive", bias, rng)
        self.inhib2inhib = Linear(in_features, out_features, "positive", bias, rng)

        # inhibitory synapses, which connect heterogeneous neurons
        self.inhib2excit = Linear(in_features, out_features, "negative", bias, rng)
        self.excit2inhib = Linear(in_features, out_features, "negative", bias, rng)

        # activation function
        self.activ = Sigmoid(a=5.0)
        
    def forward(self, excit_in, inhib_in):
        # output of previous layer
        self.excit_in = excit_in
        self.inhib_in = inhib_in

        # total input
        self.excit_total = self.excit2excit(excit_in) + self.inhib2excit(inhib_in)
        self.inhib_total = self.inhib2inhib(inhib_in) + self.excit2inhib(excit_in)

        # output
        excit_out = self.activ(self.excit_total)
        inhib_out = self.activ(self.inhib_total)

        return excit_out, inhib_out
    
    def update(self, residual: np.array):
        # differential
        diff_excit_total = self.activ.diff(self.excit_total)
        diff_inhib_total = self.activ.diff(self.inhib_total)

        # update weight
        dw = np.einsum("n,ni,nj->nij", residual, np.abs(self.excit_in), diff_excit_total).mean(0)
        self.excit2excit.weight += self.lr * self.decay * dw

        dw = np.einsum("n,ni,nj->nij", residual, np.abs(self.inhib_in), diff_inhib_total).mean(0)
        self.inhib2inhib.weight -= self.lr * self.decay * dw

        dw = np.einsum("n,ni,nj->nij", residual, np.abs(self.excit_in), diff_inhib_total).mean(0)
        self.excit2inhib.weight -= self.lr * self.decay * dw

        dw = np.einsum("n,ni,nj->nij", residual, np.abs(self.inhib_in), diff_excit_total).mean(0)
        self.inhib2excit.weight += self.lr * self.decay * dw

        if not self.bias: return

        # update bias
        db = np.einsum("n,nj->nj", residual, diff_excit_total).mean(0)
        self.excit2excit.bias += self.lr * self.decay * db
        self.inhib2excit.bias += self.lr * self.decay * db

        db = np.einsum("n,nj->nj", residual, diff_inhib_total).mean(0)
        self.inhib2inhib.bias -= self.lr * self.decay * db
        self.excit2inhib.bias -= self.lr * self.decay * db

class Layer2():
    def __init__(self, in_features, out_features, bias=True, lr=0.8, decay=0.9, rng=default_rng()):
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        
        self.lr = lr
        self.decay = decay

        self.synapse = Synapse(in_features, out_features, bias, rng)
        self.activ = Sigmoid(a=5.0)

    def forward(self, X: np.array):
        # output of previous layer
        self.prev_out = X.copy()

        # total input of activation function
        self.total_in = self.synapse.forward(X)

        return self.activ(self.total_in)
    
    def update(self, residual: np.array):
        # differential
        diff_total_in = self.activ.diff(self.total_in)

        # update weight
        dw = np.einsum("n,ni,nj->nij", residual, np.abs(self.prev_out), diff_total_in).mean(0)
        dw[:, self.synapse.out_exc_fs:] *= -1
        self.synapse.weight += self.lr * self.decay * dw

        if not self.bias: return

        # update bias
        db = np.einsum("n,nj->nj", residual, diff_total_in).mean(0)
        db[self.synapse.out_exc_fs:] *= -1
        self.synapse.bias[0] += self.lr * self.decay * db
        self.synapse.bias[1] += self.lr * self.decay * db
        