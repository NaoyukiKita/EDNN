import numpy as np

class Sigmoid():
    def __init__(self, a=1.0):
        self.a = a
    
    def __call__(self, input: np.array):
        return 1 / (1 + np.exp(- self.a*input))
    
    def diff(self, input: np.array):
        e = np.exp(- self.a*input)
        return self.a * e / ((1 + e)**2)
