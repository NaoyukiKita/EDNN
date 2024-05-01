import sys
sys.path.append("./layer")

import numpy as np
from numpy.random import default_rng

from modules.activ import Sigmoid
from modules.connection import Synapse
from layer import Layer1, Layer2
from ednn import EDNN1, EDNN2

rng = default_rng(19990603)

model1 = EDNN1(2, [8, 8, 8, 8], rng=rng)
model2 = EDNN2(2, [8, 8, 8, 8], rng=rng)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = X[:, 0] ^ X[:, 1]

model1.fit(X, y)
model2.fit(X, y)
    
y_hat1 = model1.forward(X)
y_hat2 = model2.forward(X)

for n in range(X.shape[0]):
    print(f"teacher: {y[n]}, estimate1: {y_hat1[n]}, estimate2: {y_hat2[n]}")
