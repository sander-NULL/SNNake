'''
Created on Apr 25, 2024

@author: sander
'''

import numpy as np
from numpy import dtype

#dimensions of playing field in blocks
FIELD_WIDTH = 20
FIELD_HEIGHT = 20

def sigmoid(x):
    return 1/(1+np.exp(-x))

# create version of sigmoid for vectors and matrices
vsigmoid = np.vectorize(sigmoid)

def normalize(inputvec):
    out = np.copy(inputvec).astype(float)
    out[0] = 2/FIELD_WIDTH * inputvec[0] - 1   #maps it to [-1, 1)
    out[1] = 1 - 2/FIELD_HEIGHT * inputvec[1]  #maps it to (-1, 1]
    out[4] = 2/FIELD_WIDTH * inputvec[4] - 1   #maps it to [-1, 1)
    out[5] = 1 - 2/FIELD_HEIGHT * inputvec[5]  #maps it to (-1, 1]
    return out