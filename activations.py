import numpy as np
import time 
import random

def sigmoid(x, grad=False):
    if grad:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


X_sigmoid = sigmoid(np.array([1, 2, 3]), grad=True)
print(X_sigmoid)

def softmax(x):
    return np.exp(x) / np.sum(x, axis=1, keepdims=True)

x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
print(softmax(x))



