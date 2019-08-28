"""
LossFunction
@author: Arthur Semione
27 Aug 2019

"""
import numpy as np
import random as rd


def L(X, y, W):
    """
      fully-vectorized implementation :
      - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
      - y is array of integers specifying correct class (e.g. 50,000-D array)
      - W are weights (e.g. 10 x 3073)
    """
    D = W.shape[0]  # number of classes
    N = W.shape[1]  # number of weights W[DxN] X[Nxn]
    n = X.shape[1]  # number examples

    delta = 1.0
    scores = W.dot(X)  # matrix [Dxn] score of each class along all examples (n)

    margins = np.zeros(int(D * n))
    margins = margins.reshape(D, n)

    Index = np.zeros(n)
    for ii in range(n):
        Index[ii] = np.argmax(y[:, ii])

    for res in range(n):
        margins[:, res] = np.maximum(0, scores[:, res] - scores[int(Index[res]), res] + delta)
        margins[int(Index[res]), res] = 0

    loss_i = np.sum(margins)
    return loss_i


"--------------------------------------- Example of this ------------------------------------------"


Weights = np.zeros(16)
for i in range(16):
    Weights[i] = rd.randint(1, 3)
Weights = Weights.reshape(4, 4)

ImageVector = np.zeros(12)
for i in range(12):
    ImageVector[i] = rd.randint(1, 4)
ImageVector = ImageVector.reshape(4, 3)

CorrectClass = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

loss = L(ImageVector, CorrectClass, Weights)
print(loss)
