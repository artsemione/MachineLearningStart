"""
Nearest Neighbor Classifier (CS231n)

23 Aug 2019
Author: CS231

"""


import numpy as np
import os
import pickle


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, x, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        self.Xtr = x
        self.Ytr = y

    def predict(self, x):
        #num_test = x.shape[0]
        num_test = 50
        # lets make sure that the output type matches the input type (same dimension)
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - x[i, :]), axis=1)  # x[] = Image to be labeled
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.Ytr[min_index]  # predict the label of the nearest example

        return Ypred


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  #
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

''' numpy.shape():
        The shape attribute for numpy arrays returns the dimensions of the array. 
        If Y has n rows and m columns, then Y.shape is (n,m). 
        Y.shape[0] = n   Y.shape[1] = m
        
    numpy.reshape(n, m):
        Transform an array of an dimension into another array of dimension
        of your choice, with the same number of elements.
        
    EXAMPLE:
        Original array reshaped to 3D (.reshape(2, 2, 2) ) : 
        [[[0 1]
        [2 3]]

        [[4 5]
        [6 7]]]
'''
''' open():
        returns a file object: open(filename, mode).
        -rb: reading binary
'''
''' os.path.join():
        join paths from the OS, exemplo:
        print(os.path.join('cifar-10-batches-py', 'data_batch_1'))
        cifar-10-batches-py\data_batch_1 
'''
''' numpy.zeros(shape, dtype=float, order='C'):
        - shape : int or tuple of ints (length)
        Return a new array of given shape and type, filled with zeros.
        
        ListOfNumbers = [1, 3, 4, 5] > can be changed 
        TuplesOfNumber = (1, 3, 4, 5) > cannot be changed by code
'''
'''  range():
        A função range() irá imediatamente gerar uma lista contendo um bilhão de inteiros e alocar essa lista na memória. 
        Com a função xrange() cada um dos inteiros será gerado de uma vez, economizando memória e tempo de startup.
        
        >>> x = range(0, 10)
        >>> print x
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> y = xrange(0, 10)
        >>> print y
            xrange(10)  
'''
'''   np.sum(arr, axis, dtype, out, keepdims = True/False):
        axis : axis along which we want to calculate the sum value. 
            axis = 0 means along the column. 
            axis = 1 means working along the row.
        
        arr = [[14, 17, 12, 33, 44],    
        [15, 6, 27, 8, 19],   
        [23, 2, 54, 1, 4,]]   
   
        Sum of arr(axis = 0) :  [52 25 93 42 67]
        Sum of arr(axis = 1) :  [120  75  84]
        Sum of arr (keepdimension is True): 
            [[120]
             [75]
             [84]]
'''