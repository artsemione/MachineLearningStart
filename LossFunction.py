def L_i(x, y, W):
    """
    Unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (3073x1 in CIFAR - 10) Transposed
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    D = W.shape[0]  # number of classes, e.g. 10
    delta = 1.0
    scores = W.dot(x)
    correctClassScore = scores[y]
    loss_i = 0.0

    for j in range(D):
        if y == j:
            continue
        loss_i += max(0, scores[j] - correctClassScore + delta)
    return loss_i









