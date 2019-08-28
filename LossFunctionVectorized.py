import numpy as np


def L_i_vectorized(x, y, W):
    """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
    delta = 1.0
    scores = W.dot(x)

# compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)

# on y-th position scores[y] - scores[y] canceled and gave delta. We want to ignore.
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


""" np.maximum():
    - Compara dois vetores e retorna um outro vetor com os maiores valores

>>> np.maximum(0, [-2, 5, 3])
[0 5 3]

"""