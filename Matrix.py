"""
np.maximum()
Inicializacao de vetores e matrizes np.zero()

"""
import numpy as np
import random as rd

""" Cria vetor de zeros """
W = np.zeros(16)

""" Inicializa valores aleatorios na matriz """
for i in range(16):
    W[i] = rd.randint(1, 9)

""" Altera dimensao do vetor [1x16] > [4x4] """
W = W.reshape(4, 4)

x = np.zeros(12)
for i in range(12):
    x[i] = rd.randint(1, 9)

x = x.reshape(4, 3)
scores = W.dot(x)


margins = np.zeros(int(4 * 3))
margins = margins.reshape(4, 3)

for i in range(3):
    margins[:, i] = np.maximum(0, scores[:, i] - 5)

CorrectClass = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
CorrectClass = CorrectClass.transpose()

y = np.zeros(3)
for ii in range(3):
    y[ii] = np.argmax(CorrectClass[:, ii])
print(y)



""" np.maximum():
    - Compara dois vetores e retorna um outro vetor com os maiores valores
    
>>> np.maximum(0, [-2, 5, 3])
[0 5 3]

"""

