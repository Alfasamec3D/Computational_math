import numpy as np
import copy

def lu_decomposition(A_in):
    n = A_in.shape[0]
    A = copy.deepcopy(A_in)
    L = np.eye(n)
    U = copy.deepcopy(A_in)
    
    for k in range(n-1):
        for i in range(k+1, n):
            if U[i,k] != 0:
                c = U[i,k] / U[k,k]
                L[i,k] = c
                U[i,k:] = U[i,k:] - c * U[k,k:]
    
    return L, U
def lu_solve(L, U, b):
    n = b.size
    y = np.zeros_like(b)
    
    # прямой ход: L y = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i,:i], y[:i])
    
    # обратный ход: U x = y
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    
    return x

A2 = np.array([[2., -1., 0.], 
               [-1., 2., -1.], 
               [1e-16, 1., -1.]])
b2 = np.array([1., 0., 0.])

L, U = lu_decomposition(A2)
x = lu_solve(L, U, b2)

print("L =\n", L)
print("U =\n", U)
print("Решение x =", x)

