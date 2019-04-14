import numpy as np
import scipy.linalg as la
import random
from functools import reduce
import os
import lu


def main():
    print("Rand(r) or stdin(s)?")
    str = input()
    n = random.randint(2, 7)

    a = lu.get_matrix(n, str)
    b = lu.get_b(n, str)

    x = jacobi(a, b)
    print(a @ x[0])
    print(b)
    print(x[1])

    #print(la.norm(a))

def jacobi(a, b):
    n = a.shape[0]
    x = np.array([0.0 for i in range(n)])
    d = np.array([[a[i][i] if (i == j) else 0.0 for j in range(n)] for i in range(n)])

    d_inv = np.array([[1.0 / a[i][i] if (i == j) else 0.0 for j in range(n)] for i in range(n)])
    r = a - d

    norm = la.norm(d_inv @ r)
    #print(norm)
    eps = 0.000001
    k = 0
    while True:
        x1 = d_inv @ (b - (r @ x))
        k += 1
        if (norm < 0.5) and (la.norm(x1 - x) < eps):
            return (x1, k)
        if (norm >= 0.5) and (la.norm(x1 - x) <= ((1-norm)/norm) * eps):
            return (x1, k)
        x = x1

if __name__ == "__main__":
    main()
