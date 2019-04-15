import numpy as np
import scipy.linalg as la
import random
from functools import reduce
import os
import lu
import math


def gen_a(q):
    n = random.randint(2, 7)
    long = 1000

    r1 = -100.0
    r2 = -r1
    arr = [[random.uniform(r1, r2) for i in range(n)] for j in range(n)]
    for i in range(n):
        arr[i][i] = random.uniform(q * (sum(abs(arr[j][i]) for j in range(n)) - abs(arr[i][i])), long)
    return np.array(arr)

def gen_b(n):
    r1 = -100.0
    r2 = -r1
    return np.array([random.uniform(r1, r2) for i in range(n)])
def main():
    '''
    print("Rand(r) or stdin(s)?")
    str = input()
    n = random.randint(2, 7)

    a = lu.get_matrix(n, str)
    b = lu.get_b(n, str)
    '''
    print("Rand(r) or stdin(s)?")
    str = input()
    if (str == "r"):
        q = random.randint(3, 6)
        a = gen_a(q)
        b = gen_b(a.shape[0])
    else:
        n = int(input())
        a = np.array([list(map(float, input().split())) for i in range(n)])
        b = np.array(list(map(float, input().split())))

    print(a)
    print("------")
    x = jacobi(a, b)
    print(a @ x[0])
    print(b)
    print("prior:")
    print(x[1])
    print("posterior:")
    print(x[2])

    print("_____")
    x = zeid(a, b)
    print(a @ x[0])
    print(b)
    print("prior:")
    print(x[1])
    print("posterior:")
    print(x[2])

    #print(la.norm(a))

def jacobi(a, b, eps=0.000001):
    n = a.shape[0]
    x = np.array([0.0 for i in range(n)])
    d = np.array([[a[i][i] if (i == j) else 0.0 for j in range(n)] for i in range(n)])

    d_inv = np.array([[1.0 / a[i][i] if (i == j) else 0.0 for j in range(n)] for i in range(n)])
    r = a - d

    norm = la.norm(d_inv @ r)
    k = 0
    k_prior = 0
    x0 = np.array([0.0 for i in range(n)])
    x1 = []
    while True:
        xk = d_inv @ (b - (r @ x))
        k += 1
        '''
        if (not flag_prior) and (norm < 0.5) and (la.norm(x1 - x) < eps):
            flag_prior = True
            k_prior = k
        if (not flag_prior) and (norm >= 0.5) and (la.norm(x1 - x) <= ((1-norm)/norm) * eps):
            flag_prior = True
            k_prior = k
        if (not flag_post) and (la.norm((a @ x1) - b) < eps):
            flag_post = True
            k_post = k
        '''

        if (k == 1):
            x1 = xk
            k_prior = math.ceil(np.log((eps / la.norm(x1 - x0)) * (1 - norm))/np.log(norm))

        if ((la.norm(xk - x) <= ((1-norm)/norm) * eps)):
            return (xk, k_prior, k)
        x = xk

def zeid(a, b, eps=0.000001):
    n = a.shape[0]
    x = np.array([0.0 for i in range(n)])

    #l_inv = np.array([[1.0 / a[i][i] if (i == j) else 0.0 for j in range(n)] for i in range(n)])
    u = [[a[i][j] if (i < j) else 0.0 for j in range(n)] for i in range(n)]
    l = a - u
    l_inv = la.inv(l)

    norm = la.norm(l_inv @ u)
    k = 0
    k_prior = 0
    x0 = np.array([0.0 for i in range(n)])
    x1 = []
    while True:
        xk = l_inv @ (b - (u @ x))
        k += 1
        if (k == 1):
            x1 = xk
            k_prior = math.ceil(np.log((eps / la.norm(x1 - x0)) * (1 - norm))/np.log(norm))

        if ((la.norm(xk - x) <= ((1-norm)/norm) * eps)):
            return (xk, k_prior, k)
        x = xk

if __name__ == "__main__":
    main()













    #







    #
