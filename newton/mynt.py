import lu
import numpy as np
import scipy.linalg as la
import random
from functools import reduce
import os
from math import sin, cos

def get_f():
    f1 = lambda x : sin(2*x[0] - x[1]) - 1.2*x[0] - 0.4
    f2 = lambda x : 0.8*(x[0]**2) + 1.5*(x[1]**2) - 1
    return [f1, f2]
def get_j():
    f00 = lambda x : 2*cos(2*x[0] - x[1]) - 1.2
    f01 = lambda x : -cos(2*x[0] - x[1])
    f10 = lambda x : 1.6*x[0]
    f11 = lambda x : 3*x[1]
    return [[f00, f01], [f10, f11]]
def get_x0():
    return [0.4, -0.75]

def main():
    x = nt(get_f(), get_j(), get_x0())
    print(x)
    x = mod_nt(get_f(), get_j(), get_x0())
    print(x)

def nt_r1(f, fd=None, eps=0.000001, x0=None):
    while True:
        x1 = x0 - (f(x0) / fd(x0))
        if abs(x1 - x0) < eps:
            return x1
        x1 = x

def apply(fm, x, dim):
    n = len(fm)
    if dim == 1:
        return np.array([fm[i](x) for i in range(n)])
    return np.array([[fm[i][j](x) for j in range(n)] for i in range(n)])

def nt(f, j, x0, eps=0.000001):
    xk = x0
    while True:
        jk = apply(j, xk, 2)
        fk = -apply(f, xk, 1)
        delta = lu.solve_sys_q(jk, fk)
        xk = delta + xk
        if la.norm(delta) < eps:
            return xk

def mod_nt(f, j, x0, eps=0.000001):
    j0 = apply(j, x0, 2)
    xk = x0
    while True:
        fk = -apply(f, xk, 1)
        delta = lu.solve_sys_q(j0, fk)
        xk = delta + xk
        if la.norm(delta) < eps:
            return xk

if __name__ == "__main__":
    main()












#
