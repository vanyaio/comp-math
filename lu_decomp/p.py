import numpy as np
import scipy.linalg as la
import random
from functools import reduce
import os

def get_b(n, str):
    if (str == "r"):
        arr = [random.uniform(-100, 100) for i in range(n)]
    else:
        arr = [float(i) for i in input().split()]
    return np.array(arr)

def get_matrix(n, str):
    if (str == "r"):
        arr = [[random.uniform(-100, 100) for i in range(n)] for j in range(n)]
    else:
        n = int(input())
        arr = []
        for i in range(n):
            arr.append([float(i) for i in input().split()])

    return np.array(arr)

def correct(bl, str):
    print(str + " is correct\n") if bl else print(str + " is NOT correct\n")

def main():
    print("Rand(r) or stdin(s)?")
    str = input()

    n = random.randint(2, 7)
    a = get_matrix(n, str)
    pluq = get_pivot_q_lu(a)

    if (is_zero_det(pluq)):
        print("matrix has zero det")
        os._exit(-1)

    print("A: ")
    print(a)
    print("P: ")
    print(pluq[0])
    print("Q: ")
    print(pluq[3])
    print("L: ")
    print(pluq[1])
    print("U: ")
    print(pluq[2])
    correct(np.allclose(pluq[0][0] @ a @ pluq[3][0], pluq[1] @ pluq[2]), "PLUQ decomposition")


    b = get_b(n, str)
    print("b:")
    print(b)
    x = solve_sys_q(a, b, pluq)
    print("x:")
    print(x)
    correct(np.allclose(x, la.solve(a, b)), "sys solution")

    print("det:")
    print(det(a))
    correct(np.allclose(det(a), la.det(a)), "det")

    a_i = inv(a, pluq)
    print("a inversed:")
    print(a_i)
    correct(np.allclose(a_i, la.inv(a)), "inversion")

def pivot(a):
    n = (a.shape)[0]
    a1 = a.copy()
    id = [[float(i == j) for i in range(n)] for j in range(n)]
    id_det = 1
    for i in range(n):
        maxc, row = a1[i][i], i
        for j in range(i, n):
            if (a1[j][i] > maxc):
                maxc, row = a1[j][i], j
        if (i != row):
            id_det *= -1
            id[i], id[row] = id[row], id[i]
            tmp = a1[row]
            a1[row] = a1[i]
            a1[i] = tmp

    return (np.array(id), id_det)

def pivot_q(a):
    t = pivot(a.T)
    return (t[0].T, t[1])

def get_lu(a):
    n = (a.shape)[0]
    l = [[0.0 for x in range(n)] for y in range(n)]
    u = [[0.0 for x in range(n)] for y in range(n)]

    for j in range(n):

        l[j][j] = 1.0
        for i in range(j + 1):
            s1 = sum(u[k][j] * l[i][k] for k in range(i))
            u[i][j] = a[i][j] - s1

        for i in range(j, n):
            s2 = sum(u[k][j] * l[i][k] for k in range(j))
            l[i][j] = (a[i][j] - s2) / u[j][j]

    return [np.array(l), np.array(u)]

def get_pivot_lu(a):
    p = pivot(a)
    return [p] + get_lu(p[0] @ a)

def get_pivot_q_lu(a):
    q = pivot_q(a)
    return get_pivot_lu(a @ q[0]) + [q]

def has_nan(a):
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if np.isnan(a[i][j]):
                return True
    return False

def is_zero_det(pluq):
    if (has_nan(pluq[1]) or has_nan(pluq[2])):
        return True
    return False

def solve_sys_tr_l(a, b):
    x = []
    n = a.shape[0]
    x.append(b[0] / a[0][0])
    for i in range(1, n):
        x.append( (b[i] - sum(a[i][j] * x[j] for j in range(i)))/a[i][i] )
    return np.array(x)

def solve_sys_tr_r(a, b):
    x = []
    n = a.shape[0]
    x = [0.0 for i in range(n)]
    x[n-1] = b[n-1] / a[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i+1, n)))/a[i][i]
    return np.array(x)

def solve_sys(a, b, plu=None):
    if (plu == None):
        plu = get_pivot_lu(a)

    #y = la.solve(plu[1], plu[0][0] @ b)
    y = solve_sys_tr_l(plu[1], plu[0][0] @ b)
    #return la.solve(plu[2], y)
    return solve_sys_tr_r(plu[2], y)

def solve_sys_q(a, b, pluq=None):
    if (pluq == None):
        pluq = get_pivot_q_lu(a)
    '''
    y = la.solve(pluq[1], pluq[0][0] @ b)
    x = la.solve(pluq[2], y)
    return pluq[3][0] @ x
    '''
    y = solve_sys_tr_l(pluq[1], pluq[0][0] @ b)
    x = solve_sys_tr_r(pluq[2], y)
    return pluq[3][0] @ x

def det_of_tr(a):
    n = a.shape[0]
    res = 1
    for i in range(n):
        res *= a[i][i]
    return res

def det(a, pluq=None):
    if (pluq == None):
        pluq = get_pivot_q_lu(a)
    '''
    res = 1
    for i in range(4):
        res *= la.det(pluq[i])
    return res
    '''
    return pluq[0][1] * det_of_tr(pluq[1]) * det_of_tr(pluq[2]) * pluq[3][1]

def inv(a, pluq=None):
    if (pluq == None):
        pluq = get_pivot_q_lu(a)
    n = a.shape[0]
    res = []
    for i in range(n):
        e = [0.0 for x in range(n)]
        e[i] = 1.0
        #solve_sys_q(a, e, pluq)
        res.append(solve_sys_q(a, e, pluq))
    return np.array(res).T

main()

'''
a = get_matrix()
print(a)
print(pivot_q(a))
print(a @ pivot_q(a))
'''

'''
print(solve_sys(get_matrix(), get_b()))
print("______________")
print(solve_sys_q(get_matrix(), get_b()))
print("______________")
print(la.solve(get_matrix(), get_b()))
'''

















#
