import lu
import numpy as np
import scipy.linalg as la
import random
from functools import reduce
import os
from math import sin, cos, sinh, exp, cosh
import time

count_apply_f = 20*10
def get_f():
    f0=lambda x: cos(x[0]*x[1])-exp(-3*x[2])+(x[3]*x[4]*x[4])-x[5]-sinh(2*x[7])*x[8]+2*x[9]+2.0004339741653854440
    f1=lambda x: sin(x[0]*x[1])+x[2]*x[8]*x[6]-exp(-x[9]+x[5])+3*x[4]*x[4]-x[5]*(x[7]+1)+10.886272036407019994
    f2=lambda x: x[0]-x[1]+x[2]-x[3]+x[4]-x[5]+x[6]-x[7]+x[8]-x[9] - 3.1361904761904761904
    f3=lambda x: 2*cos(-x[8]+x[3]) + (x[4]/(x[2]+x[0])) - sin(x[1]*x[1]) + (cos(x[6]*x[9])**2) - x[7] - 0.170747270502230475
    f4=lambda x: sin(x[4])+2*x[7]*(x[2]+x[0])-exp(-x[6]*(-x[9]+x[5])) + 2*cos(x[1]) - (1.0/(x[3]- x[8])) - 0.368589627310127786

    f5=lambda x: exp(x[0]-x[3]-x[8])+(x[4]*x[4])/x[7] + 0.5*cos(3*x[9]*x[1]) - x[5]*x[2]+2.049108601677187511
    f6=lambda x: x[1]*x[1]*x[1]*x[6]-sin((x[9]/x[4])+x[7])+(x[0]-x[5])*cos(x[3])+x[2]- 0.7380430076202798014
    f7=lambda x: x[4]*((x[0]-2*x[5])**2)-2*sin(-x[8]+x[2])+1.5*x[3]-exp(x[1]*x[6]+x[9])+3.566832198969380904
    f8=lambda x: (7.0/x[5]) + exp(x[4]+x[3])-2*x[1]*x[7]*x[9]*x[6]+3*x[8]-3*x[0]- 8.439473450838325749
    f9=lambda x: x[9]*x[0]+x[8]*x[1]-x[7]*x[2]+sin(x[3]+x[4]+x[5])*x[6]- 0.7823809523809523809
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]
count_apply_j = 100*6
def get_j():
    cg = [[0 for i in range(10)] for j in range(10)]

    cg[0][0]=lambda x: -sin(x[0]*x[1])*x[1]
    cg[0][1]=lambda x: -sin(x[0]*x[1])*x[0]
    cg[0][2]=lambda x: 3.*exp(-(3*x[2]))
    cg[0][3]=lambda x: x[4]*x[4]
    cg[0][4]=lambda x: 2.*x[3]*x[4]
    cg[0][5]=lambda x: -1.
    cg[0][6]=lambda x: 0.
    cg[0][7]=lambda x: -2.*cosh((2*x[7]))*x[8]
    cg[0][8]=lambda x: -sinh((2*x[7]))
    cg[0][9]=lambda x: 2.
    cg[1][0]=lambda x: cos(x[0]*x[1])*x[1]
    cg[1][1]=lambda x: cos(x[0]*x[1])*x[0]
    cg[1][2]=lambda x: x[8]*x[6]
    cg[1][3]=lambda x: 0.
    cg[1][4]=lambda x: 6.*x[4]
    cg[1][5]=lambda x: -exp(-x[9]+x[5])-x[7]-0.1e1
    cg[1][6]=lambda x: x[2]*x[8]
    cg[1][7]=lambda x: -x[5]
    cg[1][8]=lambda x: x[2]*x[6]
    cg[1][9]=lambda x: exp(-x[9]+x[5])
    cg[2][0]=lambda x: 1.
    cg[2][1]=lambda x: -1.
    cg[2][2]=lambda x: 1.
    cg[2][3]=lambda x: -1.
    cg[2][4]=lambda x: 1.
    cg[2][5]=lambda x: -1.
    cg[2][6]=lambda x: 1.
    cg[2][7]=lambda x: -1.
    cg[2][8]=lambda x: 1.
    cg[2][9]=lambda x: -1.
    cg[3][0]=lambda x: -x[4]*pow(x[2]+x[0],-2.)
    cg[3][1]=lambda x: -2.*cos(x[1]*x[1])*x[1]
    cg[3][2]=lambda x: -x[4]*pow(x[2]+x[0],-2.)
    cg[3][3]=lambda x: -2.*sin(-x[8]+x[3])
    cg[3][4]=lambda x: 1./(x[2]+x[0])
    cg[3][5]=lambda x: 0.
    cg[3][6]=lambda x: -2.*cos(x[6]*x[9])*sin(x[6]*x[9])*x[9]
    cg[3][7]=lambda x: -1.
    cg[3][8]=lambda x: 2.*sin(-x[8]+x[3])
    cg[3][9]=lambda x: -2.*cos(x[6]*x[9])*sin(x[6]*x[9])*x[6]
    cg[4][0]=lambda x: 2*x[7]
    cg[4][1]=lambda x: -2.*sin(x[1])
    cg[4][2]=lambda x: 2*x[7]
    cg[4][3]=lambda x: pow(-x[8]+x[3],-2.)
    cg[4][4]=lambda x: cos(x[4])
    cg[4][5]=lambda x: x[6]*exp(-x[6]*(-x[9]+x[5]))
    cg[4][6]=lambda x: -(x[9]-x[5])*exp(-x[6]*(-x[9]+x[5]))
    cg[4][7]=lambda x: (2*x[2])+2.*x[0]
    cg[4][8]=lambda x: -pow(-x[8]+x[3],-2.)
    cg[4][9]=lambda x: -x[6]*exp(-x[6]*(-x[9]+x[5]))
    cg[5][0]=lambda x: exp(x[0]-x[3]-x[8])
    cg[5][1]=lambda x: -3./2.*sin(3.*x[9]*x[1])*x[9]
    cg[5][2]=lambda x: -x[5]
    cg[5][3]=lambda x: -exp(x[0]-x[3]-x[8])
    cg[5][4]=lambda x: 2*x[4]/x[7]
    cg[5][5]=lambda x: -x[2]
    cg[5][6]=lambda x: 0.
    cg[5][7]=lambda x: -x[4]*x[4]*pow(x[7],(-2))
    cg[5][8]=lambda x: -exp(x[0]-x[3]-x[8])
    cg[5][9]=lambda x: -3./2.*sin(3.*x[9]*x[1])*x[1]
    cg[6][0]=lambda x: cos(x[3])
    cg[6][1]=lambda x: 3.*x[1]*x[1]*x[6]
    cg[6][2]=lambda x: 1.
    cg[6][3]=lambda x: -(x[0]-x[5])*sin(x[3])
    cg[6][4]=lambda x: cos(x[9]/x[4]+x[7])*x[9]*pow(x[4],(-2))
    cg[6][5]=lambda x: -cos(x[3])
    cg[6][6]=lambda x: pow(x[1],3.)
    cg[6][7]=lambda x: -cos(x[9]/x[4]+x[7])
    cg[6][8]=lambda x: 0.
    cg[6][9]=lambda x: -cos(x[9]/x[4]+x[7])/x[4]
    cg[7][0]=lambda x: 2.*x[4]*(x[0]-2.*x[5])
    cg[7][1]=lambda x: -x[6]*exp(x[1]*x[6]+x[9])
    cg[7][2]=lambda x: -2.*cos(-x[8]+x[2])
    cg[7][3]=lambda x: 0.15e1
    cg[7][4]=lambda x: pow(x[0]-2.*x[5],2.)
    cg[7][5]=lambda x: -4.*x[4]*(x[0]-2.*x[5])
    cg[7][6]=lambda x: -x[1]*exp(x[1]*x[6]+x[9])
    cg[7][7]=lambda x: 0.
    cg[7][8]=lambda x: 2.*cos(-x[8]+x[2])
    cg[7][9]=lambda x: -exp(x[1]*x[6]+x[9])
    cg[8][0]=lambda x: -3.
    cg[8][1]=lambda x: -2.*x[7]*x[9]*x[6]
    cg[8][2]=lambda x: 0.
    cg[8][3]=lambda x: exp((x[4]+x[3]))
    cg[8][4]=lambda x: exp((x[4]+x[3]))
    cg[8][5]=lambda x: -0.7e1*pow(x[5],-2.)
    cg[8][6]=lambda x: -2.*x[1]*x[7]*x[9]
    cg[8][7]=lambda x: -2.*x[1]*x[9]*x[6]
    cg[8][8]=lambda x: 3.
    cg[8][9]=lambda x: -2.*x[1]*x[7]*x[6]
    cg[9][0]=lambda x: x[9]
    cg[9][1]=lambda x: x[8]
    cg[9][2]=lambda x: -x[7]
    cg[9][3]=lambda x: cos(x[3]+x[4]+x[5])*x[6]
    cg[9][4]=lambda x: cos(x[3]+x[4]+x[5])*x[6]
    cg[9][5]=lambda x: cos(x[3]+x[4]+x[5])*x[6]
    cg[9][6]=lambda x: sin(x[3]+x[4]+x[5])
    cg[9][7]=lambda x: -x[2]
    cg[9][8]=lambda x: x[1]
    cg[9][9]=lambda x: x[0]


    return cg

def get_x0():
    return [0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]
'''
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
'''
def print_counter_macro(x, f):
    end = time.time()
    print(x)
    print(apply(f, x[0], 1))
    print("operations: " + str(lu.Counter))

    lu.Counter = 0
    print('time: ' + str(end - start))
    print('-'*20)

start = 0
def main(x0, swap_step=4, each_step=4):
    f = get_f()
    j = get_j()

    global start

    start = time.time()
    x = nt(f, j, x0)
    print_counter_macro(x, f)

    start = time.time()
    x = mod_nt(f, j, x0)
    print_counter_macro(x, f)

    start = time.time()
    x = sw_nt(f, j, x0, swap_step)
    print('swap on step: ' + str(swap_step))
    print_counter_macro(x, f)

    start = time.time()
    x = hybr_nt(f, j, x0, each_step)
    print('swap each ' + str(each_step) + 'th step')
    print_counter_macro(x, f)

def old_main():
    '''
    f = get_f()
    j = get_j()
    x0 = get_x0()

    x = nt(f, j, x0)
    print(x)
    print(apply(f, x[0], 1))
    print(lu.Counter)
    lu.Counter = 0
    print('-'*20)
    x = mod_nt(f, j, x0)
    print(x)
    print(apply(f, x[0], 1))
    print(lu.Counter)
    lu.Counter = 0
    print('-'*20)
    swap_step = 4
    x = sw_nt(f, j, x0, swap_step)
    print('swap on step: ' + str(swap_step))
    print(x)
    print(apply(f, x[0], 1))
    print(lu.Counter)
    lu.Counter = 0
    print('-'*20)
    each_step = 4
    print('swap each ' + str(each_step) + 'th step')
    x = hybr_nt(f, j, x0, each_step)
    print(x)
    print(apply(f, x[0], 1))
    print(lu.Counter)
    lu.Counter = 0
    print('-'*20)
    '''

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
    k = 0
    while True:
        k += 1
        jk = apply(j, xk, 2)
        fk = -apply(f, xk, 1)
        lu.Counter += count_apply_f + count_apply_j
        delta = lu.solve_sys_q(jk, fk)
        xk = delta + xk
        if la.norm(delta) < eps:
            return (xk, k)

def mod_nt(f, j, x0, eps=0.000001):
    j0 = apply(j, x0, 2)
    lu.Counter += count_apply_j
    pluq = lu.get_pivot_q_lu(j0)
    xk = x0
    k = 0
    while True:
        k += 1
        fk = -apply(f, xk, 1)
        lu.Counter += count_apply_f
        delta = lu.solve_sys_q(j0, fk, pluq)
        xk = delta + xk
        if la.norm(delta) < eps:
            return (xk, k)


def sw_nt(f, j, x0, ks, eps=0.000001):
    xk = x0
    for k in range(ks):
        jk = apply(j, xk, 2)
        fk = -apply(f, xk, 1)
        lu.Counter += count_apply_f + count_apply_j
        delta = lu.solve_sys_q(jk, fk)
        xk = delta + xk
        if la.norm(delta) < eps:
            return (xk, k+1)
    x = mod_nt(f, j, xk, eps=eps)
    return (x[0], x[1] + ks)

def hybr_nt(f, j, x0, step, eps=0.000001):
    k = 0
    xk = x0
    js = 0
    pluq = 0
    while True:
        if (k%step == 0):
            js = apply(j, xk, 2)
            lu.Counter += count_apply_j
            pluq = lu.get_pivot_q_lu(js)
        k += 1
        fk = -apply(f, xk, 1)
        lu.Counter += count_apply_f
        delta = lu.solve_sys_q(js, fk, pluq)
        xk = delta + xk
        if la.norm(delta) < eps:
            return (xk, k)


if __name__ == "__main__":
    x0 = get_x0()
    main(x0, swap_step=3, each_step=4)
    x0[4] = -0.2
    main(x0, swap_step=4, each_step=5)













#
