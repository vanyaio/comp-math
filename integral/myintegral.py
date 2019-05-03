import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import scipy.special as special
import random
from functools import reduce
import os
from math import sin, cos, sinh, exp, cosh
import time

#integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)

def get_pol_sol(a):
	b = [a[i] for i in range(a.shape[0])]
	b.append(1.0)
	b.reverse()
	return np.roots(b)

def calc_mu(p, n, a, b):
	return np.array([ integrate.quad(lambda x: p(x)*(x**j), a, b)[0] for j in range(n)]) 
	
def ikf(f, p, a, b, n=None, x=None, mu=None):
	if x != None:
		n = x.shape[0]
	else:
		x = np.array([a + i*(b-a)/(n-1) for i in range(n)]) 
	if mu == None:
		mu = calc_mu(p, n, a, b)
	
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu)
	
	return sum(A[i] * f(x[i]) for i in range(n))	
		

def kf_gauss(f, p, a, b, n, mu=None):
	if mu == None:
		mu = calc_mu(p, 2*n, a, b)
	
	ma = np.array([[mu[j+s] for j in range(n)] for s in range(n)])
	mb = np.array([-mu[n+s] for s in range(n)])
	aw = la.solve(ma, mb)
	
	x = get_pol_sol(aw)
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu[:n])
		
	return sum(A[i] * f(x[i]) for i in range(n))	
		
def s_newton_cotse(f, p, a, b, h, m):
	res = 0
	while (a < b):
		res += ikf(f, p, a, min(a + h, b), n=m)
		a += h
	return res

def richardson(f, p, a, b, h, m, r):
	A = [[h[i]**(m+j) for j in range(r + 1)] for i in range(r + 1)]
	b = [-s_newton_cotse(f, p, a, b, h[i], m) for i in range(r + 1)]
	x = la.solve(A, b)		
	
	return sum(x[i + 1] * (h[0] ** (m + i)) for i in range(r))

def var1_main():
	eps = 1e-6
	L = 0.95
	m = 
	r = 
	while richardson 
if __name__ == "__main__":
	f = lambda x: x**2
	p = lambda x: 1.0
	a = -10
	b = 10
	n = 10
	print((integrate.quad(lambda x: f(x)*p(x), a, b))[0])
	print('--'*10)
	
	print(ikf(f, p, a, b, n=10))
	print('--'*10)

	print(kf_gauss(f, p, a, b, n=10))
	













	
