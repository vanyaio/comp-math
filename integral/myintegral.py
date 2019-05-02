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

def calc_mu(p, n, a, b):
	return np.array([integrate.quad(lambda x: p*(x**j), a, b) for j in range(n)]) 
	
def ikf(f, p, x, a, b, mu=None):
	n = x.shape[0]
	if mu == None:
		mu = calc_mu(p, n, a, b)
	
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu)
	for i in range(n):
		pass
		
		

def kf_gauss(f, p, a, b, n, mu=None):
	if mu == None:
		mu = calc_mu(p, 2*n, a, b)
	
	ma = np.array([[mu[j+s] for j in range(n)] for s in range(n)])
	mb = np.array([-mu[n+s] for s in range(n))#n-1?
	aa = la.solve(ma, mb)
	
	x = get_pol_sol(aa)
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu[:n])
		
	
if __name__ == "__main__":






























#
