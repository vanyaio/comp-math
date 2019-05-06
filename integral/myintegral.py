import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import scipy.special as special
import random
from functools import reduce
import os
from math import sin, cos, sinh, exp, cosh
import math
import time

#integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)

def get_pol_sol(a):
	b = [a[i] for i in range(a.shape[0])]
	b.append(1.0)
	b.reverse()
	return np.roots(b)

def calc_mu(p, n, a, b):
	return np.array([integrate.quad(lambda x: p(x)*(x**j), a, b)[0] for j in range(n)]) 
	
def icf(f, p, a, b, n=None, x=None, mu=None):
	if x != None:
		n = x.shape[0]
	else:
		x = np.array([a + i*(b-a)/(n-1) for i in range(n)]) 
	if mu == None:
		mu = calc_mu(p, n, a, b)
	
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu)
	
	return sum(A[i] * f(x[i]) for i in range(n))	
		

def cf_gauss(f, p, a, b, n, mu=None):
	if mu == None:
		mu = calc_mu(p, 2*n, a, b)
	
	ma = np.array([[mu[j+s] for j in range(n)] for s in range(n)])
	mb = np.array([-mu[n+s] for s in range(n)])
	aw = la.solve(ma, mb)
	
	x = get_pol_sol(aw)
	xs = np.array([[x[i]**s for i in range(n)] for s in range(n)])
	A = la.solve(xs, mu[:n])
		
	return sum(A[i] * f(x[i]) for i in range(n))	
		
def s_newton_cotse(f, p, a, b, h, m, meth):
	res = 0
	while (a < b):
		res += meth(f, p, a, min(a + h, b), n=m)
		a += h
	return res

def richardson(f, p, a, b, h, m, r, meth):
	A = [[h[i]**(m+j) for j in range(r + 1)] for i in range(r + 1)]
	for i in range(r + 1):
		A[i][0] = -1.0

	b = [-s_newton_cotse(f, p, a, b, h[i], 3, meth) for i in range(r + 1)]
	x = la.solve(A, b)		
	
	return sum(x[i + 1] * (h[0] ** (m + i)) for i in range(r))

def aitken(f, p, a, b, h, L, m, meth):
	h1 = h
	h2 = h1 * L
	h3 = h2 * L
	s1 = s_newton_cotse(f, p, a, b, h1, m, meth) 
	s2 = s_newton_cotse(f, p, a, b, h2, m, meth)
	s3 = s_newton_cotse(f, p, a, b, h3, m, meth)
	#if abs(s2 - s1) < 1e-8 or ((s3 - s2) / (s2 - s1)) < 0:
		#return -1337
		
	return -np.log((s3 - s2) / (s2 - s1)) / np.log(1/L)	


def eps_from_step_with_icf(f, p, a, b, h, meth):
	eps = 1e-6
	L = 0.9
	m = 3 
	r = 5
	aitken_arr = []
	while True:
		m = aitken(f, p, a, b, h, L, 3, meth)
		rich = richardson(f, p, a, b, [h * (L**i) for i in range(r + 1)], m, r, meth)
		if abs(rich) < eps:
			break
		h = h*L
		try:
			aitken_arr.append(aitken(f, p, a, b, h, L, m, meth))
		except ValueError:
			aitken_arr.append(-1337)
	return (h, aitken_arr)

def get_h_opt(f, p, a, b, h, meth):
	eps = 1e-6
	L = 0.5
	#h = b - a
	ms = 3
	m = aitken(f, p, a, b, h, L, ms, meth)
	
	h1 = h
	h2 = h1 * L
	s1 = s_newton_cotse(f, p, a, b, h1, ms, meth) 
	s2 = s_newton_cotse(f, p, a, b, h2, ms, meth)
	
	return (h * ((eps*(1 - ((1/L) ** -m))/abs(s2 - s1)) ** (1.0/m)), m)

def var1(f, p, a, b, meth):
	#0:
	print("real value:")
	print(integrate.quad(lambda x: p(x)*f(x), a, b)[0])
	
	
	print("step1")
	res1 = s_newton_cotse(f, p, a, b, b - a, 3, meth)
	print("3 nodes: " + str(res1))

	print("step2")
	res1 = s_newton_cotse(f, p, a, b, b - a, 3, meth)
	h = 1.0 
	res2 = eps_from_step_with_icf(f, p, a, b, h, meth)
	h2 = res2[0]
	m2_arr = res2[1]

	print("value: " + str(s_newton_cotse(f, p, a, b, h2, 3, meth)))	
	print("required step h: " + str(h2))
	print("aitken:")
	print(m2_arr)

	print("step3")
	#res1 = s_newton_cotse(f, p, a, b, b - a, 3, meth)
	h_opt = get_h_opt(f, p, a, b, h, meth)
	print("h opt: " + str(h_opt))
	print("value for h opt: " + str(s_newton_cotse(f, p, a, b, h_opt[0], 3, meth)))
	res3 = eps_from_step_with_icf(f, p, a, b, h_opt[0], meth)
	h3 = res3[0]
	m3_arr = res3[1]

	print("value: " + str(s_newton_cotse(f, p, a, b, h3, 3, meth)))
	print("required step h: " + str(h3))
	print("aitken:")
	print(m3_arr)
	'''
	print("step3 s")
	#res1 = s_newton_cotse(f, p, a, b, b - a, 3, meth)
	h_opt1 = get_h_opt(f, p, a, b, h_opt[0], meth)
	print("h opt1: " + str(h_opt1))
	res3 = eps_from_step_with_icf(f, p, a, b, h_opt1[0], meth)
	h3 = res3[0]
	m3_arr = res3[1]

	print("value: " + str(s_newton_cotse(f, p, a, b, h3, 3, meth)))
	print("required step h: " + str(h3))
	print("aitken:")
	print(m3_arr)
	'''
 
if __name__ == "__main__":
	a = 1.5
	b = 3.3
	f = lambda x: 2*cos(2.5*x)*exp(x/3.0) + 4*sin(3.5*x)*exp(-3*x) + x
	al = 1.0/3
	bet = 0.0
	p = lambda x: ((x - a) ** -al) * ((b - x) ** -bet)

	var1(f, p, a, b, icf)
	print("PART II")
	var1(f, p, a, b, cf_gauss)





	
