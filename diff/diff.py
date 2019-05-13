import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import scipy.special as special
import random
from functools import reduce
import os
from math import sin, cos, sinh, exp, cosh, log
import math
import time
import matplotlib.pyplot as plt
import sys
import os

def rk_step(f, x0, h, y0, s=2): 
	c2 = 0.25
	a = [[0. for j in range(s)] for i in range(s)]
	c = [0. for i in range(s)]
	b = [0. for i in range(s)]
	K = [0. for i in range(s)]

	if s == 2:	
		c[1] = c2
		a[1][0] = c[1]
		b[1] = 1.0 / (2 * c2)
		b[0] = 1 - b[1]
		#K = [f(x0 + c[i] * h)] 
		#K += [np.array([y0 + h * sum(a[i][j] * K[j] for j in range(i - 1))]) for i in range(s)]
		for i in range(s):
			K[i] = f(x0 + c[i] * h, y0 + h * sum(a[i][j] * K[j] for j in range(i - 1)))
		return y0 + h * sum(b[i] * K[i] for i in range(s))

	if s == 4:
		K[0] = f(x0, y0) 
		K[1] = f(x0 + 0.5 * h, y0 + 0.5 * h * K[0])
		K[2] = f(x0 + 0.5 * h, y0 + 0.5 * h * K[1])
		K[3] = f(x0 + h, y0 + h * K[2])
		return y0 + h * ( (1.0/6)*K[0] + (1.0/3) * K[1] +(1.0/3) * K[2] + (1.0/6) * K[3])		

Counter = 0
def fun(x, y):
	A = 1.0
	B = 1.5
	C = -2
	
	res = []
	res.append(2 * x * (y[1] ** (1.0/B)) * y[3])
	res.append(2 * B * x * exp( (B / C) * (y[2] - A)) * y[3])
	res.append(2 * C * x * y[3])
	res.append(-2 * x * np.log(y[0]))
	
	global Counter
	Counter += 1	
	return np.array(res)

def rk(f, y0, a, b, h, s=2):
	y = [y0]
	x_step = a
	y_step = y0
	
	it = 1
	while x_step < b:
		y.append(rk_step(f, x_step, h, y_step, s))
		y_step = np.array(y[len(y)-1])
		x_step += h
		it += 1

	return np.array(y) #?np.array

def true_solution(a, b, h):
	A = 1.0
	B = 1.5
	C = -2.0
	
	y1 = lambda x: exp(sin(x**2))
	y2 = lambda x: exp(B * sin(x**2))
	y3 = lambda x: C * sin(x**2) + A
	y4 = lambda x: cos(x**2)
	
	#return np.array([y1(x), y2(x), y3(x), y4(x)])
	y = []
	x = a
	while x <= b:
		y.append(np.array([y1(x), y2(x), y3(x), y4(x)]))
		x += h
	
	return np.array(y)

def runge_full_error(yn, y2n, p=2):
	return la.norm(y2n - yn) / (-(2 ** -p) + 1)
def runge_local_error(ys, yss, p=2):
	return la.norm(yss - ys) / (-(2 ** -p) + 1)
def get_h_tol(h, rn, p=2, tol=1e-5):
	return h * ((tol / rn) ** (1.0 / p))

def rk_with_step(f, y0, a, b, h0=1.0, s=2,p=2):
	tol = 1e-8#-10
	y = [(y0, a)]
	x_step = a
	y_step = y0

	h = h0
	while x_step < b:
		y1 = rk_step(f, x_step, h * 0.5, y_step, s)
		yss = rk_step(f, x_step + h * 0.5, h * 0.5, y1, s)
		ys = rk_step(f, x_step, h, y_step, s)
		rn = runge_local_error(ys, yss, p)
		
		if rn > (tol * (2 ** p)):	
			h *= 0.5			
			continue

		if (tol < rn) and (rn <= (tol * (2 ** p))):
			h *= 0.5
			y_step = yss

		if (tol * (2 ** (-p-1)) <= rn) and (rn <= tol):
			y_step = ys
		
		if rn < tol * (2 ** (-p-1)):			
			h = 2 * h
			y_step = ys
		
		x_step += h
		y.append((y_step, x_step))
	
	return y

def rk_with_step_atol(f, y0, a, b, h0=1.0, s=2, p=2, rtol=1e-8):
	atol = 1e-12
	y = [(y0, a)]
	x_step = a
	y_step = y0

	h = h0
	while x_step < b:
		y1 = rk_step(f, x_step, h * 0.5, y_step, s)
		yss = rk_step(f, x_step + h * 0.5, h * 0.5, y1, s)
		ys = rk_step(f, x_step, h, y_step, s)
		rn = runge_local_error(ys, yss, p)
		
		tol = rtol * la.norm(ys) + atol
		if rn > (tol * (2 ** p)):	
			h *= 0.5			
			continue

		if (tol < rn) and (rn <= (tol * (2 ** p))):
			h *= 0.5
			y_step = yss

		if (tol * (2 ** (-p-1)) <= rn) and (rn <= tol):
			y_step = ys
		
		if rn < tol * (2 ** (-p-1)):			
			h = 2 * h
			y_step = ys
		
		x_step += h
		y.append((y_step, x_step))
	
	return y
	
def main(s=2):
	A = 1.0
	a = 0.0
	b = 5.0
	y0 = np.array([1.0, 1.0, 1.0, A])

	if s == 4:
		k0 = 3
	if s == 2:
		k0 = 10
	h = 1.0 / (2 ** k0)#10

	print("true solution:")
	res_true = true_solution(a, b, h)
	print(res_true)
	#1:
	'''
	x1 = []
	y1 = []
	x2 = []
	y2 = []
	for k in range(6):
		h1 = 1.0 / (2 ** (k0 + k))
		res = rk(fun, y0, a, b, h1, s=s)
		r = la.norm(res_true[-1] - res[-1])
		x1.append(k + k0)
		y1.append(log(r, 2))
		x2.append(k + k0)
		y2.append(-s * (k + k0))
	x1 = np.array(x1)
	y1 = np.array(y1)
	x2 = np.array(x2)
	y2 = np.array(y2)
	plt.plot(x1, y1, label='error')
	plt.plot(x2, y2, label='linear f')
	plt.legend()
	plt.show()
	'''

	#2:
	res1 = rk(fun, y0, a, b, h, s=s)
	print(str(s) + "s solution:")
	print(res1)	 
	res12 = rk(fun, y0, a, b, h * 0.5, s=s)
	print(str(s) + "s h/2 solution:")
	print(res12)	 
	rn = runge_full_error(res1[-1], res12[-1], p=s)
	h_tol = get_h_tol(h, rn, p=s)
	print("h tol: " + str(h_tol))
	
	res_h_tol = rk(fun, y0, a, b, h_tol, s=s)
	print("h tol solution:")
	print(res_h_tol)

	print("h_tol and true solution compare:")
	print(la.norm(res_h_tol[-1] - res_true[-1]))

	xa = []
	ya = []
	A = 1.0
	B = 1.5
	C = -2.0
	y1 = lambda x: exp(sin(x**2))
	y2 = lambda x: exp(B * sin(x**2))
	y3 = lambda x: C * sin(x**2) + A
	y4 = lambda x: cos(x**2)

	x_step = a
	for y_tol in res_h_tol:
		xa.append(x_step)
		ya.append(la.norm(np.array([y1(x_step), y2(x_step), y3(x_step), y4(x_step)]) - y_tol))
		x_step += h_tol		
	xa = np.array(xa)
	ya = np.array(ya)
	plt.plot(xa, ya)
	plt.show()

	#AUTO  STEP PART
	res3 = rk_with_step_atol(fun, y0, a, b, s=s, p=s)
	'''
	for i in range(len(res3) - 1):
		y = res3[i][0]
		x = res3[i][1]	
		print("*"*20)
		true_y = np.array([y1(x), y2(x), y3(x), y4(x)])
		print(y)
		print(true_y)
		print(y - true_y)	
	'''
	y = res3[-1][0]
	print(len(res3))
	print("solutuion with auto step:")
	print(y)
	#print("true solut:")
	#print(np.array([y1(x), y2(x), y3(x), y4(x)]))
	
	#1
	'''
	xa = []
	ya = []
	for sol in res3:
		xa.append(sol[1])
		ya.append(sol[0])
	xa = np.array(xa)
	ya = np.array(ya)
	plt.plot
	'''
	#2
	xa = []
	ya = []
	for i in range(len(res3) - 1):
		xnow = res3[i][1]
		xnext = res3[i+1][1]
		xa.append(xnow)
		ya.append(xnext - xnow)
	xa = np.array(xa)
	ya = np.array(ya)
	plt.plot(xa, ya)
	plt.show()

	#3
	xa = []
	ya = []
	for sol in res3:
		x_step = sol[1]
		xa.append(x_step)
		ya.append(la.norm(np.array([y1(x_step), y2(x_step), y3(x_step), y4(x_step)]) - sol[0]))
	
	xa = np.array(xa)
	ya = np.array(ya)
	plt.plot(xa, ya)
	plt.show()

	#4:
	xa = []
	ya = []
	global Counter
	rtol_list = [1e-6, 1e-7, 1e-8]
	for rtol in rtol_list:
		Counter = 0
		res = rk_with_step_atol(fun, y0, a, b, s=s, p=s, rtol=rtol)
		xa.append(log(rtol, 2))		
		ya.append(log(Counter, 2))

	plt.plot(np.array(xa), np.array(ya))
	plt.show()

if __name__ == "__main__":
	if len(sys.argv) == 1:
		main(s=2)
		print("*"*20)
		main(s=4) 	
		os._exit(0)
	
	if sys.argv[1] == '2':
		main(s=2)
	if sys.argv[1] == '4':
		main(s=4)








#	
