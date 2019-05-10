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

def fun(x, y):
	A = 1.0
	B = 1.5
	C = -2
	
	res = []
	res.append(2 * x * (y[1] ** (1.0/B)) * y[3])
	res.append(2 * B * x * exp( (B / C) * (y[2] - A)) * y[3])
	res.append(2 * C * x * y[3])
	res.append(-2 * x * np.log(y[0]))
		
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
	return la.norm(y2n - yn) / ((2 ** p) - 1)
def runge_local_error(ys, yss, p=2):
	return la.norm(yss - ys) / (-(2 ** -p) + 1)
def get_h_tol(h, rn, p=2):
	return h * ((tol / rn) ** (1.0 / p))

def rk_with_step(f, y0, a, b, h0=1.0, s=2,p=2):
	tol = 1e-10
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

def main1():
	A = 1.0
	a = 0.0
	b = 5.0
	y0 = np.array([1.0, 1.0, 1.0, A])
	#h = (b - a) / 8
	h = 1.0 / (2 ** 1)
	print("true solution:")
	res_true = true_solution(a, b, h)
	print(res_true)

	res1 = rk(fun, y0, a, b, h)
	print("2s solution:")
	print(res1)	 
	
	res2 = rk(fun, y0, a, b, h, s=4)
	print("4s solution:")
	print(res2)	 

	#def rk_with_step(f, y0, a, b, h0=1.0, s=2,p=2):
	res3 = rk_with_step(fun, y0, a, b)
	#print("2s solution with auto step:")
	#print(res3) 
	A = 1.0
	B = 1.5
	C = -2.0
	y1 = lambda x: exp(sin(x**2))
	y2 = lambda x: exp(B * sin(x**2))
	y3 = lambda x: C * sin(x**2) + A
	y4 = lambda x: cos(x**2)
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
	i = len(res3)-1
	y = res3[i][0]
	x = res3[i][1]
	print(len(res3))
	print(x)
	print(y)
	print(np.array([y1(x), y2(x), y3(x), y4(x)]))

	print("diff (true&2s):")
	for i in range(len(res_true) - 1):
		print(la.norm(res1[i] - res_true[i]))


if __name__ == "__main__":
	main1()










#	
