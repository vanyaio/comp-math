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
	if s == 4:
		K[0] = f(x0, y0) 
		K[1] = f(x0 + 0.5 * h, y0 + 0.5 * h * K[0])
		K[2] = f(x0 + 0.5 * h, y0 + 0.5 * h * K[1])
		K[3] = f(x0 + h, y0 + h * K[2])

	return y0 + h * sum(b[i] * K[i] for i in range(s))

def fun(x, y):
	A = 1.0
	B = 1.5
	C = -2
	
	res = []
	res.append(2 * x * (y[1] ** (1.0/B)) * y[3])
	res.append(2 * B * exp( (B / C) * (y[2] - A)) * y[3])
	res.append(2 * C * x * y[3])
	res.append(-2 * x * np.log(y[0]))
	
	return np.array(res)

def rk(f, y0, a, b, h, s=2):
	y = []
	x_step = a
	y_step = y0
	while x_step < b:
		y.append(rk_step(f, x_step, h, y_step, s))
		y_step = np.array(y[len(y)-1])
		x_step += h
	
	return np.array(y) #?np.array

def true_solution(a, b, h):
	A = 1.0
	B = 1.5
	C = -2
	
	y1 = lambda x: exp(sin(x**2))
	y2 = lambda x: exp(B * sin(x**2))
	y3 = lambda x: C * sin(x**2) + A
	y4 = lambda x: cos(x**2)
	
	#return np.array([y1(x), y2(x), y3(x), y4(x)])
	y = []
	x = a
	while x < b:
		y.append(np.array([y1(x), y2(x), y3(x), y4(x)]))
		x += h
	
	return np.array(y)

def main1():
	A = 1.0
	a = 0.0
	b = 5.0
	y0 = np.array([1.0, 1.0, 1.0, A])
	h = (b - a) / 8
	
	print("true solution:")
	print(true_solution(a, b, h))

	res1 = rk(fun, y0, a, b, h)
	print("2s solution: " + str(res1))	 

if __name__ == "__main__":
	main1()










#	
