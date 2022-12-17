from binascii import a2b_hqx
from sympy import *
from numpy import linalg as lg
import numpy as np
 
m1 = Matrix([[1,8,-7],[0,1,-3],[0,0,1]])
m_inv = m1.inv()
b = Matrix([2**0.5,3**0.5,2])
#print(m_inv*b)
# print(m1.eigenvals())
A = np.array([[-2,2,-3],[2,1,-6],[-1,-2,0]])
#print(A**3)

A2 = np.array([[4,1,1],[1,5,-1],[0,1,-3]])
#print(A2.eigenvals())
#print(A2.eigenvects())
w,v = lg.eig(A2)
print(w)
print(v)


