# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
#-----------------CONSTANTS-----------------------
L = 0.3 # length of the beam (m)
#-----------------INPUTS---------------------------
F = 3.92 # downwards force (N) 1-100
E = 200 # Young's modulus (GPa) 50-300
a = 0.0304 # side length of the beam perpendicular to the force (m) 0.005-0.02
b = 0.00078 # side length of the beam parallel to the force (m) 0.005-0.02

I = (a * b **3)/ 12 # bending inertia (m^4)

alpha = (F * L ** 2)/ (2 * E * 10**9 * I) #non-dimensional load parameter
print(alpha)

#Setting up functions:
def function(phi0):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    return quad(f, 0, phi0)[0]

def solve_phi0(alpha):
    return brentq(lambda p: function(p) - 2*np.sqrt(alpha), 1e-9, np.pi/2 - 1e-9)

phi0 = solve_phi0(alpha)

def x(phi):
    constant = np.sqrt((2* E * I)/F)
    val = constant * (np.sqrt(np.sin(phi0)) - np.sqrt(np.sin(phi0) - np.sin(phi)))
    return L - val

def y(phi):
    constant = np.sqrt((E*I)/(2*F))
    f = lambda theta: (np.sin(theta)/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    val = constant * integrate.quad(f,0,phi)[0]
    return 0 -val 

print(f'x max: {x(phi0)} y max: {y(phi0)}')

#for f in range F