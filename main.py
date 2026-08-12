# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
#-----------------CONSTANTS-----------------------
L = 1 # length of the beam (m)
#-----------------INPUTS---------------------------
F = 100 # downwards force (N) 1-100
E = 3 # Young's modulus (GPa) 50-300
a = 0.01 # side length of the beam perpendicular to the force (m) 0.005-0.02
b = 0.01 # side length of the beam parallel to the force (m) 0.005-0.02

I = (a * b^3)/ 12 # bending inertia (m^4)

alpha = (F * L^2)/ (2 * E * I) #non-dimensional load parameter


#Setting up functions:
def solve_phi0(alpha):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    integral = lambda phi0: quad(f, 0, phi0)[0]
    solve = lambda p: integral(p) - 2*np.sqrt(alpha)
    return brentq(solve, 0, np.pi/2)

phi0 = solve_phi0(alpha)

def x(phi):
    constant = np.sqrt((2* E * I)/F)
    return constant * (np.sqrt(np.sin(phi0)) - np.sqrt(np.sin(phi0) - np.sin(phi)))

def y(phi):
    constant = np.sqrt((E*I)/(2*F))
    f = lambda theta: (np.sin(theta)/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    return constant * integrate.quad(f,0,phi)

print(f'x max: {x(phi0)} y max: {y(phi0)}')

for f in range F