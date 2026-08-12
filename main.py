# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
#-----------------CONSTANTS-----------------------
L = 100 # length of the beam 
#-----------------INPUTS---------------------------
F = 5 # lateral force 
E = 3 # modulus of elasticity 
a = 2 # side length of the beam penpendicular to the force 
b = 2 # other side length of the beam parrallel to the force 

I = (a * b^3)/ 12

def alpha(F,L,E,I):
    return  (F * L^2)/ (2 * E * I)
alpha = alpha(F,L,E,I)



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