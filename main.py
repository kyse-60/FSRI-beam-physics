# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
#-----------------CONSTANTS-----------------------
L = 100 # length of the beam 
#-----------------INPUTS---------------------------
F = 1000 # lateral force 
E = 3 # modulus of elasticity 
a = 1 # side length of the beam penpendicular to the force
b = 1 # other side length of the beam parrallel to the force

I = (a * b^3)/ 12

alpha = (F * L^2)/ (2 * E * I)

def solve_phi0(alpha):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    I = lambda phi0: quad(f, 0, phi0)[0]
    solve = lambda p: I(p) - 2*np.sqrt(alpha)
    return brentq(solve, 0, np.pi/2)

def x(phi0)
