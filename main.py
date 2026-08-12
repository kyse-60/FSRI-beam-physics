# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
#-----------------CONSTANTS-----------------------
L = 100 # length of the beam 
#-----------------INPUTS---------------------------
F = 5 # lateral force 
E = 3 # modulus of elasticity 
a = 2 # side length of the beam penpendicular to the force
b = 2 # other side length of the beam parrallel to the force

I = (a * b^3)/ 12

alpha = (F * L^2)/ (2 * E * I)

