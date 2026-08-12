# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
#-----------------CONSTANTS-----------------------
L = 100 # length of the beam (meters)
#-----------------INPUTS---------------------------
F = 5 # downwards force at tip of beam (newtons)
E = 3 # modulus of elasticity (n/m^2)
a = 2 # side length of the beam perpendicular to the force (meters)
b = 2 # side length of the beam parallel to the force (meters)

I = (a * b^3)/ 12 # second moment of area for square cross-section (m^4)

alpha = (F * L^2)/ (2 * E * I) #non-dimensional load parameter

