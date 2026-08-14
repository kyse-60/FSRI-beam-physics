# THIS CODE IS OUTDATED SEE generateV!.py FOR THE NEWER VERSION --> THIS DOES NOT INCLUDE TIMESTAMPS AND WILL NOT RUN ANYTHING
# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
#-----------------CONSTANTS-----------------------
#-----------------INPUTS---------------------------
F = [10,100] # downwards force (N) 10-100
E = [100 * 10**9, 300 * 10**9] # Young's modulus (GPa) 100-300
B = [0.005,0.02] # side length of the beam perpendicular to the force (m) 0.005-0.02
H = [0.005,0.02] # side length of the beam parallel to the force (m) 0.005-0.02
L = [0.1,2] # length of the beam (m) [0.1, 2]

def inertia(a,b):
    return (a * b **3)/ 12

def alpha(F,E,a,b,L):
    I = (a * b **3)/ 12 # bending inertia (m^4)
    return (F * L ** 2)/ (2 * E * I) #non-dimensional load parameter

#Setting up functions:
def function(phi0):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    return quad(f, 0, phi0)[0]

def solve_phi0(alpha):
    return brentq(lambda p: function(p) - 2*np.sqrt(alpha), 1e-9, np.pi/2 - 1e-9)

def x(phi,e,i,f,L):
    constant = np.sqrt((2* e * i)/f)
    val = constant * (np.sqrt(np.sin(phi0)) - np.sqrt(np.sin(phi0) - np.sin(phi)))
    return L - val

def y(phi,e,i,g):
    constant = np.sqrt((e*i)/(2*g))
    f = lambda theta: (np.sin(theta)/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    val = constant * integrate.quad(f,0,phi)[0]
    return 0 -val

def Gen_data(num_f,num_e,num_b,num_h,num_L,nodes, name ="output"):
    df = pd.DataFrame()
    for f in np.arange(F[0], F[1], (F[1]-F[0])/num_f):
        s = pd.Series()
        s.loc['F'] = f
        for e in np.arange(E[0], E[1], (E[1]-E[0])/num_e):
            s.loc['E'] = e
            for b in np.arange(B[0], B[1], (B[1]-B[0])/num_b):
                s.loc['base'] = b
                for h in np.arange(H[0], H[1], (H[1]-H[0])/num_h):
                    s.loc['height'] = h
                    for l in np.arange(L[0], L[1], (L[1]-L[0])/num_L):
                        s.loc['length'] = l
                        alph = alpha(f,e,b,h,l)
                        I = inertia(b,h)
                        phi0 = solve_phi0(alph)
                        phipart = phi0/nodes  # for future: random points
                        phival = 2 * (phi0/nodes)
                        index = 0
                        while phival <= phi0:
                            coord = (x(phival,e,I,f,l), y(phival,e,I,f))
                            s.loc['coord' + str(index)] = coord
                            phival += phipart
                            index += 1
                        df = pd.concat([df, s.to_frame().T])
    df.to_csv( name + ".csv", index= False)
    print(df)


# Gen_data(5,5,5,5,5,10,"output")