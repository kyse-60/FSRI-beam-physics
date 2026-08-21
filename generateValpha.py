# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
import pandas as pd
from time import perf_counter
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
#-----------------CONSTANTS-----------------------
#-----------------INPUTS---------------------------
F = [100,1000] # downwards force (N) 10-100
E = [100 * 10**9, 300 * 10**9] # Young's modulus (GPa) 100-300
B = [0.01,0.02] # side length of the beam perpendicular to the force (m) 0.005-0.02
H = [0.01,0.02] # side length of the beam parallel to the force (m) 0.005-0.02
L = [0.2,1] # length of the beam (m) [0.1, 2]

def inertia(a,b):
    return (a * b **3)/ 12

def alpha(F,E,b,h,L):
    I = (b * h **3)/ 12 # bending inertia (m^4)
    return (F * L ** 2)/ (2 * E * I) #non-dimensional load parameter

#Setting up functions:
def function(phi0):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    return quad(f, 0, phi0)[0]

def solve_phi0(alpha):
    return brentq(lambda p: function(p) - 2*np.sqrt(alpha), 1e-9, np.pi/2 - 1e-9)

def s(phi,alpha,phi0):
    constant = 1/(2 * np.sqrt(alpha))
    f = lambda theta: (1/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    val = constant * integrate.quad(f,0,phi)[0]
    return val

def xratio(phi,alpha,phi0):
    constant = 1/ np.sqrt(alpha)
    val = constant * (np.sqrt(np.sin(phi0)) - np.sqrt(np.sin(phi0) - np.sin(phi)))
    #slen = s(phi,alpha,phi0)
    return val

def yratio(phi,alpha,phi0):
    constant = 1/(2 * np.sqrt(alpha))
    f = lambda theta: (np.sin(theta)/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    val = constant * integrate.quad(f,0,phi)[0]
    return 0 -val

# Something to note about this is that this essentially is like "normalizing" all our values 
# we return the coordinates as a ratio of L so when visualizing all will be the same looking
# even if not from alphas that represent different L values 

def Gen_data(nodes,numiter, name ="output"):
    df = pd.DataFrame()
    bigstart = perf_counter()
    largestalpha = alpha(F[1],E[0],B[0],H[0],L[1])
    smallestalpha = alpha(F[0],E[1],B[1],H[1],L[0])
    for alph in np.arange(smallestalpha, largestalpha, (largestalpha- smallestalpha)/numiter):
        start = perf_counter()
        s = pd.Series()
        s.loc["alpha"] = alph
        phi0 = solve_phi0(alph)
        for index in range(nodes +1):
            if index == nodes:
                phival = phi0
            else:
                phival = phi0 * index/nodes
            coord = (float(xratio(phival,alph,phi0)), float(yratio(phival,alph,phi0)))
            s.loc['coord' + str(index)] = coord
        end = perf_counter()
        time_beam = end - start
        s.loc['time'] = time_beam
        df = pd.concat([df, s.to_frame().T])
    bigend = perf_counter()
    bigtime = bigend - bigstart
    df.to_csv(name + "-time-" + str(bigtime) + "s.csv", index= False)
    print(df)

Gen_data(10,100,"outputValpha(100).3")