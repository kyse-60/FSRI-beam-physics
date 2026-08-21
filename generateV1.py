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

def alpha(F,E,a,b,L):
    I = (a * b **3)/ 12 # bending inertia (m^4)
    return (F * L ** 2)/ (2 * E * I) #non-dimensional load parameter

#Setting up functions:
def function(phi0):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    return quad(f, 0, phi0)[0]

def solve_phi0(alpha):
    return brentq(lambda p: function(p) - 2*np.sqrt(alpha), 1e-9, np.pi/2 - 1e-9)

# def s(phi,e,i,g,phi0):
    # constant = np.sqrt((e*i)/(2*g))
    # f = lambda theta: (1/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    # val = constant * integrate.quad(f,0,phi)[0]
    # return val

def x(phi,e,i,f,phi0):
    constant = np.sqrt((2* e * i)/f)
    val = constant * (np.sqrt(np.sin(phi0)) - np.sqrt(max(0.0, np.sin(phi0) - np.sin(phi))))
    #slen = s(phi,e,i,f,phi0)
    return val

def y(phi,e,i,g,phi0):
    constant = np.sqrt((e*i)/(2*g))
    f = lambda theta: (np.sin(theta)/ np.sqrt(max(0.0, np.sin(phi0)- np.sin(theta))))
    val = constant * integrate.quad(f,0,phi)[0]
    return 0 -val

def Gen_data(num_f,num_e,num_b,num_h,num_L,nodes, name ="output"):
    df = pd.DataFrame()
    bigstart = perf_counter()
    for f in np.linspace(F[0], F[1], num_f):
        s = pd.Series()
        s.loc['F'] = f
        for e in np.linspace(E[0], E[1], num_e):
            s.loc['E'] = e
            for b in np.linspace(B[0], B[1], num_b):
                s.loc['base'] = b
                for h in np.linspace(H[0], H[1], num_h):
                    s.loc['height'] = h
                    for l in np.linspace(L[0], L[1], num_L):
                        s.loc['length'] = l
                        start = perf_counter()
                        alph = alpha(f,e,b,h,l)
                        I = inertia(b,h)
                        phi0 = solve_phi0(alph)
                        for index in range(nodes +1):
                            phival = phi0 * index/nodes
                            coord = (float(x(phival,e,I,f,phi0)), float(y(phival,e,I,f,phi0)))
                            s.loc['coord' + str(index)] = coord
                        end = perf_counter()
                        time_beam = end - start
                        s.loc['time'] = time_beam
                        df = pd.concat([df, s.to_frame().T])
    bigend = perf_counter()
    bigtime = bigend - bigstart
    df.to_csv(name + "-time-" + str(bigtime) + "s.csv", index= False)
    print(df)


Gen_data(5,5,5,5,5,10,"outputV1.2")