# IMPORTANT IMPORTS!! 
import numpy as np 
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as integrate
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
#-----------------CONSTANTS-----------------------
L = 1 # length of the beam (m) [0.1, 2]
#-----------------INPUTS---------------------------
F = [10,100] # downwards force (N) 10-100
E = [100 * 10**9, 300 * 10**9] # Young's modulus (GPa) 100-300
A = [0.005,0.02] # side length of the beam perpendicular to the force (m) 0.005-0.02
B = [0.005,0.02] # side length of the beam parallel to the force (m) 0.005-0.02

# I = (a * b **3)/ 12 # bending inertia (m^4)

def inertia(a,b):
    return (a * b **3)/ 12

# alpha = (F * L ** 2)/ (2 * E * I) #non-dimensional load parameter
# print(alpha)

def alpha(F,E,a,b):
    I = (a * b **3)/ 12 # bending inertia (m^4)
    return (F * L ** 2)/ (2 * E * I) #non-dimensional load parameter

#Setting up functions:
def function(phi0):
    f = lambda phi: 1.0 / np.sqrt(np.sin(phi0) - np.sin(phi))
    return quad(f, 0, phi0)[0]

def solve_phi0(alpha):
    return brentq(lambda p: function(p) - 2*np.sqrt(alpha), 1e-9, np.pi/2 - 1e-9)

# phi0 = solve_phi0(alpha)
# print(phi0)

def x(phi,e,i,f):
    constant = np.sqrt((2* e * i)/f)
    val = constant * (np.sqrt(np.sin(phi0)) - np.sqrt(np.sin(phi0) - np.sin(phi)))
    return L - val

def y(phi,e,i,g):
    constant = np.sqrt((e*i)/(2*g))
    f = lambda theta: (np.sin(theta)/ np.sqrt(np.sin(phi0)- np.sin(theta)))
    val = constant * integrate.quad(f,0,phi)[0]
    return 0 -val 

# print(f'x max: {x(phi0)} y max: {y(phi0)}')
num = 100
df = pd.DataFrame()
for f in np.arange(F[0], F[1], (F[1]-F[0])/num):
    s = pd.Series()
    s.loc['F'] = f
    for e in np.arange(E[0], E[1], (E[1]-E[0])/num):
        s.loc['E'] = e
        for a in np.arange(A[0], A[1], (A[1]-A[0])/num):
            s.loc['width'] = a
            for b in np.arange(B[0], B[1], (B[1]-B[0])/num):
                s.loc['length'] = a
                alph = alpha(f,e,a,b)
                I = inertia(a,b)
                phi0 = solve_phi0(alph)
                phipart = phi0/100  # make this 10 points instead --> for future random points hmmmm
                phival = 2 * (phi0/100)
                index = 0
                while phival <= phi0:
                    coord = (x(phival,e,I,f), y(phival,e,I,f))
                    s.loc['coord' + str(index)] = coord
                    phival += phipart
                    index += 1
                df = pd.concat([df, s.to_frame().T])

print(df)
df.to_csv('output100.csv', index= False)


'''
Notes:
- not a problem to be equidistant --> if it is equidistant it can overfit
- know what quad does 
- know what brentq does
- try hpc
- randomly pick some cases and visualize (maybe look at interactive)
- graph could be just F change on the same graph 
- num should be diff for each
- want to have control over the number of poitns we generate 
- use a package to save the runtime 
- another way than the paper: see nb
- record time in both methods 
- sklearn
- next week: 
- code more general 
- learn waht optimizer we already used 
- implement the new eq as a seperate code 
- learn whats happening and understand physcis-informed nueral netowrk 
- literature review newton rafson (Advantage and disadvantage of this and the other optimizer we have )--> so go thru diff optimizers and slides 
- can use hybrid
- backprpogationn is a larger term for gradient descent 
- we will seperate into training validation and test 
- try generating max and min of alpha and run through that and see what the time difference is since all alphas have the same x and y 
- in the version we have right now --> might as wel have various Ls
- num being 5 is a good amount 
'''