import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from time import perf_counter


F = [100,1000] # downwards force (N) 10-100
E = [100 * 10**9, 300 * 10**9] # Young's modulus (GPa) 100-300
B = [0.01,0.02] # side length of the beam perpendicular to the force (m) 0.005-0.02
H = [0.01,0.02] # side length of the beam parallel to the force (m) 0.005-0.02
L = [0.2,1] # length of the beam (m) [0.1, 2]
#I = (B * H**3)/12 # second moment of area (m^4)


def eq_system(s, matrix, F, E, I): # defines the system of equations and takes d/ds of all four items

    phi, k, x, y = matrix

    dphi_ds = k
    dk_ds = -(F / (E * I)) * np.cos(phi)
    dx_ds = np.cos(phi)
    dy_ds = np.sin(phi)

    return [dphi_ds, dk_ds, dx_ds, dy_ds]

def shoot(k0, L, F, E, I): # shooting function to guess the value of k(0), since that isn't given

    phi0 = 0.0
    x0 = 0.0
    y0 = 0.0
    initial = [phi0, k0, x0, y0] # initial values of phi, x, and y are known, k is guessed

    # solves the four-equation system and integrates from 0 to L, returns a table of values at different distances s
    solution = solve_ivp(eq_system, [0, L], initial, args=(F, E, I), dense_output=True, rtol=10**(-9), atol=10**(-11)) 

    kL = solution.y[1, -1] # we get the calculated value of k(L), ideally k(L)=0

    return kL, solution

def residual(k0, L, F, E, I): # gets the difference between k(L) for the guessed value of k0 and what it should be
    kL = shoot(k0, L, F, E, I)[0] # [0] since we don't care about solution until the guess for k0 is accurate
    return kL


def Gen_data(num_f,num_e,num_b,num_h,num_L,nodes,name="output4eq"):
    df = pd.DataFrame()
    bigstart = perf_counter()    
    for force in np.linspace(F[0], F[1], num_f):
        ser = pd.Series()
        ser.loc['F'] = force
        for elastic in np.linspace(E[0], E[1], num_e):
            ser.loc['E'] = elastic
            for base in np.linspace(B[0], B[1], num_b):
                ser.loc['base'] = base
                for height in np.linspace(H[0], H[1], num_h):
                    ser.loc['height'] = height
                    for length in np.linspace(L[0], L[1], num_L):
                        ser.loc['length'] = length
                        start = perf_counter()
                        inertia = (base * height**3)/12
                        result = root_scalar(residual, args=(length, force, elastic, inertia), bracket=[0.0, 5.0], method='brentq') # uses Brent's root finder method to narrow down on the value of k0
                        real_k0 = result.root
                        kL, solution = shoot(real_k0, length, force, elastic, inertia)
                        for index in range(nodes):
                            n = np.linspace(0,length,nodes)[index]
                            x_coord = solution.sol(n)[2]
                            y_coord = solution.sol(n)[3]
                            coord = (float(x_coord), float(-1 * y_coord))
                            ser.loc['coord' + str(index)] = coord
                        end = perf_counter()
                        time_beam = end - start
                        ser.loc['time'] = time_beam
                        df = pd.concat([df, ser.to_frame().T])
    bigend = perf_counter()
    bigtime = bigend - bigstart
    df.to_csv(name + "-time-" + str(bigtime) + "s.csv", index= False)

Gen_data(5,5,5,5,5,10)



# result = root_scalar(residual, args=(L, F, E, I), bracket=[0.0, 5.0], method='brentq') # uses Brent's root finder method to narrow down on the value of k0

# real_k0 = result.root
# kL, solution = shoot(real_k0, L, F, E, I)

# s = np.linspace(0, L, 10)

# x = solution.sol(s)[2]
# y = solution.sol(s)[3]

# print(x, y)
