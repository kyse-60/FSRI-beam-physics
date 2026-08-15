import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


F = 4 # downward force (N)
E = 200 * 10**9 # Young's modulus (Pa)
L = 0.3 # length of beam (m)
B = 0.0304 # base of beam cross-section (m)
H = 0.00078 # height of beam cross-section (m)
I = (B * H**3)/12 # second moment of area (m^4)


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

result = root_scalar(residual, args=(L, F, E, I), bracket=[0.0, 5.0], method='brentq') # uses Brent's root finder method to narrow down on the value of k0

real_k0 = result.root
kL, solution = shoot(real_k0, L, F, E, I)

s = np.linspace(0, L, 20)

x = solution.sol(s)[2]
y = solution.sol(s)[3]


plt.figure()
plt.plot(x, -y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Beam Deflection")
plt.grid(True)
plt.show()