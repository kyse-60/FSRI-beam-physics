import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


L = 0.3          # beam length [m]
F = 4        # force [N]
E = 200 * 10**9       # Young's modulus [Pa]
B = 0.0304
H = 0.00078
I = (B * H**3)/12        # second moment of area [m^4]


def elastica_ode(s, matrix, F, E, I):

    phi, k, x, y = matrix

    dphi_ds = k
    dk_ds = -(F / (E * I)) * np.cos(phi)
    dx_ds = np.cos(phi)
    dy_ds = np.sin(phi)

    return [dphi_ds, dk_ds, dx_ds, dy_ds]

def shoot(k0, L, F, E, I):

    phi0 = 0.0
    x0 = 0.0
    y0 = 0.0

    matrix0 = [phi0, k0, x0, y0]

    # Integrate from s = 0 to s = L
    solution = solve_ivp(elastica_ode, [0, L], matrix0, args=(F, E, I), rtol=10**(-9), atol=10**(-11), dense_output=True)

    # Boundary condition at s = L:
    # k(L) = 0
    kL = solution.y[1, -1]

    return kL, solution

def residual(k0, L, F, E, I):
    kL = shoot(k0, L, F, E, I)[0]
    return kL

result = root_scalar(residual, args=(L, F, E, I), bracket=[0.0, 5.0], method='brentq')

k0 = result.root

kL, solution = shoot(k0, L, F, E, I)

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