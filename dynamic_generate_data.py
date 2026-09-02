"""Generate the dynamic cantilever dataset: solve, visualize, save.

Data generation follows Section 3 of the paper. Each sample is one random
tip-force history f(tau) from the general load family plus one damping value
gamma, and the resulting transverse deflection field eta(xi, tau) of the
damped Euler-Bernoulli cantilever, Eqs. (10)-(12),

    eta_tt + 2*gamma*eta_t + (1/beta1^4) eta_xxxx = 0,
    eta(0,tau) = eta_xi(0,tau) = 0,
    eta_xixi(1,tau) = 0,   eta_xixixi(1,tau) = -f(tau),

starting from rest, computed with an exact reference solver. The script
shows a few solved samples, then saves everything the training script needs
to ``dynamic_beam_dataset.npz``.

Run:
    python dynamic_generate_data.py          (then: python dynamic_train_lstm.py)
"""

import numpy as np
import matplotlib.pyplot as plt   # figures open in a window; close it to continue
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import brentq

# problem discretization (paper Section 3)
dt = 0.1                          # time step Delta-tau
tau = np.arange(0.0, 20.0 + 0.5 * dt, dt)          # 201 time stations
xi = np.linspace(0.0, 1.0, 20)                     # 20 spatial stations
n_cases = 100                                      # solved load histories
n_modes = 12                                       # modes kept in the reference
BETA1 = 1.875104068711961                          # first clamped-free eigenvalue

# ----------------------------------------------------------------------
# 1. The reference solver. The PDE separates into independent oscillators
#    q_m'' + 2*gamma*q_m' + w_m^2 q_m = c_m f(tau), one per vibration
#    mode, each integrated exactly; the field is their weighted sum.
# ----------------------------------------------------------------------
def cantilever_roots(count):
    """First `count` roots of cos(b)*cosh(b) = -1 (clamped-free beam).

    Each root b_m sets one natural frequency, w_m = (b_m / beta1)^2 in the
    normalized time of this problem. Used once, in section 2, to build the
    mode shapes and constants of the reference solver.
    """
    roots = []
    for m in range(1, count + 1):
        center = (m - 0.5) * np.pi
        roots.append(brentq(lambda b: np.cos(b) + 1.0 / np.cosh(b),
                            max(1e-8, center - 0.49 * np.pi),
                            center + 0.49 * np.pi))
    return np.array(roots)


def cantilever_mode(xi_points, root):
    """Mode shape of a clamped-free beam, normalized to 1 at the tip.

    The classical closed form built from cosh/cos/sinh/sin. Used in
    section 2 to assemble eta(xi, tau) = sum_m q_m(tau) * shape_m(xi) and
    to compute the solver constants.
    """
    sigma = (np.cosh(root) + np.cos(root)) / (np.sinh(root) + np.sin(root))
    value = (np.cosh(root * xi_points) - np.cos(root * xi_points)
             - sigma * (np.sinh(root * xi_points) - np.sin(root * xi_points)))
    tip = (np.cosh(root) - np.cos(root)
           - sigma * (np.sinh(root) - np.sin(root)))
    return value / tip


def random_load_history(rng):
    """One random tip-force history from the general load family (Sec. 3).

    A mixture of two to four sine tones (omega in [0.12, 2.5]), one chirp
    sweeping to omega <= 2.8, one or two smooth pulses, and a smooth random
    spline, ramped up softly so f(0) = f'(0) = 0 and normalized to a random
    peak between 0.6 and 1.4. Called n_cases times in section 2.
    """
    f = np.zeros_like(tau)
    for _ in range(rng.integers(2, 5)):                        # sine tones
        f += rng.uniform(0.15, 0.55) * np.sin(rng.uniform(0.12, 2.5) * tau
                                              + rng.uniform(0, 2 * np.pi))
    w0, w1 = rng.uniform(0.10, 0.7), rng.uniform(1.0, 2.8)     # one chirp
    f += rng.uniform(0.15, 0.55) * np.sin(w0 * tau
                                          + 0.5 * (w1 - w0) * tau**2 / tau[-1]
                                          + rng.uniform(0, 2 * np.pi))
    for _ in range(rng.integers(1, 3)):                        # smooth pulses
        start = rng.uniform(0.10 * tau[-1], 0.65 * tau[-1])
        width = rng.uniform(0.08 * tau[-1], 0.28 * tau[-1])
        sharp = rng.uniform(2.0, 5.0)
        f += rng.uniform(-0.8, 0.8) * 0.5 * (np.tanh(sharp * (tau - start))
                                             - np.tanh(sharp * (tau - start - width)))
    knots = np.linspace(0.0, tau[-1], 9)                       # random spline
    values = rng.normal(0.0, 0.35, len(knots))
    values[0] = 0.0
    f += CubicSpline(knots, values, bc_type="natural")(tau)

    f -= f[0]
    f *= (1.0 - np.exp(-tau / 0.55))**2                        # soft start
    return rng.uniform(0.6, 1.4) * f / max(np.abs(f).max(), 1e-10)


def solve_beam_response(load, gamma, ratios, forcing, shapes):
    """Reference deflection field for one load history.

    Integrates each oscillator q'' + 2*gamma*q' + w^2 q = c*f(tau) exactly
    for a load held constant over each time step (matrix exponential of
    the augmented state), then sums the modes. Used only in the generation
    loop of section 2; its outputs are the labels the LSTM trains on.
    """
    q = np.zeros((len(tau), len(ratios)))
    for mode, (w, c) in enumerate(zip(ratios, forcing)):
        # continuous system [q, q']' = a @ [q, q'] + [0, c] * f
        a = np.array([[0.0, 1.0, 0.0],
                      [-w**2, -2.0 * gamma, c],
                      [0.0, 0.0, 0.0]])            # third row: constant load
        step = expm(a * dt)                        # exact one-step map
        state = np.zeros(2)                        # beam starts at rest
        for k in range(len(tau) - 1):
            state = step[:2, :2] @ state + step[:2, 2] * load[k]
            q[k + 1, mode] = state[0]
    return q @ shapes.T                            # (n_time, n_xi)


# ----------------------------------------------------------------------
# 2. Generate the samples: draw the load family, solve every case.
# ----------------------------------------------------------------------
roots = cantilever_roots(n_modes)
shapes = np.column_stack([cantilever_mode(xi, r) for r in roots])  # (n_xi, 12)
xi_fine = np.linspace(0.0, 1.0, 5001)
mode_mass = np.trapezoid(np.column_stack(
    [cantilever_mode(xi_fine, r) for r in roots])**2, xi_fine, axis=0)
ratios = (roots / BETA1)**2                        # natural frequencies w_m
forcing = 1.0 / (mode_mass * BETA1**4)             # force coefficients c_m

rng = np.random.default_rng(seed=0)
loads = np.stack([random_load_history(rng) for _ in range(n_cases)])
damping = rng.uniform(0.01, 0.06, n_cases)         # one gamma per history
fields = np.stack([solve_beam_response(loads[case], damping[case],
                                       ratios, forcing, shapes)
                   for case in range(n_cases)])    # (100, n_time, n_xi)
print(f"generated {n_cases} response histories, "
      f"field shape {fields.shape} = (case, time, xi)")

# ----------------------------------------------------------------------
# 3. Visualize a few samples. First figure: three force histories and the
#    tip responses they produce (same colors). Second figure: the full
#    space-time field eta(xi, tau) for one sample.
#    The load family spans slow one-sided envelopes to strongly oscillatory
#    histories; show the most oscillatory cases (most tip-response sign
#    changes). Display choice only -- the dataset itself is untouched.
# ----------------------------------------------------------------------
tip_sign_changes = np.abs(np.diff(np.sign(fields[:, :, -1]), axis=1)).sum(axis=1)
show = list(np.argsort(tip_sign_changes)[-3:][::-1])
print(f"showing the three most oscillatory cases: {show}")

figure, (ax_load, ax_tip) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                         constrained_layout=True)
for case in show:
    line, = ax_load.plot(tau, loads[case],
                         label=f"case {case}, $\\gamma={damping[case]:.3f}$")
    ax_tip.plot(tau, fields[case, :, -1], color=line.get_color())
ax_load.set(ylabel="tip force $f(\\tau)$", title="inputs: force histories")
ax_tip.set(xlabel="normalized time $\\tau$",
           ylabel="tip deflection $\\eta(1,\\tau)$",
           title="outputs: tip of the response fields (same colors)")
ax_load.legend(fontsize=8)
figure.savefig("dynamic_samples.png", dpi=150)
print("wrote dynamic_samples.png -- close the figure window to continue")
plt.show()                                         # blocks until closed

figure, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
sample_max = np.abs(fields[show[0]]).max()
image = ax.pcolormesh(tau, xi, fields[show[0]].T / sample_max,
                      cmap="RdBu_r", vmin=-1.0, vmax=1.0)
figure.colorbar(image, ax=ax, label="$\\eta/\\eta_{\\max}$")
ax.set(xlabel="normalized time $\\tau$",
       ylabel="position $\\xi$ (0 = clamp, 1 = tip)",
       title=f"one solved sample: the full field of case {show[0]}")
figure.savefig("dynamic_field_sample.png", dpi=150)
print("wrote dynamic_field_sample.png -- close the figure window to continue")
plt.show()                                         # blocks until closed

# ----------------------------------------------------------------------
# 4. Save everything the training script needs in one file.
# ----------------------------------------------------------------------
np.savez_compressed("dynamic_beam_dataset.npz",
                    tau=tau, xi=xi, loads=loads, damping=damping,
                    fields=fields, dt=dt, n_modes=n_modes)
print("saved dynamic_beam_dataset.npz "
      f"(loads {loads.shape}, damping {damping.shape}, fields {fields.shape})")
