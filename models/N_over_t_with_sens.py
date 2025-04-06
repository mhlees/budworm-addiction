import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
from scipy.optimize import brentq
from init_rcParams import set_mpl_settings

# --- Configuration ---
# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set matplotlib settings if available
try:
    set_mpl_settings()
except ValueError:
    pass

# Set the resolution of the figures
mpl.rc("figure", dpi=330)

# --- Models ---
# Logistic growth model
def growth(N, r, K):
    """
    Calculate the growth rate of the population N.

    Parameters:
    N (float): Population size.
    r (float): Growth rate.
    K (float): Carrying capacity.

    Returns:
    float: Growth rate of the population.
    """
    return r * N * (1 - (N / K))

# Predation model
def predation(N, A, B):
    """
    Calculate the predation rate on the population N.

    Parameters:
    N (float): Population size.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.

    Returns:
    float: Predation rate on the population.
    """
    return (B * N**2) / (A**2 + N**2)

# Combined model for the system of differential equations
def total(y, t, params):
    """
    Calculate the rate of change of population N and resource r over time.

    Parameters:
    y (list): List containing the current values of N and r.
    t (float): Current time.
    params (dict): Dictionary containing the parameters of the model.

    Returns:
    list: List containing the rate of change of N and r.
    """
    N, r = y
    # Extract base parameters
    current_params = {key: value['base'] for key, value in params.items()}
    # Adjust parameters based on time-dependent adjustments
    for key, adjustments in params.items():
        for adjustment in adjustments.get('adjustments', []):
            if adjustment['start'] <= t <= adjustment['end']:
                current_params[key] = adjustment['value']
                break
    # Calculate the rate of change of r and N
    drdt = current_params['r_g'] * N - current_params['r_d'] * r
    dNdt = growth(N, r, current_params['K']) - predation(N, current_params['A'], current_params['B'])
    return [dNdt, drdt]

# Differential equation for N
def dN_dt(N, r, K, A, B) -> float:
    """
    Calculate the rate of change of population N.

    Parameters:
    N (float): Population size.
    r (float): Growth rate.
    K (float): Carrying capacity.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.

    Returns:
    float: Rate of change of population N.
    """
    return r * N * (1 - N / K) - (B * N**2) / (A**2 + N**2)

# Compute the roots of the attractor
def compute_attractor_roots(K, r, A, B):
    """
    Compute the roots of the attractor for given parameters.

    Parameters:
    K (float): Carrying capacity.
    r (float): Growth rate.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.

    Returns:
    list: List of attractor roots.
    """
    attractor_roots = []
    search_intervals = np.linspace(0, K, 100)
    for a, b in zip(search_intervals[:-1], search_intervals[1:]):
        try:
            root: float = brentq(dN_dt, a, b, args=(r, K, A, B), full_output=False) # type: ignore
            root += 1.05 # should this be done?
            if dN_dt((root), r, K, A, B) < 0 and not np.isclose(root, 0, atol=1e-3):
                attractor_roots.append(root)
        except ValueError:
            continue
    return attractor_roots

# --- Simulation Parameters ---
# Define the parameters for the simulation
params = {
    'K': {'base': 10},
    'A': {'base': 3},
    'B': {'base': 1},
    'r_g': {'base': 0.004},
    'r_d': {'base': 0.03}
}

# Initial conditions for N and r
initial_conditions = [1, 0.1]

# Time range for the simulation
t = range(0, 250)

# --- Solve ODE ---
# Solve the system of differential equations
solution = odeint(total, initial_conditions, t, args=(params,))
N_solution, r_solution = solution[:, 0], solution[:, 1]

# --- Plot Time Series ---
# Plot the time series for N and r
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, N_solution)
ax1.set_ylabel('N')
ax2.plot(t, r_solution)
ax2.set_xlabel('Time')
ax2.set_ylabel('r')
plt.tight_layout()
plt.show()

# --- Bifurcation Diagram ---
# Parameters for the bifurcation diagram
A, B = 3, 1
r_values = np.linspace(0.13, 0.8, 100)
K_values = np.linspace(3.5, 20, 100)
sols = np.full((len(K_values), len(r_values)), np.nan)

# Compute the attractor roots for different values of K and r
for k_index, K in enumerate(K_values):
    for r_index, r in enumerate(r_values):
        roots = compute_attractor_roots(K, r, A, B)
        if len(roots) == 1:
            sols[k_index, r_index] = roots[0]

# Plot the bifurcation diagram
cmap = plt.get_cmap('RdGy')
cmap.set_bad(color='#CCE8CC')
plt.figure(figsize=(8, 6))
plt.pcolormesh(K_values, r_values, sols.T, shading='auto', cmap=cmap)
plt.title(f'A = {A}, B = {B}')
plt.xlabel('K')
plt.ylabel('r')
cbar = plt.colorbar(orientation='vertical', pad=0.01, fraction=0.05)
cbar.ax.set_ylabel('N value')
plt.show()
