# Refactored budworm system with bifurcation, interactive widgets, and multiple visualization modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import warnings

# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Core dynamics ---
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
    return r * N * (1 - N / K)

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

def net_dynamics(N, r, K, A, B):
    """
    Calculate the net dynamics of the population N.

    Parameters:
    N (float): Population size.
    r (float): Growth rate.
    K (float): Carrying capacity.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.

    Returns:
    float: Net change in population size.
    """
    return growth(N, r, K) - predation(N, A, B)

# --- Root finding and classification ---
def get_roots(func, ax, plot=True):
    """
    Find the roots of the function and classify them as attractors or repellors.

    Parameters:
    func (function): The function for which to find roots.
    ax (matplotlib.axes.Axes): The axes on which to plot the roots.
    plot (bool): Whether to plot the roots.

    Returns:
    list: List of attractor roots.
    """
    x_guesses = [-1, 0, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]
    result = root(func, x_guesses, method='broyden1')
    roots = result.x[np.isclose(result.fun, 0, atol=1e-6)]
    roots = np.unique(np.round(roots, 2))  # Clean up duplicates

    attractors = []
    for root_val in roots:
        stability = func(root_val + 0.01)
        if stability < 0:
            if plot:
                ax.scatter(root_val, func(root_val), color='black', s=80, label='Attractor')
            attractors.append(root_val)
        else:
            if plot:
                ax.scatter(root_val, func(root_val), facecolors='none', edgecolors='black', s=80, label='Repellor')
    return attractors

# --- Interactive plot ---
def interactive_budworm(r, K, A, B):
    """
    Create an interactive plot of the budworm population dynamics.

    Parameters:
    r (float): Growth rate.
    K (float): Carrying capacity.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.
    """
    N = np.linspace(0, 10, 200)
    G = growth(N, r, K)
    P = predation(N, A, B)
    T = G - P

    fig, ax = plt.subplots()
    ax.plot(N, G, label='Growth')
    ax.plot(N, P, label='Control', color='crimson')
    ax.plot(N, T, label='Net Change', color='purple')
    ax.axhline(0, linestyle='--', color='black', alpha=0.5)
    ax.set_ylim(-0.5, 2)
    get_roots(lambda n: net_dynamics(n, r, K, A, B), ax)
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('dN')
    plt.show()

# Widget interface
from ipywidgets import FloatSlider

# Create sliders for interactive plot
A_slider = FloatSlider(min=0.25, max=3, step=0.25, value=1.0)
B_slider = FloatSlider(min=0.25, max=3, step=0.25, value=1.0)
from ipywidgets import interactive

# Create interactive plot
interactive_plot = interactive(interactive_budworm, r=(0, 1, 0.025), K=(0, 25, 1), A=A_slider, B=B_slider)
interactive_plot

# --- Phase plot using Euler's method ---
def simulate_growth(r_values, K, dt=0.01, max_t=30):
    """
    Simulate the growth of the population using Euler's method.

    Parameters:
    r_values (list): List of growth rates.
    K (float): Carrying capacity.
    dt (float): Time step.
    max_t (float): Maximum time.

    Returns:
    None
    """
    time = np.arange(0, max_t, dt)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for r in r_values:
        N_vals = np.zeros_like(time)
        N_vals[0] = 0.1
        for i in range(1, len(time)):
            N_vals[i] = N_vals[i-1] + growth(N_vals[i-1], r, K) * dt
        mu_list = [growth(x, r, K) for x in np.linspace(0, K, len(time))]
        axs[0].plot(np.linspace(0, K, len(time)), mu_list, label=f'r={r}')
        axs[1].plot(time, N_vals, label=f'r={r}')

    axs[0].set_xlabel('N')
    axs[0].set_ylabel('dN')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('N')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# simulate_growth([0.3, 0.45, 0.7], K=10)

# --- Predator simulation ---
def simulate_control(B=1, A_values=[0.5, 1, 1.5], dt=0.01, max_t=30):
    """
    Simulate the control of the population by predators.

    Parameters:
    B (float): Maximum predation rate.
    A_values (list): List of half-saturation constants.
    dt (float): Time step.
    max_t (float): Maximum time.

    Returns:
    None
    """
    p = np.linspace(0, 5, 200)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for A in A_values:
        pred = lambda p_val: predation(p_val, A, B)
        pred_vals = pred(p)
        axs[0].plot(p, pred_vals, label=f'A={A}')

        time = np.arange(0, max_t, dt)
        P_vals = np.zeros_like(time)
        P_vals[0] = 0.1
        for i in range(1, len(time)):
            P_vals[i] = P_vals[i-1] + pred(P_vals[i-1]) * dt
        axs[1].plot(time, P_vals, label=f'A={A}')

    axs[0].axhline(y=B, linestyle='--', color='purple')
    axs[0].text(0.1, B - 0.05, 'B', color='purple')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('dC')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Control')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# simulate_control()

# --- Quiver visualization ---
def quiver_plot(r=0.5, K=10, A=1, B=1):
    """
    Create a quiver plot of the system.

    Parameters:
    r (float): Growth rate.
    K (float): Carrying capacity.
    A (float): Half-saturation constant.
    B (float): Maximum predation rate.

    Returns:
    None
    """
    N = np.linspace(0, 12, 13)
    N1, N2 = np.meshgrid(N, N)
    U = predation(N1, A, B)
    V = growth(N2, r, K)
    plt.quiver(N1, N2, U, V, color='r')
    plt.xlabel('Control')
    plt.ylabel('Growth')
    plt.title('Quiver Plot of the System')
    plt.grid(True)
    plt.show()

# quiver_plot()
