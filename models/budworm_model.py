
# models/budworm_model.py

"""
Budworm Population Dynamics Model

This module simulates the dynamics of a budworm population using a differential
equation model based on the spruce budworm outbreak literature.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def budworm_dynamics(N, t, r, K, B, a):
    """
    Differential equation for spruce budworm population dynamics.

    Parameters:
    - N (float): Budworm population at time t
    - t (float): Time
    - r (float): Intrinsic growth rate
    - K (float): Carrying capacity
    - B (float): Bird predation parameter
    - a (float): Predation response steepness

    Returns:
    - dNdt (float): Derivative of N at time t
    """
    predation = B * (N**2) / (a**2 + N**2)
    dNdt = r * N * (1 - N / K) - predation
    return dNdt


def simulate_budworm(N0=1.0, t_end=50, r=1.5, K=10.0, B=1.0, a=1.0):
    """
    Simulates the budworm population over time.

    Parameters:
    - N0 (float): Initial population
    - t_end (float): Simulation time
    - r, K, B, a: Model parameters

    Returns:
    - t (np.ndarray): Time points
    - N (np.ndarray): Population over time
    """
    t = np.linspace(0, t_end, 1000)
    N = odeint(budworm_dynamics, N0, t, args=(r, K, B, a)).flatten()
    return t, N


def plot_population(t, N):
    """
    Plots the budworm population over time.

    Parameters:
    - t (np.ndarray): Time points
    - N (np.ndarray): Population values
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, N, label="Budworm population")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Budworm Population Dynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example simulation
    t, N = simulate_budworm()
    plot_population(t, N)
