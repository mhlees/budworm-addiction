import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# --- Model definitions ---

def smoking(t, y, parms):
    """
    ODE model for smoking dynamics.

    Parameters:
        t (float): Time
        y (list): List of state variables [S, r]
        parms (dict): Dictionary of model parameters:
            - k: carrying capacity
            - b_S: inhibitory strength for smoking
            - decay_r: decay rate of regulation
            - growth_r: growth rate of regulation
            - a: smoothness parameter for inhibition

    Returns:
        list: Derivatives [dS, dr]
    """
    S, r = y
    k, b_S, decay_r, growth_r, a = parms['k'], parms['b_S'], parms['decay_r'], parms['growth_r'], parms['a']
    dS = r * S * (1 - S / k) - (b_S * S**2) / (a**2 + S**2)
    dr = (growth_r * S - decay_r * r) * 0.001
    return [dS, dr]

def vaping_smoking(t, y, parms):
    """
    ODE model for smoking and vaping dynamics.

    Parameters:
        t (float): Time
        y (list): List of state variables [S, V, r]
        parms (dict): Dictionary of model parameters:
            - k, b_S, b_V, decay_r, growth_r, a

    Returns:
        list: Derivatives [dS, dV, dr]
    """
    S, V, r = y
    k, b_S, b_V, decay_r, growth_r, a = parms['k'], parms['b_S'], parms['b_V'], parms['decay_r'], parms['growth_r'], parms['a']
    dr = (growth_r * (S + V) - decay_r * r) * 0.001
    dS = r * S * (1 - (S + V) / k) - (b_S * S**2) / (a**2 + S**2)
    dV = r * V * (1 - (V + S) / k) - (b_V * V**2) / (a**2 + V**2)
    return [dS, dV, dr]

def coke_alcohol(t, y, parms):
    """
    ODE model for alcohol and cocaine use dynamics.

    Parameters:
        t (float): Time
        y (list): List of state variables [N, C, r_N, r_C]
        parms (dict): Dictionary of model parameters:
            - k, b_C, b_N, decay_r, growth_r, a

    Returns:
        list: Derivatives [dN, dC, dr_N, dr_C]
    """
    N, C, r_N, r_C = y
    k, b_C, b_N, decay_r, growth_r, a = parms['k'], parms['b_C'], parms['b_N'], parms['decay_r'], parms['growth_r'], parms['a']
    dN = r_N * N * (1 - N / k) - (b_N * N**2) / (a**2 + N**2)
    dC = r_C * C * (1 - C / k) - ((b_C + np.sin(N / 2) * 1.5) * C**2) / (a**2 + C**2)
    dr_N = (growth_r * N - decay_r * r_N) * 0.001
    dr_C = (growth_r * C - decay_r * r_C) * 0.001
    return [dN, dC, dr_N, dr_C]

# --- General simulation runner ---

def run(model, state, parms, tmax=50, after=None, table=True):
    """
    Integrates the ODE system using scipy's solve_ivp.

    Parameters:
        model (callable): Function that computes the derivatives
        state (dict): Initial values of state variables
        parms (dict): Parameter dictionary
        tmax (float): Simulation time horizon
        after (str or None): Optional intervention code, executed during simulation
        table (bool): Whether to return a pandas DataFrame

    Returns:
        pd.DataFrame or OdeSolution: Simulation results
    """
    t_eval = np.linspace(0, tmax, 500)

    def wrapped(t, y):
        if after:
            # Allows user-defined interventions during simulation
            exec(after, {}, {'state': y, 't': t})
        return model(t, y, parms)

    sol = solve_ivp(wrapped, [0, tmax], list(state.values()), t_eval=t_eval)

    if table:
        df = pd.DataFrame(sol.y.T, columns=state.keys())
        df['time'] = sol.t
        return df
    else:
        return sol

# --- Example simulation runs ---

# Scenario 1: smoking only, stable regulation
state = {'S': 0.1, 'r': 1.2}
parms = {'k': 10, 'b_S': 2.1, 'decay_r': 2, 'growth_r': 1, 'a': 1.5}
smoking_data = run(smoking, state, parms, tmax=50)

# Scenario 2: smoking and vaping with moderate inhibition
state = {'S': 0.1, 'V': 0.1, 'r': 1.2}
parms = {'k': 10, 'b_S': 2.1, 'b_V': 2, 'decay_r': 2, 'growth_r': 1, 'a': 1.5}
vape_save = run(vaping_smoking, state, parms, tmax=50)

# Scenario 3: smoking only, stronger inhibition
state = {'S': 0.1, 'r': 1}
parms = {'k': 10, 'b_S': 2.4, 'decay_r': 2, 'growth_r': 1, 'a': 0.9}
no_smoke = run(smoking, state, parms, tmax=100)

# Scenario 4: vaping worsens smoking during a time window
state = {'S': 0.1, 'V': 0.1, 'r': 1.2}
parms = {'k': 10, 'b_S': 2, 'b_V': 1.5, 'decay_r': 2, 'growth_r': 1.4, 'a': 0.9}
vape_bad = run(
    vaping_smoking, state, parms, tmax=100,
    after='if 50 < t < 60: state[1] = 0.01'  # Reduce V (index 1) temporarily
)

# Scenario 5: co-use of alcohol and cocaine, both regulated
state = {'N': 0.1, 'C': 0.1, 'r_N': 1.2, 'r_C': 1.2}
parms = {'k': 10, 'b_C': 4, 'b_N': 2, 'decay_r': 2, 'growth_r': 1, 'a': 1.5}
coke_escalation = run(coke_alcohol, state, parms)

# Scenario 6: force no alcohol — expect suppression of cocaine
no_coke = run(coke_alcohol, state, parms, after='state[0] = 0')  # Set N to 0

# Scenario 7: force no cocaine — check isolated alcohol use
only_coke = run(coke_alcohol, state, parms, after='state[1] = 0')  # Set C to 0
