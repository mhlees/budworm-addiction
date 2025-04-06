
# models/social_phenomena_sim.py

# Python translation of the R Budworm model with dynamic networks and addiction dynamics
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

np.random.seed(42)

# --- Model parameters ---
n = 100
h_b = 1
t_b = 0.1
k_physical = 10
r_growth = 0.0005
r_decay = 0.003
r_base = np.random.uniform(0.5, 1.5, n)
a = np.random.uniform(0.8, 2.3, n)
k_base = np.random.uniform(1e-6, 1.1e-6, n)
b_base = np.random.uniform(0.3, 2.3, n)
beta = 0.1
kappa = 0.3

# --- Initial state setup ---
network = np.random.choice([0, 1], size=(n, n), p=[0.95, 0.05])
network = np.triu(network, 1)
network += network.T
d = np.diag(np.zeros(n))
network = network + d

state = np.concatenate([
    np.full(n, 0.0001),  # N
    r_base,              # r
    network.flatten()    # network
])

# --- Derivative function ---
def budworm_ode(state, t, parms):
    N = state[:n]
    r = state[n:2*n]
    m = state[2*n:].reshape((n, n))

    k_base = parms['k_base']
    b_base = parms['b_base']
    a = parms['a']

    logistic = 1 / (1 + np.exp((N - h_b) / t_b))
    b = b_base + beta * logistic @ m
    k = np.minimum(k_base + (N @ m) * kappa, 10)

    dN = r * N * (1 - N / k) - (b * N**2) / (a**2 + N**2)
    dr = r_growth * N - r_decay * (r - r_base)
    dnetwork = np.zeros(n * n)

    return np.concatenate([dN, dr, dnetwork])

# --- Dynamic update of the network based on utility ---
def utility_network(net, distance_matrix, h_l):
    threshold = np.exp(-distance_matrix / h_l)
    return (threshold > np.random.rand(n, n)).astype(int)

# --- Simulation runner ---
def run_simulation(state, parms, tmax=300, intervention_time=None, intervention=None):
    dt = 1
    times = np.arange(0, tmax, dt)
    results = []

    for t in times:
        if intervention_time and t == intervention_time and intervention is not None:
            intervention(state, parms)

        state = odeint(budworm_ode, state, [t, t+dt], args=(parms,))[-1]

        # Update network after each step
        N_vals = state[:n]
        net = state[2*n:].reshape((n, n))
        distance = np.abs(N_vals[:, None] - N_vals)
        updated_net = utility_network(net, distance, h_l=n * 0.1)
        np.fill_diagonal(updated_net, 0)
        state[2*n:] = updated_net.flatten()

        results.append(state.copy())

    return np.array(results)

# --- Visualization helpers ---
def extract_networks(sim_data):
    return [sim_data[i, 2*n:].reshape(n, n) for i in range(sim_data.shape[0])]

def extract_N_values(sim_data):
    return sim_data[:, :n]

# --- Example run ---
parms = {
    'k_base': k_base,
    'b_base': b_base,
    'a': a
}

sim_data = run_simulation(state, parms)

N_trajectories = extract_N_values(sim_data)
networks = extract_networks(sim_data)

# Plot N values over time
plt.figure(figsize=(10, 5))
plt.plot(N_trajectories[:, :10])  # Plotting first 10 individuals
plt.xlabel('Time')
plt.ylabel('N (Population)')
plt.title('Addiction Spread Over Time')
plt.grid(True)
plt.show()

# Plot final network state
G = nx.from_numpy_array(networks[-1])
pos = nx.spring_layout(G, seed=42)
colors = N_trajectories[-1, :]
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=80, cmap=plt.cm.viridis) # type: ignore
plt.title("Final Network State (Color = N)")
plt.show()
