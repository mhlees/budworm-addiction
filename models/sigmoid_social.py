import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from init_rcParams import set_mpl_settings

# Set matplotlib settings (custom from init_rcParams, if available)
try:
    set_mpl_settings()
except ValueError:
    pass

# Set figure resolution for high-quality plots
mpl.rc("figure", dpi=330)

# --- Sigmoid Function Plot ---

def B_i(N_j, T_B, h_B):
    """
    Sigmoid function defining how influence B_i changes based on N_j.
    This models threshold effects with a smooth transition.

    Parameters:
        N_j : float or array-like
            Input value(s) (e.g., consumption or exposure level).
        T_B : float
            Steepness parameter controlling transition sharpness.
        h_B : float
            Midpoint or threshold value (e.g., halfway point).

    Returns:
        float or np.ndarray
            Output of the sigmoid function.
    """
    return 1 / (1 + np.exp((N_j - h_B) / T_B))

def plot_sigmoid_responses(h_B=2.5, T_B_values=[0.1, 0.3, 0.5], N_j_range=(0, 5)):
    """
    Plot the sigmoid function B_i(N_j) for a range of steepness values (T_B).

    Parameters:
        h_B : float
            Midpoint of the sigmoid, same for all T_B values.
        T_B_values : list of float
            Different steepness values for comparison.
        N_j_range : tuple
            The (min, max) range of N_j values to evaluate.
    """
    # Generate N_j values over specified range
    N_j_values = np.linspace(N_j_range[0], N_j_range[1], 500)

    # Different line styles for plotting each T_B curve
    linestyles = ['--', '-', '-.']

    # Create the plot
    plt.figure(figsize=(8, 5))
    for T_B, ls in zip(T_B_values, linestyles):
        B_i_values = B_i(N_j_values, T_B, h_B)
        plt.plot(N_j_values, B_i_values, label=rf'$T_{{B}}={T_B}$', linestyle=ls)

    # Highlight and annotate the h_B point for the first curve
    y_hb = B_i(h_B, T_B_values[0], h_B)
    plt.scatter([h_B], [y_hb], color='#A663CC', zorder=5, s=100)
    plt.text(h_B + 0.4, y_hb + 0.03, '$h_B$', color='#A663CC',
             verticalalignment='bottom', horizontalalignment='right')

    # Axis labels and title
    plt.xlabel(r'$N_j$')
    plt.ylabel(r'$N_j \rightarrow B_i$')
    plt.legend()
    plt.title("Sigmoid Mapping from N_j to B_i")
    plt.tight_layout()
    plt.show()

# Call the plotting function to display curves for different T_B values
plot_sigmoid_responses()
