# init_rcParams.py
import matplotlib as mpl

def set_mpl_settings():
    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2,
        "figure.dpi": 150
    })
