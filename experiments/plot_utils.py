"""
Plotting and Visualization Utilities.
Configures publication-quality figure formatting matching the paper's style.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def setup_matplotlib_style():
    """Apply clean academic styling for publication figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.5,
    })


# Paper color mappings
PALETTE = {
    "AHD-SHGA": "#C83E74",        # Vivid Magenta/Crimson for proposed method
    "Proposed AHD-SHGA": "#C83E74",
    "NGA": "#4A90E2",            # Blue
    "NGA [21]": "#4A90E2",
    "MGEA": "#F5A623",           # Amber/Orange
    "MGEA [22]": "#F5A623",
    "GA-SMLD": "#E74C3C",        # Coral Red
    "GA-SMLD [23]": "#E74C3C",
    "LLM-GWO": "#7ED321",        # Lime Green
    "LLM-GWO [24]": "#7ED321",
    "HA-GA-NNQ": "#9013FE",      # Purple
    "HA-GA-NNQ [25]": "#9013FE",
    "LLM-ECG": "#50E3C2",        # Teal/Cyan
    "LLM-ECG [26]": "#50E3C2",
    "JADE": "#BD10E0",
    "JADE [32]": "#BD10E0",
}
