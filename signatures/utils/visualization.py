import matplotlib
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from signatures import config


def plot_signature(signature_array: np.ndarray, axs: matplotlib.axes.Axes | None = None, text: str = "") -> None:
    width = max(signature_array.shape)
    ylim = signature_array.max() + config.Y_MARGIN_SINGLE_SIGNATURE
    
    x = np.arange(width)
    if axs is None:
        _, axs = plt.subplots(1, figsize=(20, 10))
    axs.bar(x, signature_array, edgecolor="black", color=config.SIGNATURES_COLOR_ARRAY)
    patches = [
        mpatches.Patch(color=color, label=mutation_type)
        for color, mutation_type in zip(
            config.SIGNATURES_COLORS, 
            config.COSMIC_MUTATION_TYPES_SBS6
        )
    ]
    axs.legend(handles=patches, loc="upper center", ncol=6, fontsize=18, mode="expand")
    plt.ylim(0, ylim)
    plt.yticks(fontsize=10)
    axs.set_xlim(-0.5, width) 
    axs.set_ylabel("Probability of mutation \n", fontsize=12)
    axs.set_xticks([])
    axs.set_xticks(x)  
    axs.set_xticklabels(config.COSMIC_MUTATION_TYPES, rotation=90, fontsize=7) 
    if text:
        plt.text(2, ylim-0.005, text, fontsize=15)
