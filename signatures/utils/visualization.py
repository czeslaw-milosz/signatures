import matplotlib
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from signatures import config


# def plot_signatures(discovered_signatures: pd.DataFrame, reference_signatures: pd.DataFrame) -> None:
#     fig, axs = plt.subplots(len(discovered_signatures.columns), 2)
#     fig.suptitle("ProdLDA-discovered signatures vs. cosine-nearest COSMIC signatures")
#     plt.rcParams["figure.dpi"] = 240
#     plt.rcParams['figure.figsize'] = [6.4, 8]
#     for i in range()


#     plt.rcParams["figure.dpi"] = 240
#     plt.rcParams['figure.figsize'] = [6.4, 8]
#     fig, axs = plt.subplots(len(discovered_plots), 2)
#     fig.suptitle("ProbLDA-discovered signatures vs. cosine-nearest COSMIC signatures")
#     for i in range(len(discovered_plots)):
#         axs[i,0].imshow(discovered_plots[i])
#         axs[i,0].axis("off")
#         axs[i,0].text(0.7, 0.7, f"cos similarity: {str(round(nearest_signatures[i][1], 2))}", size=8)
#         axs[i,1].imshow(mpimg.imread(f"./data/cosmic/plots/{nearest_signatures[i][0]}.png"))
#         axs[i,1].axis("off")
#     plt.savefig(config.SIGNATURES_PLOT_PATH)


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
