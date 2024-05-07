import itertools

DATA_DIR = "data"
MUTATIONS_FILE_NAME = "simple_somatic_mutation.open.BRCA-EU.tsv"
COSMIC_FILE_PATH = "resources/data/cosmic/COSMIC_v3.4_SBS_GRCh37.txt"
BRCA_COUNTS_FILE_PATH = "resources/data/brca/matrices/sigmatrix_96.csv"

COSMIC_MUTATION_TYPES = [
    "A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T", "A[C>G]A", "A[C>G]C",
    "A[C>G]G", "A[C>G]T", "A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T",
    "A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T", "A[T>C]A", "A[T>C]C",
    "A[T>C]G", "A[T>C]T", "A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T",
    "C[C>A]A", "C[C>A]C", "C[C>A]G", "C[C>A]T", "C[C>G]A", "C[C>G]C",
    "C[C>G]G", "C[C>G]T", "C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T",
    "C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T", "C[T>C]A", "C[T>C]C",
    "C[T>C]G", "C[T>C]T", "C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T",
    "G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T", "G[C>G]A", "G[C>G]C",
    "G[C>G]G", "G[C>G]T", "G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T",
    "G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T", "G[T>C]A", "G[T>C]C",
    "G[T>C]G", "G[T>C]T", "G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T",
    "T[C>A]A", "T[C>A]C", "T[C>A]G", "T[C>A]T", "T[C>G]A", "T[C>G]C",
    "T[C>G]G", "T[C>G]T", "T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T",
    "T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T", "T[T>C]A", "T[T>C]C",
    "T[T>C]G", "T[T>C]T", "T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"
]
COSMIC_MUTATION_TYPES_SBS6 = [
        "C>A", "C>G", "C>T", "T>A", "T>C", "T>G"
]
COUNTS_DF_PATH = "data/brca/matrices/sigmatrix_96.csv"

SIGNATURES_COLORS = [
    (0.196,0.714,0.863),  # COSMIC blue
    (0.102,0.098,0.098),  # black
    (0.816,0.180,0.192),  # COSMIC red
    (0.777,0.773,0.757),  # COSMIC grey
    (0.604,0.777,0.408),  # COSMIC green
    (0.902,0.765,0.737),  # COSMIC pink-ish
]
SIGNATURES_COLOR_ARRAY = list(
    itertools.chain.from_iterable(
        (color_tuple,)*16 for color_tuple in SIGNATURES_COLORS
        )
    )
FIGSIZE_SINGLE_SIGNATURE = (20, 10)
Y_MARGIN_SINGLE_SIGNATURE = 0.01

RANDOM_SEED = 2137213721

APPLY_AUGMENTATION = False
BATCH_SIZE = 32
# LEARNING_RATE = 1e-3 * 2
LEARNING_RATE = 1e-3
BETAS = (0.95, 0.999)
DROPOUT = 0.2
HIDDEN_SIZE = 100
NUM_EPOCHS = 50
N_SIGNATURES_TARGET = 14

LOSS_REGULARIZER = None
REGULARIZER_LAMBDA = 1e+04

EXPERIMENT_NAME = "ProdLDA"
PRODLDA_MODEL_PATH = f"resources/saved_models/{EXPERIMENT_NAME}_model.pt"
LOSS_PLOT_PATH = f"resources/plots/{EXPERIMENT_NAME}_loss.png"
SIGNATURES_PLOT_PATH = f"resources/plots/{EXPERIMENT_NAME}_signatures.png"
SAVE_MODEL = False
