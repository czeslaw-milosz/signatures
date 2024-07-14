import numpy as np
import pandas as pd
import scipy


# def get_nearest_signatures(P: np.ndarray, cosmic: np.ndarray, distance_fn: callable = scipy.spatial.distance.cosine) -> np.ndarray:
#     """Get the nearest signatures in the cosmic signatures to the given matrix P.

#     Args:
#         P (np.ndarray): The matrix of signatures to compare.
#         cosmic (np.ndarray): The cosmic signatures.
#         distance_fn (callable): The distance function to use.

#     Returns:
#         np.ndarray: The nearest signatures.
#     """
#     return np.array([
#         cosmic.iloc[np.argmin([distance_fn(discovered_signature, cosmic_signature) for cosmic_signature in cosmic.values], axis=0)].index
#         for discovered_signature in P
#     ])


def get_nearest_signatures(
        discovered_signatures: pd.DataFrame | np.ndarray,
        reference_signatures: pd.DataFrame,
        distance_function: callable = scipy.spatial.distance.cosine
    ) -> dict[int, tuple[str, float]]:
    nearest_signatures = {}
    for i in range(discovered_signatures.shape[0]):
        argmin_idx = np.argmin(
            [distance_function(
                discovered_signatures[i],
                reference_signatures.iloc[:,j]
                ) for j in range(reference_signatures.shape[1])]
            )
        nearest_signatures[i] = (reference_signatures.iloc[:, argmin_idx].name, 
                                 1 - distance_function(
                                     discovered_signatures[i], reference_signatures.iloc[:, argmin_idx]
                                     )
                                )
    return nearest_signatures