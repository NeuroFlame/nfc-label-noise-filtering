from typing import TypedDict
import numpy as np


class Centroids(TypedDict):
    center_sz: np.ndarray
    center_hc: np.ndarray
    selected_features: np.ndarray
    typ_sz: np.ndarray
    typ_hc: np.ndarray
