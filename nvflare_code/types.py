from typing import TypedDict, NotRequired, Unpack
import numpy as np

class Centroids(TypedDict):
    center_sz: np.ndarray
    center_hc: np.ndarray
    selected_features: np.ndarray
    typ_sz: np.ndarray
    typ_hc: np.ndarray


class HeatMapOptions(TypedDict):
    colorbar_name: str
    title: str
    name: str
    domain_names: NotRequired[list[int]]
    path: str