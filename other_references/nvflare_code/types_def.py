from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, Optional
import numpy as np
from typing_extensions import NotRequired

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

class SourceDataKeys(Enum):
    """
    Enum to represent different keys in the original mat file.
    """
    FILE_ID = 'FILE_ID'
    ANALYSIS_ID = 'analysis_ID'
    ANALYSIS_SCORE = 'analysis_SCORE'
    SFNC = 'sFNC'

@dataclass(frozen=True)
class RelabelResult:
    t: float                          # adaptive threshold
    labels: np.ndarray                # shape (N,), in ORIGINAL order: {"HC","SZ","Boundary"}
    idx_neg_ref: Optional[int]  # 1-based index in the sorted valid scores (negative side), or None
    idx_pos_ref: Optional[int]  # 1-based index in the sorted valid scores (positive side), or None
    scores_sorted: np.ndarray         # sorted valid scores (ascending), for audit
    original_index: np.ndarray        # mapping: scores_sorted[i] came from original_index[i]
