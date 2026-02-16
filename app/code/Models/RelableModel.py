from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RelabelResult:
    t: float                          # adaptive threshold
    labels: np.ndarray                # shape (N,), in ORIGINAL order: {"HC","SZ","Boundary"}
    idx_neg_ref: Optional[int]  # 1-based index in the sorted valid scores (negative side), or None
    idx_pos_ref: Optional[int]  # 1-based index in the sorted valid scores (positive side), or None
    scores_sorted: np.ndarray         # sorted valid scores (ascending), for audit
    original_index: np.ndarray        # mapping: scores_sorted[i] came from original_index[i]
