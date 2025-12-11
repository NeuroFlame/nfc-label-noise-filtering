from enum import Enum
from typing import NamedTuple, Dict, Any, TypedDict, List, Optional
from typing_extensions import NotRequired
import numpy as np
from .logger import NvFlareLogger
from dataclasses import dataclass


class SourceDataKeys(Enum):
    """
    Enum to represent different keys in the original mat file.
    """
    FILE_ID = 'FILE_ID'
    ANALYSIS_ID = 'analysis_ID'
    ANALYSIS_SCORE = 'analysis_SCORE'
    SFNC = 'sFNC'

class Centroids(TypedDict):
    group1_center: np.ndarray
    group2_center: np.ndarray
    selected_features: np.ndarray
    group1_typ_subjects: np.ndarray
    group2_typ_subjects: np.ndarray

@dataclass(frozen=True)
class RelabelResult:
    t: float                          # adaptive threshold
    labels: np.ndarray                # shape (N,), in ORIGINAL order: {"HC","SZ","Boundary"}
    idx_neg_ref: Optional[int]  # 1-based index in the sorted valid scores (negative side), or None
    idx_pos_ref: Optional[int]  # 1-based index in the sorted valid scores (positive side), or None
    scores_sorted: np.ndarray         # sorted valid scores (ascending), for audit
    original_index: np.ndarray        # mapping: scores_sorted[i] came from original_index[i]


class HeatMapOptions(TypedDict):
    colorbar_name: str
    title: str
    name: str
    domain_names: NotRequired[List[int]]
    path: str

class ComputationParamDTO(TypedDict):
    SamplingThs: float
    CrfIterations: int
    CrfTrees: int
    NoiseIntensityThs: int
    TypicalThs: float
    TruncationParameter: float
    LabelGroups: Dict[str, Dict[str, Any]]
    LogLevel: str

class ConfigDTO(NamedTuple):
    data_path: str
    output_path: str
    cache_path: str
    computation_params: ComputationParamDTO
    cache_dict: Dict[str, Any]
    logger: NvFlareLogger
    site_name: str
