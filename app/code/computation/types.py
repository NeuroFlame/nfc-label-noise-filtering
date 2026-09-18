"""Values exchanged across the local/remote boundary by the LAMP workflow."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ValidatedInputs:
    """One site's validated data, plus its self-identity token.

    ``data`` is the combined upper-triangle FNC features + label array (as
    returned by ``validation.validate_and_get_inputs``): its last column is
    the subject label.
    """

    data: np.ndarray
    self_token: str


@dataclass
class CentroidSet:
    """One site's typical-subject centroids and the features that define them."""

    center_sz: np.ndarray
    center_hc: np.ndarray
    selected_features: np.ndarray


@dataclass
class CrfLocalResult:
    """A site's round-1 payload: its centroids plus local FNC sum/count per label."""

    self_token: str
    centroids: CentroidSet
    fnc_sum: Dict[str, np.ndarray]
    fnc_count: Dict[str, np.ndarray]


@dataclass
class GlobalCentroidResult:
    """The aggregator's round-1 broadcast: every site's centroids, by display name.

    ``token_to_name`` lets a site resolve which entry in ``site_centroids`` is
    its own, using the ``self_token`` it minted in ``load_inputs`` -- the
    framework does not otherwise expose a site's own display name to author
    code (mirrors ``nfc-multi-round-regression-freesurfer``'s
    ``token_to_column``).
    """

    site_centroids: Dict[str, CentroidSet]
    token_to_name: Dict[str, str]


@dataclass
class LocalScoreResult:
    """A site's round-2 payload: its per-subject average dimensional score."""

    scores: np.ndarray


@dataclass
class LocalRelabelResult:
    """A site's round-3 payload: local FNC sum/count per label, post-relabel."""

    fnc_sum: Dict[str, np.ndarray]
    fnc_count: Dict[str, np.ndarray]


@dataclass
class GlobalReportData:
    """The aggregator's final broadcast used to build every site's report."""

    global_avg_fnc_original: Dict[str, np.ndarray]
    global_avg_fnc_relabeled: Dict[str, np.ndarray]
    adaptive_threshold: float
