import os
import numpy as np
from enum import Enum
from .data_loaders import load_data_matfile
from scipy.io import savemat


class SourceDataKeys(Enum):
    """
    Enum to represent different keys in the original mat file.
    """
    FILE_ID = 'FILE_ID'
    ANALYSIS_ID = 'analysis_ID'
    ANALYSIS_SCORE = 'analysis_SCORE'
    SFNC = 'sFNC'


def convert_fnc_to_features(fnc_path: str, dest_path: str, name: str):
    file_path = os.path.join(fnc_path, 'data.mat')
    original_data = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
            SourceDataKeys.FILE_ID.value,
            SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )

    # --- find diagnosis column (case-insensitive) ---
    file_ids = original_data[SourceDataKeys.FILE_ID.value]
    label_index = next(
        (i for i, col in enumerate(file_ids) if "diagnosis" in col.lower()),
        None
    )
    if label_index is None:
        raise ValueError(f'No "diagnosis" column found in FILE_ID for {name}')

    labels = original_data[SourceDataKeys.ANALYSIS_SCORE.value][:, label_index]
    # Optional: match MATLAB’s strict check
    # uniq = np.unique(labels)
    # if not np.all(np.isin(uniq, [1, 2])):
    #     raise ValueError(f"Unexpected diagnosis codes: {uniq.tolist()}")
    labels = labels.reshape(-1, 1)

    fnc_matrices = original_data[SourceDataKeys.SFNC.value]  # shape: (N, P, P)
    N, P, _ = fnc_matrices.shape

    # --- build the lower-triangle mask excluding diagonal (k=-1) ---
    mask = np.tril(np.ones((P, P), dtype=bool), k=-1)  # P x P

    # --- replicate MATLAB's column-major flattening ---
    # 1) reshape each P×P to length P*P along Fortran (column-major) order
    # 2) take the linear indices where mask is True, also in Fortran order
    linear_idx = np.where(mask.ravel(order='F'))[0]           # size: P*(P-1)/2
    fnc_flat_F = fnc_matrices.reshape(N, P * P, order='F')    # N x (P*P)

    source_data = fnc_flat_F[:, linear_idx]                   # N x (P*(P-1)/2)
    # keep dtype default (float64) to match MATLAB; or cast if you prefer
    # source_data = source_data.astype(np.float64, copy=False)

    # append labels as the last column
    out = np.hstack([source_data, labels])

    # save with variable name = dataset name (like MATLAB)
    out_path = os.path.join(dest_path, f'{name}.mat')
    savemat(out_path, {name: out}, do_compression=True)

    return out


def find_typical_subjects(
    original_labels: np.ndarray,
    label_count: np.ndarray,
    typical_threshold: float,
):

    label_count[:, 3] = np.round(label_count[:, 3], 1)
    print(len(original_labels), len(label_count))

    typical_indexes = np.where(label_count[:, 3] >= typical_threshold)[0]
    print('len of typical subjects: ', len(typical_indexes))

    typical_labels = original_labels[typical_indexes]

    assert len(typical_labels) == len(typical_indexes)

    typical_sz = typical_indexes[typical_labels == 1]
    typical_hc = typical_indexes[typical_labels == 2]

    return typical_hc, typical_sz
