import os
import numpy as np
from enum import Enum

from numpy import ndarray
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from typing import NotRequired, TypedDict, Unpack

from .data_loaders import load_data_matfile
from scipy.io import savemat

from nvflare_code.types import HeatMapOptions

class SourceDataKeys(Enum):
    """
    Enum to represent different keys in the original mat file.
    """
    FILE_ID = 'FILE_ID'
    ANALYSIS_ID = 'analysis_ID'
    ANALYSIS_SCORE = 'analysis_SCORE'
    SFNC = 'sFNC'


def convert_fnc_to_features(original_dataset, dest_path: str, name: str):

    # --- find diagnosis column (case-insensitive) ---
    file_ids = original_dataset[SourceDataKeys.FILE_ID.value]
    label_index = next(
        (i for i, col in enumerate(file_ids) if "diagnosis" in col.lower()),
        None
    )
    if label_index is None:
        raise ValueError(f'No "diagnosis" column found in FILE_ID for {name}')

    labels = original_dataset[SourceDataKeys.ANALYSIS_SCORE.value][:, label_index]
    labels = labels.reshape(-1, 1)

    fnc_matrices = original_dataset[SourceDataKeys.SFNC.value]  # shape: (N, P, P)
    N, P, _ = fnc_matrices.shape

    # --- build the lower-triangle mask excluding diagonal (k=-1) ---
    mask = np.tril(np.ones((P, P), dtype=bool), k=-1)  # P x P

    linear_idx = np.where(mask.ravel(order='F'))[0]           # size: P*(P-1)/2
    fnc_flat_F = fnc_matrices.reshape(N, P * P, order='F')    # N x (P*P)

    source_data = fnc_flat_F[:, linear_idx]                   # N x (P*(P-1)/2)

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
) -> tuple[ndarray, ndarray]:

    label_count[:, 3] = np.round(label_count[:, 3], 1)
    print(len(original_labels), len(label_count))

    typical_indexes = np.where(label_count[:, 3] >= typical_threshold)[0]
    print('len of typical subjects: ', len(typical_indexes))

    typical_labels = original_labels[typical_indexes]

    assert len(typical_labels) == len(typical_indexes)

    typical_sz = typical_indexes[typical_labels == 1]
    typical_hc = typical_indexes[typical_labels == 2]

    return np.array(typical_hc), np.array(typical_sz)


def fnc_heatmap(fnc_matrix: np.ndarray, options: HeatMapOptions):
    # Symmetric color range

    colorbar_name = options.get('colorbar_name', "")
    title = options.get('title', "Heat Map")
    name = options.get('name', "heatmap.png")
    domain_breaks: list = options.get('domain_names', [])
    output_path = options.get('path', "")

    max_value = np.nanmax(np.abs(fnc_matrix))

    max_value = max_value if np.isfinite(max_value) and max_value > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    im = ax.imshow(fnc_matrix, cmap="RdYlBu_r", vmin=-max_value, vmax=max_value, origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name, rotation=270, labelpad=10)

    # Optional domain gridlines (edit to your atlas boundaries)
    if domain_breaks:
        for g in domain_breaks:
            ax.axhline(g-0.5, color='k', lw=0.6)
            ax.axvline(g-0.5, color='k', lw=0.6)

    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    output_filename = os.path.join(output_path, name)

    plt.savefig(output_filename)
    plt.show()


def upper_triangle_bonferroni(t_values: np.ndarray, p_values: np.ndarray, alpha=0.01):
    n = 53
    upper_triangle_indexes = np.triu_indices(n, k=1)  # strict upper-triangle indices
    m_tests = len(upper_triangle_indexes[0])  # 53*52/2 = 1378
    bonferroni_correction_value = alpha / m_tests

    t_matrix = np.array(t_values, copy=True)
    np.fill_diagonal(t_matrix, 0.0)

    # Upper triangle: mask if NOT Bonferroni-significant
    sig_u = p_values[upper_triangle_indexes] < bonferroni_correction_value
    t_matrix[(upper_triangle_indexes[0][~sig_u], upper_triangle_indexes[1][~sig_u])] = np.nan

    return t_matrix


def split_dataset(dataset: np.ndarray, labels: np.ndarray) -> tuple[ndarray, ndarray]:
    assert dataset.ndim == 3 and dataset.shape[1:] == (53, 53), "A53 must be (N,53,53)"
    labels12 = np.asarray(labels).reshape(-1)

    assert dataset.shape[0] == labels12.shape[0], "N mismatch between data and labels"
    assert set(np.unique(labels12)).issubset({1, 2, -1}), "labels must be 1/2 or boundary"

    group_sz = dataset[labels12 == 1]  # SZ
    group_hc = dataset[labels12 == 2]  # HC

    return group_sz, group_hc

def computer_two_sample_ttest(dataset: np.ndarray, labels12=None):
    """
    dataset: np.ndarray of shape (N, 53, 53)  -- FNC per subject
    labels12: np.ndarray of shape (N) with values in {1,2,-1}
    """

    group_sz, group_hc = split_dataset(dataset, labels12)

    if group_sz.size == 0 or group_hc.size == 0:
        raise ValueError("Both groups need at least one subject.")

    # Welch t-test per cell (across subjects)
    t_values, p_values = ttest_ind(group_hc, group_sz, axis=0, equal_var=False, nan_policy="omit") # HC Vs SZ

    return t_values, p_values # 53x53
