import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.io import savemat
from scipy.stats import ttest_ind

from executor.types_def import HeatMapOptions
import h5py
import time

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
    # plt.show()


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

def compute_two_sample_ttest(dataset: np.ndarray, labels12=None):
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


def find_sum_count_fnc(X: np.ndarray, clip: int = 1-1e-7):
    """
    X: shape (n_g, 53, 53) for one site & one group (HC or SZ), correlations in [-1,1].
    Returns edge-wise Fisher-z SUM and COUNT (and optional SUMSQ for variance if needed).
    """
    n, p, _ = X.shape

    clipped_data = np.clip(X, -clip, clip) # ranging values from -x to x
    z_transformed = np.arctanh(clipped_data) # Fisher Z transformed

    diag = np.arange(p)
    z_transformed[:, diag, diag] = 0.0

    """ 
        S = X[0] + X[1] + X[2] + .... + X[n] 
        sum of each matrix (53x53)
        
        C = for each S[i][j] no.of X[i] matrix X[i][j] not NaN
    """
    aggregated_sum_matrix = np.nansum(z_transformed, axis=0)          # (p, p)
    total_count = np.sum(~np.isnan(z_transformed), axis=0)  # (p, p)

    ## Local Mean:
    z_transformed_mean = np.nanmean(z_transformed, axis=0)

    # Back-transform to r-space
    inv_z_transformed = np.tanh(z_transformed_mean)

    # Housekeeping
    inv_z_transformed = (inv_z_transformed + inv_z_transformed.T) / 2.0
    np.fill_diagonal(inv_z_transformed, 1.0)
    return {"sum": aggregated_sum_matrix, "count": total_count, "avg_fnc": inv_z_transformed}


def average_fnc_by_label(
        X: np.ndarray,  # shape: (N, 53, 53), correlations in [-1,1]
        y: np.ndarray,  # shape: (N,), labels in {1,2}
        labels: np.ndarray,  # which labels to compute
) -> tuple[dict[int, ndarray], dict[int, dict[str, any]]]:
    """
    Returns: dict {label: avg_matrix} with each avg_matrix shape (53, 53).
    Steps: Fisher z per subject -> element-wise mean in z -> tanh back to r.
    Uses np.nanmean so any NaNs in X are ignored.
    """
    assert X.ndim == 3 and X.shape[1] == X.shape[2] == 53, "X must be (N,53,53)"
    assert y.shape[0] == X.shape[0], "y must align with X"

    avg_fnc = {}
    local_sum_count = {}

    for lbl in labels:
        idx = (y == lbl)

        if not np.any(idx):
            raise ValueError(f"No subjects with label {lbl}")

        local_mean_data = find_sum_count_fnc(X[idx])

        avg_fnc[lbl] = local_mean_data['avg_fnc']
        local_sum_count[lbl] = {'sum': local_mean_data['sum'], 'count': local_mean_data['count']}

    return avg_fnc, local_sum_count


def global_mean_from_sites(site_packets: dict):
    """
    site_packets: list of dicts returned by site_stats_z for the SAME group.
    Supports either upper-triangle or full-matrix mode (must match across sites).
    Returns global average correlation matrix (p,p) for that group.
    """
    print('Global: ')

    result = {}
    for label in (1, 2):
        aggregated_sum = np.zeros((53, 53), dtype=float)
        aggregated_total = np.zeros((53, 53), dtype=float)

        print(" label: ", label, "  count: ", len(site_packets[label]))

        for pkt in site_packets[label]:
            aggregated_sum += pkt["sum"]
            aggregated_total += pkt["count"]

        avg_fnc = np.divide(aggregated_sum, aggregated_total, out=np.zeros_like(aggregated_sum), where=aggregated_total > 0)

        inverse_avg_fnc = np.tanh(avg_fnc)
        inverse_avg_fnc = (inverse_avg_fnc + inverse_avg_fnc.T) / 2.0
        np.fill_diagonal(inverse_avg_fnc, 1.0)

        result[label] = inverse_avg_fnc

    return result

def save_fnc_h5(filepath: str, fnc: np.ndarray, space: str = "z"):
    """
    Save a plain FNC matrix (N x N) or a stack (S x N x N) to an HDF5 file.

    Args:
        filepath: Output path ending with .h5 or .hdf5
        fnc: np.ndarray of shape (N, N) or (S, N, N); dtype will be float32 on disk
        space: "z" or "r" (optional metadata to indicate Fisher-z or correlation r)
        labels: optional list/array of length N with region/component names
    """
    fnc = np.asarray(fnc)
    if fnc.ndim == 2:
        N = fnc.shape[0]
        if fnc.shape[1] != N:
            raise ValueError("FNC must be square (N x N).")
    elif fnc.ndim == 3:
        S, N, N2 = fnc.shape
        if N2 != N:
            raise ValueError("FNC stack must be (S x N x N).")
    else:
        raise ValueError("FNC must be 2D (N x N) or 3D (S x N x N).")

    # Save
    with h5py.File(filepath, "w") as f:
        # Minimal metadata
        f.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.attrs["schema_version"] = "1.0"
        f.attrs["space"] = space  # "z" for Fisher-z, "r" for Pearson r

        # Main dataset: /fnc
        dset = f.create_dataset(
            "fnc",
            data=fnc.astype(np.float32, copy=False),
            compression="gzip",
            chunks=True  # let h5py pick reasonable chunk sizes
        )
        dset.attrs["shape_kind"] = "NxN" if fnc.ndim == 2 else "SxNxN"
        dset.attrs["dtype"] = "float32"

def load_fnc_h5(filepath: str):
    """
    Load FNC and (optional) labels back from the HDF5 file.
    Returns:
        fnc (np.ndarray), meta (dict)
    """
    with h5py.File(filepath, "r") as f:
        fnc = f["fnc"][...]  # np.ndarray
        meta = {
            "created_utc": f.attrs.get("created_utc", None),
            "schema_version": f.attrs.get("schema_version", None),
            "space": f.attrs.get("space", None),
            "shape_kind": f["fnc"].attrs.get("shape_kind", None),
        }
    return fnc, meta