import os
import csv
import numpy as np
from scipy.io import savemat

from executor.local_ancillary import find_group_differences, find_avg_fnc_measures
from utils.types import SourceDataKeys, ConfigDTO


def load_fnc_dataset(input_path):
    """
    input_path: directory containing subjects.csv and the .npy FNC files

    Returns:
        fnc_array: numpy array of shape (N, P, P)
        labels_array: numpy array of shape (N,)
    """

    csv_file = os.path.join(input_path, "subjects.csv")

    subject_files = []
    labels = []

    # 1. Read subjects.csv
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_files.append(row["subject_file"])
            labels.append(int(row["label"]))

    # 2. Load FNC matrices
    fnc_list = []

    for file_name in subject_files:
        fnc_path = os.path.join(input_path, file_name)

        # Validate file existence
        if not os.path.exists(fnc_path):
            raise FileNotFoundError(f"Missing FNC file: {fnc_path}")

        # Load matrix
        fnc_matrix = np.load(fnc_path)

        # Validate square matrix (P, P)
        if fnc_matrix.ndim != 2 or fnc_matrix.shape[0] != fnc_matrix.shape[1]:
            raise ValueError(f"FNC file {fnc_path} must be (P, P). Got {fnc_matrix.shape}")

        fnc_list.append(fnc_matrix)

    # 3. Convert list → numpy array (N, P, P)
    fnc_array = np.stack(fnc_list, axis=0)
    labels_array = np.array(labels)

    return fnc_array, labels_array

def convert_fnc_to_features(config: ConfigDTO):
    # --- find diagnosis column (case-insensitive) ---

    fnc_matrices, labels = load_fnc_dataset(config.data_path)
    label_groups = config.computation_params.get('LabelGroups')

    find_group_differences(fnc_matrices, labels, config.output_path, label_groups)
    aggregated_fnc_result = find_avg_fnc_measures(fnc_matrices, labels, config.output_path, label_groups)

    N, P, _ = fnc_matrices.shape

    # --- build the lower-triangle mask excluding diagonal (k=-1) ---
    mask = np.tril(np.ones((P, P), dtype=bool), k=-1)  # P x P

    linear_idx = np.where(mask.ravel(order='F'))[0]  # size: P*(P-1)/2
    fnc_flat = fnc_matrices.reshape(N, P * P, order='F')  # N x (P*P)

    source_data = fnc_flat[:, linear_idx]  # N x (P*(P-1)/2)

    # append labels as the last column
    out = np.hstack([source_data, labels])

    # save with variable name = dataset name (like MATLAB)
    out_path = os.path.join(config.output_path, f'{config.site_name}.mat')
    savemat(out_path, {config.site_name: fnc_matrices}, do_compression=True)

    return out, aggregated_fnc_result
