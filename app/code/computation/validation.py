"""Validate one site's FNC data/labels CSVs against the computation parameters."""

import os
from distutils.util import strtobool

import numpy as np
import pandas as pd

from . import constants


def validate_and_get_inputs(data_dir, parameters, logger):
    """Validate a site's ``data.csv``/``labels.csv`` and return the combined array.

    Returns ``(True, combined_data_and_labels)`` on success, where the
    returned array's last column is the label; on failure returns
    ``(False, None)`` after logging the reason.
    """
    try:
        ignore_subjects_with_missing_entries = parameters.get(
            "IgnoreSubjectsWithMissingData",
            constants.DEFAULT_IgnoreSubjectsWithMissingData,
        )
        ignore_subjects_with_missing_entries = bool(
            strtobool(str(ignore_subjects_with_missing_entries))
        )
        logger.info(
            f"ignore_subjects_with_missing_entries = {ignore_subjects_with_missing_entries}"
        )

        data = pd.read_csv(os.path.join(data_dir, "data.csv"), header=None)
        labels = pd.read_csv(
            os.path.join(data_dir, "labels.csv"), header=None, names=["label"]
        )

        if len(data) != len(labels):
            error_message = (
                "\n Number of rows in 'data.csv' do not match with number of rows "
                "in 'labels.csv'. Please correct and reupload."
            )
            logger.info(error_message)
            return False, None

        label_definition = parameters.get("LabelDefinition", {})
        if label_definition is None or len(label_definition.keys()) == 0:
            error_message = (
                "Please provide label descriptions for all the labels in the data "
                "in the dictionary format."
            )
            logger.info(error_message)
            return False, None

        # Make sure there are two labels defined in the user parameters
        if len(label_definition.keys()) > 2:
            error_message = (
                "The code needs '2' label types, one for each healthy and "
                "non-healthy groups. You have provided only one label. Please "
                "check your data. "
            )
            logger.info(error_message)
            return False, None
        elif len(label_definition.keys()) < 2:
            error_message = (
                "The code currently supports only 2 label types, healthy and "
                "non-healthy groups. You have provided more than two labels. "
            )
            logger.info(error_message)
            return False, None

        # Ensure that values in label file are the ones listed in "LabelDefinition" parameters
        unique_label_list_in_data = set(labels.iloc[:, 0].unique().tolist())
        unique_label_list_in_desc = set()
        for label_dict in label_definition.values():
            unique_label_list_in_desc.add(label_dict.get("label"))

        # Make sure isControlLabel is in LabelDefinition
        is_control_label = parameters.get("isControlLabel", None)
        if is_control_label is None or len(is_control_label) == 0:
            if label_definition is None or len(label_definition.keys()) == 0:
                error_message = (
                    "Please provide which of the labels in the label descriptions "
                    f"'{label_definition.keys()}' belongs to healthy controls."
                )
                logger.info(error_message)
                return False, None

        if not unique_label_list_in_data.issubset(unique_label_list_in_desc):
            error_message = (
                "Label descriptions do not contain all expected labels. Provided "
                f"labels:  {unique_label_list_in_desc}, but label file has: "
                f"{unique_label_list_in_data}.\n"
            )
            logger.info(error_message)
            return False, None

        combined_df = pd.concat([data, labels], axis=1)

        # Strip whitespace from all string/object columns
        combined_df = _trim_all_object_columns(combined_df)

        # Convert the entire DataFrame to numeric (float), coercing errors to NaN
        combined_df = combined_df.apply(pd.to_numeric, errors="coerce")

        # Rows in nan
        all_rows_to_ignore = np.where(combined_df.isna().any(axis=1))[0].tolist()

        # Check for missing values in both data and labels
        if len(all_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_entries:
                logger.info(
                    "-- Ignored following rows with incorrect column values: "
                    f"{str(_get_user_row_numbers(all_rows_to_ignore))}"
                )
                combined_df.drop(all_rows_to_ignore, inplace=True)
            else:
                err_msg = (
                    "Following rows have empty or invalid entries for columns. "
                    "Either choose to ignore these rows or correct the data and "
                    "try again. See log file for details: "
                    f"{str(_get_user_row_numbers(all_rows_to_ignore))}"
                )
                logger.error(err_msg)
            return False, None

        combined_data_and_labels = combined_df.to_numpy()
        logger.info("Data validation passed for the data. Running next steps.")
        return True, combined_data_and_labels

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        return False, None


def get_complete_FNC_matrix_data(upper_triangle_data):
    """Reconstruct symmetric FNC matrices from their upper-triangle values."""
    n = int((1 + np.sqrt(1 + 8 * upper_triangle_data.shape[1])) / 2)

    complete_fnc_matrix = []
    for subj_id in range(upper_triangle_data.shape[0]):
        subj_fnc_matrix = np.zeros((n, n))
        triu_indices = np.triu_indices(n=n, k=1)

        subj_fnc_matrix[triu_indices] = upper_triangle_data[subj_id, :]
        subj_fnc_matrix += subj_fnc_matrix.T - np.diag(np.diag(subj_fnc_matrix))

        np.fill_diagonal(subj_fnc_matrix, 1)

        complete_fnc_matrix.append(subj_fnc_matrix)

    return np.asarray(complete_fnc_matrix)


def _get_user_row_numbers(df_index_list):
    return [ri + 1 for ri in df_index_list]


def _trim_all_object_columns(df):
    """Trim whitespace from ends of each string value across all object columns."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())
    return df
