import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from distutils.util import strtobool

from utils.logger import NvFlareLogger
from . import client_constants


def validate_and_get_inputs(data_path: str, computation_parameters: Dict[str, Any], logger: NvFlareLogger) -> bool:
    """
       Performs validation on the FNC data and label files against provided computation parameters
    """
    try:
        ignore_subjects_with_missing_entries = computation_parameters.get("IgnoreSubjectsWithMissingData",
                                                                          client_constants.DEFAULT_IgnoreSubjectsWithMissingData)
        ignore_subjects_with_missing_entries = bool(strtobool(str(ignore_subjects_with_missing_entries)))
        logger.info(f'ignore_subjects_with_missing_entries = {ignore_subjects_with_missing_entries}')

        # Load the data and label files
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), header=None)
        labels = pd.read_csv(os.path.join(data_path, 'labels.csv'), header=None, names=['label'])

        # Ensure number of rows are same in both
        if len(data) != len(labels):
            error_message = (f"\n Number of rows in 'data.csv' do not match with number of rows in 'labels.csv'. Please "
                             f"correct and reupload.")
            logger.info(error_message)
            return False, None, None

        label_definition = computation_parameters.get("LabelDefinition", {})
        if label_definition is None or len(label_definition.keys()) == 0:
            error_message = (f"Please provide label descriptions for all the labels in the data in the dictionary format.")
            logger.info(error_message)
            return False, None, None

        #Make sure there are two labels defined in the user parameters
        if len(label_definition.keys()) > 2 :
            error_message = (
                f"The code needs '2' label types, one for each healthy and non-healthy groups. You have provided only one label. Please check your data. ")
            logger.info(error_message)
            return False, None, None

        elif len(label_definition.keys()) < 2:
            error_message = (
                f"The code currently supports only 2 label types, healthy and non-healthy groups. You have provided more than two labels. ")
            logger.info(error_message)
            return False, None, None

        # Ensure that values in label file are the ones listed in "LabelDefinition" parameters
        unique_label_list_in_data = set(labels.iloc[:, 0].unique().tolist())
        unique_label_list_in_desc = set()
        for label_dict in label_definition.values():
                 unique_label_list_in_desc.add(label_dict.get("label"))

        #Make sure isControlLabel is in LabelDefinition
        is_control_label = computation_parameters.get("isControlLabel", None)
        if is_control_label is None or len(is_control_label) == 0:
            if label_definition is None or len(label_definition.keys()) == 0:
                error_message = (
                    f"Please provide which of the labels in the label descriptions '{label_definition.keys()}' belongs to healthy controls.")
                logger.info(error_message)
                return False, None, None

        if not unique_label_list_in_data.issubset(unique_label_list_in_desc):
            error_message = (f"Label descriptions do not contain all expected labels. Provided labels:  "
                             f"{unique_label_list_in_desc}, but label file has: {unique_label_list_in_data}.\n")
            logger.info(error_message)
            return False, None, None

        data_label_map = get_label_map(label_definition, is_control_label)

        combined_df = pd.concat([data, labels ], axis=1)

        # Strip whitespace from all string/object columns
        combined_df = _trim_all_object_columns(combined_df)

        # Convert the entire DataFrame to numeric (float), coercing errors to NaN
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

        # Rows in nan
        all_rows_to_ignore = np.where(combined_df.isna().any(axis=1))[0].tolist()

        # Check for missing values in both data and labels
        if len(all_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_entries:
                logger.info(
                    f'-- Ignored following rows with incorrect column values: {str(_get_user_row_numbers(all_rows_to_ignore))}')
                combined_df.drop(all_rows_to_ignore, inplace=True)
            else:
                err_msg = (
                    f'Following rows have empty or invalid entries for columns. Either choose to ignore these rows '
                    f'or correct the data and try again. See log file for details: {str(_get_user_row_numbers(all_rows_to_ignore))}')
                logger.error(err_msg)
            return False, None, None

        combined_data_and_labels = combined_df.to_numpy()
        label_map = get_label_map(label_definition, is_control_label)
        # y = combined_df.pop(combined_df.columns[0]).values
        # X = combined_df.to_numpy()
        logger.info("Data validation passed for the data. Running next steps.")
        return True, combined_data_and_labels,label_map

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        return False, None, None


def get_complete_FNC_matrix_data(upper_traingle_data):
    # Assume U is your upper triangular matrix
    n = int((1 + np.sqrt(1 + 8*upper_traingle_data.shape[1])) / 2)

    complete_fnc_matrix=[]
    for subj_id in range(upper_traingle_data.shape[0]):
        subj_fnc_matrix = np.zeros((n,n))
        triu_indices = np.triu_indices(n=n, k=1)

        subj_fnc_matrix[triu_indices] = upper_traingle_data[subj_id, :]
        subj_fnc_matrix += subj_fnc_matrix.T - np.diag(np.diag(subj_fnc_matrix))

        np.fill_diagonal( subj_fnc_matrix, 1)

        complete_fnc_matrix.append(subj_fnc_matrix)

    # Create symmetric matrix
    complete_fnc_matrix = np.asarray(complete_fnc_matrix)

    return complete_fnc_matrix


def _get_user_row_numbers(df_index_list):
    return [ri + 1 for ri in df_index_list]


def get_label_map(label_definition, is_control_label):
    label_map = {}

    labels = label_definition.keys()
    for label_numbers, label_dict in label_definition.items():
        if label_dict.get("name") == is_control_label:
            HC_label_id = label_dict.get("label")
            label_map[client_constants.CODE_HC_LABEL] = HC_label_id
        else:
            SZ_label_id = label_dict.get("label")
            label_map[client_constants.CODE_SZ_label] = SZ_label_id

    return label_map


def _trim_all_object_columns(df):
    """
    Trim whitespace from ends of each string value across all object type series in the dataframe.
    """
    # Select columns with 'object' dtype (which usually means strings)
    obj_cols = df.select_dtypes(include=['object']).columns

    # Apply the vectorized str.strip() to these columns
    df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())

    return df