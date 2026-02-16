import h5py
from scipy.io import loadmat
import numpy as np


def decode_utf16_array(array):
    """Decode MATLAB UTF-16 encoded uint16 arrays to strings."""
    array = array.flatten() if array.ndim == 2 else array
    return ''.join(chr(c) for c in array if c != 0)


def handle_cell_string_dataset(dataset, file_handle):
    """Handle MATLAB-style cell array of strings stored as references."""
    result = []
    for i in range(dataset.shape[0]):
        ref = dataset[i, 0]
        deref = file_handle[ref]
        value = deref[()]
        if isinstance(value, bytes):
            result.append(value.decode('utf-8'))
        elif isinstance(value, np.ndarray) and value.dtype == np.uint16:
            result.append(decode_utf16_array(value))
        else:
            result.append(value)
    return result


def iterate_group(data: h5py.Group, final_data: dict):
    """
    Recursively iterate through a h5py Group and print its structure.
    """
    for key in data.keys():
        item = data[key]
        if isinstance(item, h5py.Group):
            final_data[key] = {}
            iterate_group(item, final_data[key])
        else:
            final_data[key] = item[:]
    return final_data


def load_data_matfile(path: str, name: list[str] = None):
    data = None
    with h5py.File(path, 'r') as f:
        # List all groups in the file
        # print("Keys in the file:", list(f.keys()))
        # Access the dataset

        if len(name) == 0:
            data = {}
            iterate_group(f, data)
            return data
        else:
            result = {}
            for key in name:
                if isinstance(f[key], h5py.Group):
                    # print("key is a group, iterating through it")
                    result[key] = iterate_group(f[key], {})
                elif key == 'FILE_ID':
                    result[key] = handle_cell_string_dataset(f[key], f)
                else:
                    # print("key is a dataset, returning data")
                    data = f[key][:]
                    data = data.T  # Transpose to match MATLAB's column-major order
                    result[key] = data
            return result


def load_result_matfile(path: str):
    data = loadmat(path)
    return data
