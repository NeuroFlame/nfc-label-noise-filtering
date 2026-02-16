import json
import base64
import numpy as np

def decode_json_to_npy(json_string):
    """
    Decodes a JSON string containing base64 encoded data to a .npy file.

    Args:
        json_string (str): A JSON string containing the base64 encoded data.

    Returns:
        numpy.ndarray: The decoded numpy array, or None if an error occurred.
    """
    try:
        json_data = json.loads(json_string)
        data_base64 = json_data['data']
        shape = json_data['shape']
        dtype = json_data['dtype']
        data_bytes = base64.b64decode(data_base64)
        data_array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        return data_array
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        return None

def encode_npy_to_json(file_path):
    """
    Encodes a .npy file to a JSON string with base64 encoding.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        str: A JSON string containing the base64 encoded data.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        data_bytes = data.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode('utf-8')
        json_data = json.dumps(
            {'data': data_base64, 'shape': data.shape, 'dtype': str(data.dtype)})
        return json_data
    except Exception as e:
        return json.dumps({'error': str(e)})
