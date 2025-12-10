import logging
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils.logger import NvFlareLogger
from utils.types import ConfigDTO
from utils.utils import get_data_directory_path, get_output_directory_path

from . import cache_store as cache
from . import methods as helpers

class DeNoiseExecutor(Executor):
    def __init__(self):
        """
        Initialize the DeNoise. This constructor sets up the logger.
        """
        super().__init__()
        logging.info('DeNoise is running')

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        computation_params = get_computation_parameters(fl_ctx)
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        client_log_name = client_name + ".log"
        output_path = get_output_directory_path(fl_ctx)

        logger = NvFlareLogger(client_log_name, output_path,
                               computation_params.get('LogLevel'))

        cache_dict = cache.CacheSerialStore(
            get_output_directory_path(fl_ctx))

        config: ConfigDTO = ConfigDTO(
            data_path=get_data_directory_path(fl_ctx),
            output_path=output_path,
            cache_path=cache_dict.get_cache_dir(),
            computation_params=computation_params,
            logger=logger,
            site_name=client_name,
            cache_dict=cache_dict.get_cache_dict()
        )

        # Prepare the Shareable object to send the result to other
        # components
        outgoing_shareable = Shareable()


        return outgoing_shareable


def load_data(fl_ctx: FLContext):
    data_dir_path = get_data_directory_path(fl_ctx)
    data_file_filepath = os.path.join(data_dir_path, "data.json")
    logging.info(f"Loading data from: {data_file_filepath}")
    try:
        with open(data_file_filepath, "r") as file:
            data = json.load(file)
        validate_data_format(data)
        return data
    except FileNotFoundError:
        raise RuntimeError(f"Data file not found at {data_file_filepath}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in {data_file_filepath}")


def validate_data_format(data):
    """
    Validates that the data is a list of numbers.

    :param data: The data to validate.
    :raises ValueError: If the data is not a list of numbers.
    """
    if not isinstance(data, list) or not all(isinstance(item, (int, float)) for item in data):
        raise ValueError("Data must be a list of numbers")


def get_computation_parameters(fl_ctx: FLContext):
    return fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS", {"decimal_places": 2})


def save_results_to_file(results: dict, file_name: str, fl_ctx: FLContext):
    output_dir = get_output_directory_path(fl_ctx)
    logging.info(f"Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results successfully saved to: {os.path.join(output_dir, file_name)}")
    except Exception as e:
        raise RuntimeError(f"Failed to save results to {file_name}: {e}")
