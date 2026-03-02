import logging

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils.logger import NvFlareLogger
from utils.task_constants import PERFORM_LOCAL_CRF, DIMENSIONAL_SCORE, RELABEL_DATA
from utils.types import ConfigDTO
from utils.utils import get_computation_parameters, get_output_directory_path, get_data_directory_path

from numpy.random import SeedSequence, PCG64, Generator

from . import cache_store as cache
from . import methods as service


class LAMPExecutor(Executor):

    def __init__(self):
        """
        Initialize the FED_LAMP. This constructor sets up the logger.
        """
        super().__init__()
        logging.info('FED-LAMP is running')

    def execute(
            self,
            task_name: str,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
    ):
        """
        Main execution entry point. Routes tasks to specific methods based on the task name.

        Parameters:
            task_name: Name of the task to perform.
            shareable: Shareable object containing data for the task.
            fl_ctx: Federated learning context.
            abort_signal: Signal object to handle task abortion.

        Returns:
            A Shareable object containing results of the task.
        """

        computation_params = get_computation_parameters(fl_ctx)
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        client_log_name = client_name + ".log"
        output_path = get_output_directory_path(fl_ctx)

        logger = NvFlareLogger(client_log_name, output_path,
                               computation_params.get("LogLevel"))

        cache_store = cache.CacheSerialStore(
            get_output_directory_path(fl_ctx))

        config: ConfigDTO = ConfigDTO(
            data_path=get_data_directory_path(fl_ctx),
            output_path=output_path,
            cache_path=cache_store.get_cache_dir(),
            computation_params=computation_params,
            logger=logger,
            site_name=client_name,
            cache_dict=cache_store.get_cache_dict()
        )

        # Prepare the Shareable object to send the result to other
        # components
        outgoing_shareable = Shareable()

        try:
            if task_name == PERFORM_LOCAL_CRF:
                round_ss = SeedSequence(20250908)
                global_seed = int(round_ss.generate_state(1)[0])
                rng_round = Generator(PCG64(round_ss))

                client_result = service.perform_local_crf(config, rng_round, global_seed)
                cache_store.update_cache_dict(client_result.get('cache', {}))
                outgoing_shareable['result'] = client_result['output']

            elif task_name == DIMENSIONAL_SCORE:
                client_result = service.calculate_dimensional_score(shareable, config)
                cache_store.update_cache_dict(client_result.get('cache', {}))
                outgoing_shareable['result'] = client_result['output']

            elif task_name == RELABEL_DATA:
                client_result = service.relabel_data(shareable, config)
                cache_store.update_cache_dict(client_result.get('cache', {}))
                outgoing_shareable['result'] = client_result['output']
                cache_store.remove_cache()

            else:
                raise ValueError({
                    'message': 'Invalid task Name',
                    'value': task_name
                })
            return outgoing_shareable

        except Exception as err:
            config.logger.error('Exception: ', err)
            raise Exception(f'exception: {err}')
        finally:
            logger.close()
