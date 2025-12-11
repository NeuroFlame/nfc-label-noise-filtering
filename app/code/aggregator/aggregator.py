import logging
import os
from typing import Dict, Any
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.fl_constant import ReservedKey

from utils.logger import NvFlareLogger
from utils.types import ConfigDTO
from utils.utils import get_output_directory_path

from . import methods as am

class DeNoiseAggregator(Aggregator):
    """
    DeNoiseAggregator handles the aggregation of results from multiple client sites.
    It stores individual site results and computes a global result based on the aggregation logic.

    This class can be customized if specific aggregation logic is needed.
    """

    def __init__(self):
        """
        Initializes the DeNoiseAggregator with a dictionary to store results from multiple sites.
        """
        super().__init__()
        # Store results as a dictionary
        self.site_results: Dict[int, Dict[str, Any]] = {}
        self.agg_cache: Dict[str, Any] = {}
        self.agg_cache_dir = ""
        self.logger = None

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """
        Accepts a result from a site and stores it for later aggregation.

        This method is called when a client site sends a result. Developers can override this 
        if they need to handle or validate the results differently before storing them.

        :param site_result: The result received from the client site.
        :param fl_ctx: The federated learning context for this run.
        :return: Boolean indicating if the result was successfully accepted.
        """
        site_name = site_result.get_peer_prop(
            key=ReservedKey.IDENTITY_NAME, default=None)
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND",
                                             default=None)

        if contribution_round is None or site_name is None:
            return False

        if contribution_round not in self.site_results:
            self.site_results[contribution_round] = {}

        if self.logger is None:
            log_level = fl_ctx.get_prop(key="log_level", default=None)
            logging.info(f'log_level for aggregator: {log_level}')
            self.logger = NvFlareLogger(
                'aggregator.log',
                get_output_directory_path(fl_ctx),
                fl_ctx.get_prop(key="log_level", default="info")
            )

            remote_path = get_output_directory_path(fl_ctx)
            self.agg_cache_dir = os.path.join(remote_path,
                                              'temp_agg_cache')
            os.makedirs(self.agg_cache_dir, exist_ok=True)

        # Store the result for the site using its identity name as the key
        self.site_results[contribution_round][site_name] = (
            site_result["result"]
        )

        self.logger.info('accepting site result from: ', site_name,
                         'from round: ', contribution_round)

        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregates the results from all accepted client sites and produces a global result.

        This is where the global aggregation logic happens. Developers can override this
        if they need to change how the results from each site are combined.

        :param fl_ctx: The federated learning context for this run.
        :return: A Shareable object containing the aggregated global result.
        """

        # Retrieve the decimal places from the computation parameters
        computation_parameters = fl_ctx.get_prop("COMPUTATION_PARAMETERS")
        outgoing_shareable = Shareable()
        contribution_round: int = fl_ctx.get_prop(key="CURRENT_ROUND",
                                             default=None)
        self.logger.info('aggregation round: ', contribution_round)

        config: ConfigDTO = ConfigDTO(
            data_path=None,
            cache_path=self.agg_cache_dir,
            computation_params=computation_parameters,
            site_name='remote',
            output_path=get_output_directory_path(fl_ctx=fl_ctx),
            logger=self.logger,
            cache_dict=self.agg_cache
        )
        try:
            if contribution_round == 0:
                agg_result = am.collect_typical_centroids(
                    self.site_results[contribution_round],
                    config
                )
                outgoing_shareable['result'] = agg_result['output']

            elif contribution_round == 1:
                agg_result = am.compute_adaptive_threshold(
                    self.site_results[contribution_round],
                    config
                )
                outgoing_shareable['result'] = agg_result['output']

            elif contribution_round == 2:
                am.relabelled_avg_fnc(
                    self.site_results[contribution_round],
                    config
                )
                config.cache_dict.clear()

            return outgoing_shareable
        except Exception as err:
            self.logger.error('Exception: ', err)
            self.logger.close()
            raise Exception(f'exception: {err}')
