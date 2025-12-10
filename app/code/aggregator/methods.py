import os
from typing import Dict, Any

from nvflare.apis.shareable import Shareable

from aggregator.helpers import global_mean_from_sites, fnc_heatmap
from utils.types import Centroids, ConfigDTO


def perform_remote_step_1(site_results: Dict[str, any], config: ConfigDTO):
    """
    Collect the n models from all the sites, then sent to all sites
    :param site_results: Description
    :type site_results: dict[str, any]
    """

    site_centroids: Dict[str, Centroids] = {}
    global_avg_per_label = {
        1: [],
        2: []
    }

    for site in site_results:
        site_centroids[site] = site_results[site]['centroids']

        for label in (1,2):
            global_avg_per_label[label].append(site_results[site]['aggregated_fnc_result'][label])

    # print(len(global_avg_per_label[1]), len(global_avg_per_label[2]))

    global_result_path = os.path.join(config.output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)


    global_avg_fnc = global_mean_from_sites(global_avg_per_label)
    for label in (1,2):
        fnc_heatmap(global_avg_fnc[label], {
            'colorbar_name': 'Avg FNC Values',
            'title': f'Average FNC of Original {group_names[label]} Subjects',
            'path': global_result_path,
            'name': f'global_original_avg_fnc_{group_names[label]}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

    return site_centroids

def perform_remote_step_3(shareable: Shareable, config: ConfigDTO):
    site_results: Dict[str, Any] = shareable.get('result')
    global_avg_per_label = {
        1: [],
        2: []
    }
    for site in site_results:
        for label in (1, 2):
            global_avg_per_label[label].append(site_results[site][label])

    labels = config.computation_params.get('Labels', {})
    global_result_path = os.path.join(config.output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)

    global_avg_fnc = global_mean_from_sites(global_avg_per_label)
    for label in labels:
        fnc_heatmap(global_avg_fnc[label], {
            'colorbar_name': 'Avg FNC Values',
            'title': f'Average FNC of Relabeled {labels[label]} Subjects',
            'path': global_result_path,
            'name': f'global_relabeled_avg_fnc_{labels[label]}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })