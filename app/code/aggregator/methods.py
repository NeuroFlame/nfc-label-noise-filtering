import os
from typing import Any, Dict

import numpy as np

from Models.CentroidModel import Centroids
from utils.helpers import global_mean_from_sites, fnc_heatmap
from utils.types import ConfigDTO

def collect_local_models(site_results: Dict[str, Any], config: ConfigDTO):
    """
        Collect the n models from all the sites, then sent to all sites
        :param site_results: Description
        :type site_results: dict[str, any]
    Args:
        config: Config values
    """

    site_centroids: dict[str, Centroids] = {}
    global_avg_per_label = {}

    labels = list()
    for key in config.computation_params.get('LabelDefinition').keys():
        label = config.computation_params.get('LabelDefinition').get(key).get('label')
        labels.append(label)
        global_avg_per_label[label] = list()

    for site in site_results:
        site_centroids[site] = site_results[site]['centroids']

        for label in labels:
            global_avg_per_label[label].append(site_results[site]['aggregated_fnc_result'][label])

    # print(len(global_avg_per_label[1]), len(global_avg_per_label[2]))

    global_result_path = os.path.join(config.output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)
    parameters = config.computation_params

    global_avg_fnc = global_mean_from_sites(global_avg_per_label)
    for label in labels:
        str_label = str(label)
        label_name = parameters.get("LabelDefinition").get(str_label).get('name')
        fnc_heatmap(global_avg_fnc[label], {
            'colorbar_name': 'Avg FNC Values',
            'title': f'Average FNC of Original {label_name} Subjects',
            'path': global_result_path,
            'name': f'global_original_avg_fnc_{label_name}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

    return {
        'output': site_centroids,
        'cache': {'global_avg_fnc_original': global_avg_fnc},
    }

def perform_adaptive_thresolding(site_results: Dict[str, np.ndarray], config: ConfigDTO):
    """
    collect scores from all the sites and then compute adaptive threshold (t)

    :param site_results: Description
    :type site_results: dict[str, any]
    """

    global_scores = np.concatenate(list(site_results.values()))
    computation_params = config.computation_params

    # ------------- prepare sorted view (for computing t only) -------------
    valid_mask = ~np.isnan(global_scores)
    valid_idx = np.nonzero(valid_mask)[0]
    s_valid = global_scores[valid_mask]

    order = np.argsort(s_valid, kind="mergesort")
    com = s_valid[order]  # sorted valid scores (ascending)

    # counts (exclude exact zeros for picking reference indices)
    neg_mask_sorted = com < 0
    pos_mask_sorted = com > 0
    N_neg = int(np.sum(neg_mask_sorted))
    N_pos = int(np.sum(pos_mask_sorted))

    # MATLAB-like nearest-integer rounding, then clamp
    def round_clamp(x: float, lo: int, hi: int) -> int:
        idx = int(np.floor(x + 0.5))
        return max(lo, min(hi, idx))

    i_neg_1b = None
    i_pos_1b = None

    i_neg0 = -1
    i_pos0 = -1

    if N_neg > 0:
        i_neg_1b = round_clamp(N_neg * (1.0 - computation_params.get("TruncationParameter")), 1, N_neg)  # 1..N_neg
        i_neg0 = i_neg_1b - 1  # 0-based in com
    if N_pos > 0:
        i_pos_1b = N_neg + round_clamp(N_pos * computation_params.get("TruncationParameter"), 1, N_pos)  # (N_neg+1)..(N_neg+N_pos)
        i_pos0 = i_pos_1b - 1

    if i_neg0 > 0 and i_pos0 > 0:
        t = abs(com[i_neg0] + com[i_pos0]) / 2.0
    elif i_neg0 > 0:
        t = abs(com[i_neg0])
    elif i_pos0 > 0:
        t = abs(com[i_pos0])
    else:
        t = 0.0

    if not np.isfinite(t):
        t = 0.0
    t = float(max(0.0, t))
    print("adaptive thresholding: ", t)

    return {
        'output': t,
        'cache': {'adaptive_threshold': t},
    }

def print_relabeled_metrics(site_results: Dict[str, np.ndarray], config: ConfigDTO):
    global_avg_per_label = {}
    labels = list()
    for key in config.computation_params.get('LabelDefinition').keys():
        label = config.computation_params.get('LabelDefinition').get(key).get('label')
        labels.append(label)
        global_avg_per_label[label] = list()

    for site in site_results:
        for label in labels:
            global_avg_per_label[label].append(site_results[site][label])

    global_avg_fnc_relabeled = global_mean_from_sites(global_avg_per_label)

    return {
        'output': {
            'global_avg_fnc_original': config.cache_dict.get('global_avg_fnc_original', {}),
            'global_avg_fnc_relabeled': global_avg_fnc_relabeled,
            'adaptive_threshold': config.cache_dict.get('adaptive_threshold', 0.0),
        }
    }