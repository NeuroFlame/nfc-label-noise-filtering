import os
import random
from typing import Dict, Any

import numpy as np
import pandas as pd
from nvflare.apis.shareable import Shareable

from scipy.spatial.distance import cdist

from utils.helpers import fnc_heatmap
from utils.types import ConfigDTO
from . import data_loaders, find_scores
from Models import MatfileModel
from .crf import crf
from .helpers import (
    convert_fnc_to_features,
    compute_two_sample_ttest,
    upper_triangle_bonferroni,
    average_fnc_by_label
)
from numpy.random import Generator

def perform_local_crf(config: ConfigDTO, rng: Generator, global_seed: int = 0) -> Dict[str, Any]:
    """
        Perform local crf.
    """

    random.seed(global_seed)
    np.random.seed(global_seed)

    file_path = os.path.join(config.data_path, 'data.mat')
    original_dataset = data_loaders.load_data_matfile(
        file_path,
        name=[
            MatfileModel.SourceDataKeys.SFNC.value,
            MatfileModel.SourceDataKeys.FILE_ID.value,
            MatfileModel.SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )

    data = convert_fnc_to_features(original_dataset, config.output_path, config.site_name)

    # data_path = os.path.join(output_path, f'{site_name}.mat')
    # data = data_loaders.load_result_matfile(data_path)[site_name]

    X = original_dataset[MatfileModel.SourceDataKeys.SFNC.value]
    y = data[:, -1]

    """ Two Sample ttest of Original labels: """
    t_values, p_values = compute_two_sample_ttest(X, y)

    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)

    fnc_heatmap(
        corrected_t_values,
        {
            'colorbar_name': 'T Values',
            'title': 'Original HC Vs SZ T-test values',
            'path': config.output_path,
            'name': 'original_labels_ttest.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        }
    )

    """ Average FNC matrix """
    unique_labels = np.unique(y).astype(int)
    avg_fnc, aggregated_fnc_result = average_fnc_by_label(X, y, unique_labels)
    parameters = config.computation_params

    for label in unique_labels:
        str_label = str(label)
        label_name = parameters.get("LabelDefinition").get(str_label)
        fnc_heatmap(avg_fnc[label], {
            'colorbar_name': "Avg FNC Values",
            'title': f'Average FNC of Original {label_name} Subjects',
            'path': config.output_path,
            'name': f'local_original_avg_fnc_{label_name}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

    subject_noise_counts = crf.perform_crf(
        data, config.output_path, config.site_name, parameters, rng=rng)

    # crf_file = os.path.join(output_path, f'{site_name}_CRF.mat')
    # subject_noise_counts = data_loaders.load_result_matfile(crf_file)['count'][:]

    centroids = find_scores.get_centroids(
        data, subject_noise_counts, parameters.get("TypicalThreshold"))

    selected_features_file = os.path.join(config.output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)

    output = {
        'centroids': centroids,
        'aggregated_fnc_result': aggregated_fnc_result
    }

    return {
        "output": output,
    }

def calculate_dimensional_score(shareable: Shareable, config: ConfigDTO):
    site_results = shareable.get('result')
    out_path = os.path.join(config.output_path, f'{config.site_name}.mat')
    ind_site_data: np.ndarray = data_loaders.load_result_matfile(out_path)
    site_scores = pd.DataFrame(columns=list(site_results.keys()))

    for site in site_results:
        if site == config.site_name:
            site_scores[site] = ind_site_data[:, -1]
            continue

        main_sz: np.ndarray = site_results[site]['center_sz']
        main_hc: np.ndarray = site_results[site]['center_hc']
        main_selected_features = site_results[site]['selected_features']

        ind_selected_data = ind_site_data[:, main_selected_features]
        dist1 = cdist(ind_selected_data, main_sz)
        dist2 = cdist(ind_selected_data, main_hc)

        distance_typical_group_sz: float = dist1.mean(axis=1)
        distance_typical_group_hc: float = dist2.mean(axis=1)

        total_distance = distance_typical_group_sz + distance_typical_group_hc

        A = distance_typical_group_sz / total_distance
        B = distance_typical_group_hc / total_distance

        scores = np.tan((A-B)*np.pi / 2)

        site_scores[site] = scores

    num_cols = site_scores.select_dtypes(
        include=[np.number]).columns.drop(config.site_name)
    site_scores["average"] = site_scores[num_cols].replace(
        0, np.nan).mean(axis=1, skipna=True)
    site_scores["average"] = site_scores["average"].fillna(0)

    output_file = os.path.join(config.output_path, f'{config.site_name}.csv')
    site_scores.to_csv(output_file)

    return {
        'output': np.array(site_scores['average'])
    }

def relabel_data(shareable: Shareable, config: ConfigDTO):
    """
        Compute Relabeled for every subject
    """
    adaptive_score = shareable.get("result")
    scores_path = os.path.join(config.output_path, f'{config.site_name}.csv')
    scores_df = pd.read_csv(scores_path)

    total_subjects = scores_df.shape[0]
    re_labels = np.full(total_subjects, -1, dtype=float)
    mask_sz = scores_df['average'] < -adaptive_score
    mask_hc = scores_df['average'] > adaptive_score

    re_labels[mask_sz] = 1.0
    re_labels[mask_hc] = 2.0

    scores_df['re_labeled'] = re_labels
    scores_df.to_csv(scores_path)

    """Two Sample t-test of relabeled subjects"""
    file_path = os.path.join(config.data_path, 'data.mat')
    original_dataset = data_loaders.load_data_matfile(
        file_path,
        name=[
            MatfileModel.SourceDataKeys.SFNC.value,
        ],
    )
    X = original_dataset[MatfileModel.SourceDataKeys.SFNC.value]
    t_values, p_values = compute_two_sample_ttest(X, re_labels)
    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)
    fnc_heatmap(
        corrected_t_values,
        {
            'colorbar_name': 'T Values',
            'title': 'Relabeled HC Vs SZ T-test values',
            'path': config.output_path,
            'name': 're_labeled_ttest.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        }
    )

    """ Avg FNC Matrix of Relabeled Subjects """
    unique_labels = np.unique(re_labels).astype(int)
    avg_fnc, relabeled_aggregated_fnc_result = average_fnc_by_label(X, re_labels, unique_labels)
    parameters = config.computation_params

    for label in unique_labels:
        if label == -1:
            continue

        str_label = str(label)
        label_name = parameters.get("LabelDefinition").get(str_label)
        fnc_heatmap(avg_fnc[label], {
            'colorbar_name': "Avg FNC Values",
            'title': f'Average FNC of Relabeled {label_name} Subjects',
            'path': config.output_path,
            'name': f'local_relabeled_avg_fnc_{label_name}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

    return {
        'output': relabeled_aggregated_fnc_result
    }