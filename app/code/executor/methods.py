import os
import numpy as np
import pandas as pd
from numpy.random import SeedSequence, PCG64, Generator
from typing import Dict, Any

from nvflare.apis.shareable import Shareable
from scipy.spatial.distance import cdist

from utils.data_loaders import load_data_matfile
from utils.types import SourceDataKeys, ConfigDTO, Centroids
from local_ancillary import convert_fnc_to_features, compute_two_sample_ttest, \
    upper_triangle_bonferroni, fnc_heatmap, average_fnc_by_label, global_mean_from_sites
from crf import crf
from find_scores import get_centroids


def filter_typical_subjects(config: ConfigDTO, rng: Generator):
    """
    Perform local step 1
    """
    file_path = os.path.join(config.data_path, 'data.mat')
    original_dataset = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
            SourceDataKeys.FILE_ID.value,
            SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )

    data = convert_fnc_to_features(original_dataset, config.output_path, config.site_name)

    # data_path = os.path.join(output_path, f'{site_name}.mat')
    # data = data_loaders.load_result_matfile(data_path)[site_name]

    X = original_dataset[SourceDataKeys.SFNC.value]
    y = data[:, -1]

    """ Two Sample ttest of Original labels: """
    t_values, p_values = compute_two_sample_ttest(X, y)

    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)

    fnc_heatmap(
        corrected_t_values,
        {
            'colorbar_name' : 'T Values',
            'title': 'Original HC Vs SZ T-test values',
            'path': config.output_path,
            'name': 'original_labels_ttest.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        }
    )

    """ Average FNC matrix """
    labels = config.computation_params.get('Labels', {})
    avg_fnc, aggregated_fnc_result = average_fnc_by_label(X, y, np.array(labels.keys()))

    for label in labels:
        fnc_heatmap(avg_fnc[label], {
            'colorbar_name' : "Avg FNC Values",
            'title': f'Average FNC of Original {labels[label]} Subjects',
            'path': config.output_path,
            'name': f'local_original_avg_fnc_{labels[label]}.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        })

    subject_noise_counts = crf.perform_crf(
        data, config.output_path, config.site_name, config.computation_params, rng=rng)

    # crf_file = os.path.join(output_path, f'{site_name}_CRF.mat')
    # subject_noise_counts = data_loaders.load_result_matfile(crf_file)['count'][:]

    centroids = get_centroids(
        data, subject_noise_counts, config.computation_params.get('TypicalThs', 0.8))

    selected_features_file = os.path.join(config.output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)

    config.cache_dict.update({
        'data': data
    })

    output = {
        'centroids': centroids,
        'aggregated_fnc_result': aggregated_fnc_result
    }

    return {
        'output': output,
        'cache': config.cache_dict
    }

def find_inter_group_differences(sharable: Shareable, config: ConfigDTO):
    """
    Docstring for perform_local_step_2

    :param site_name: Description
    :param site_results: Description
    :param output_path: str
    """
    site_results: Dict[str, Centroids] = sharable.get('result')
    ind_site_data: np.ndarray = config.cache_dict.get('data') # TODO: Don't store it as cache, better to load again.
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

def perform_relabelling(shareable: Shareable, config: ConfigDTO):
    """
    Compute Relabeled for every subject
    """
    adaptive_score = shareable.get('adaptive_score')
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
    original_dataset = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
        ],
    )
    X = original_dataset[SourceDataKeys.SFNC.value]
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
    labels = config.computation_params.get('Labels', {})
    avg_fnc, relabeled_aggregated_fnc_result = average_fnc_by_label(X, re_labels, np.array(labels.keys()))

    for label in labels:
        if label == -1:
            continue
        fnc_heatmap(avg_fnc[label], {
            'colorbar_name' : "Avg FNC Values",
            'title': f'Average FNC of Relabeled {labels[label]} Subjects',
            'path': config.output_path,
            'name': f'local_relabeled_avg_fnc_{labels[label]}.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        })

    return {
        'output': relabeled_aggregated_fnc_result
    }