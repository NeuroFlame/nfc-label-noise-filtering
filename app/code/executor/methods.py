import os
import numpy as np
import pandas as pd

from numpy.random import Generator
from typing import Dict

from nvflare.apis.shareable import Shareable
from scipy.spatial.distance import cdist

from utils.data_loaders import load_result_matfile
from utils.types import ConfigDTO, Centroids
from .local_ancillary import find_group_differences, find_avg_fnc_measures
from .crf import crf
from .find_scores import get_centroids
from .validation import convert_fnc_to_features


def filter_typical_subjects(config: ConfigDTO, rng: Generator):
    """
    Perform local step 1
    """
    config.logger.info('##### filter_typical_subjects #####')

    config.logger.info('Step: Data conversion - In_Progress')
    data, aggregated_fnc_result = convert_fnc_to_features(config)
    config.logger.info('Step: Data conversion - Done')

    config.logger.info('Step: CRF - In_Progress')
    subject_noise_counts = crf.perform_crf(
        data, config.output_path, config.site_name, config.computation_params, rng=rng)
    config.logger.info('Step: CRF - Done')

    config.logger.info('Step: Centroids - In_Progress')
    centroids = get_centroids(
        data, subject_noise_counts, config.computation_params)
    config.logger.info('Step: Centroids - Done')

    selected_features_file = os.path.join(config.output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)

    data_file_path = os.path.join(config.output_path, 'data.npy')
    np.save(data_file_path, data)

    config.cache_dict.update({
        'data_file': data_file_path
    })

    output = {
        'centroids': centroids,
        'aggregated_fnc_result': aggregated_fnc_result
    }

    return {
        'output': output,
        'cache': config.cache_dict
    }

def find_inter_group_differences(shareable: Shareable, config: ConfigDTO):
    """

    """
    site_results: Dict[str, Centroids] = shareable.get('result')
    curr_site_data_path: str = config.cache_dict.get('data_file')
    curr_site_data: np.ndarray = np.load(curr_site_data_path)

    site_scores = pd.DataFrame(columns=list(site_results.keys()))
    config.logger.info('Step: Inter_Group_Diff - In_Progress for site', config.site_name)

    for site in site_results:
        config.logger.info('Performing for site: ', site)
        if site == config.site_name:
            site_scores[site] = curr_site_data[:, -1]
            continue

        site_group1_center: np.ndarray = site_results[site]['group1_center']
        site_group2_center: np.ndarray = site_results[site]['group2_center']
        site_selected_features = site_results[site]['selected_features']

        ind_selected_data = curr_site_data[:, site_selected_features]
        dist1 = cdist(ind_selected_data, site_group1_center)
        dist2 = cdist(ind_selected_data, site_group2_center)

        distance_typical_group1: float = dist1.mean(axis=1)
        distance_typical_group2: float = dist2.mean(axis=1)

        total_distance = distance_typical_group1 + distance_typical_group2

        A = distance_typical_group1 / total_distance
        B = distance_typical_group2 / total_distance

        scores = np.tan((A-B)*np.pi / 2)

        site_scores[site] = scores

    num_cols = site_scores.select_dtypes(
        include=["number"]).columns.drop(config.site_name)
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
    adaptive_score = shareable.get('result')
    scores_path = os.path.join(config.output_path, f'{config.site_name}.csv')
    scores_df = pd.read_csv(scores_path)

    total_subjects = scores_df.shape[0]
    re_labels = np.full(total_subjects, -1, dtype=float)
    mask_group1 = scores_df['average'] < -adaptive_score
    mask_group2 = scores_df['average'] > adaptive_score

    label_groups = config.computation_params.get('LabelGroups')

    re_labels[mask_group1] = label_groups['group1']['label']
    re_labels[mask_group2] = label_groups['group2']['label']

    scores_df['re_labeled'] = re_labels
    scores_df.to_csv(scores_path)

    """Two Sample t-test of relabeled subjects"""
    file_path = os.path.join(config.output_path, f'{config.site_name}.mat')
    original_dataset = load_result_matfile(file_path)
    X = original_dataset[config.site_name]

    label_groups = config.computation_params.get('LabelGroups')
    find_group_differences(X, re_labels, config.output_path, label_groups, False)

    """ Avg FNC Matrix of Relabeled Subjects """
    relabeled_aggregated_fnc_result = find_avg_fnc_measures(X, re_labels, config.output_path, label_groups, False)

    return {
        'output': {
            'aggregated_fnc_result': relabeled_aggregated_fnc_result
        }
    }