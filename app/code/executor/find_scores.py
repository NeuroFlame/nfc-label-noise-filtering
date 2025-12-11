from typing import Mapping, Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from .local_ancillary import find_typical_subjects
from utils.types import Centroids


# cummulative feature selection using Bonferroni corrected threshold 0.01/(Col-1)
def cumulative_features_selection(pval, pvalPara):
    # Pval: 1-D array of p-values
    FeaInd = np.argsort(pval)                  # indices sorted by p-value
    SortPval = pval[FeaInd]                    # sorted p-values

    # Find first index where p-value exceeds threshold
    above_thresh = np.where(SortPval > pvalPara)[0]
    if above_thresh.size > 0:
        ind = above_thresh[0]                  # first index above threshold
    else:
        ind = len(SortPval)                    # all p-values below threshold

    Fea = FeaInd[:ind]                         # keep only significant features
    return Fea


def compute_score(independent_data, typical_data):  # -> Any:
    col = independent_data.shape[1]

    ind_fea = independent_data[:, :-1]
    typical_data_features = typical_data[:, : -1]

    typical_data_labels = typical_data[:, -1]
    typical_group_sz = typical_data_features[typical_data_labels == 1, :]
    typical_group_hc = typical_data_features[typical_data_labels == 2, :]

    t_stat, p_val = stats.ttest_ind(
        typical_group_sz, typical_group_hc, axis=0, equal_var=True)

    # print('pvals: ', p_val)

    significant_threshold = 0.01 / (col - 1)

    selected_features = cumulative_features_selection(
        p_val, significant_threshold)
    # print('selected features: ', selected_features)

    center_sz = np.mean(
        typical_group_sz[:, selected_features], axis=0).reshape(1, -1)
    center_hc = np.mean(
        typical_group_hc[:, selected_features], axis=0).reshape(1, -1)

    # print(center_sz, center_hc)

    X = ind_fea[:, selected_features]
    dist1 = cdist(X, center_sz)
    dist2 = cdist(X, center_hc)

    distance_typical_group_sz = dist1.mean(axis=1)
    distance_typical_group_hc = dist2.mean(axis=1)

    total_distance = distance_typical_group_sz + distance_typical_group_hc

    A = distance_typical_group_sz / total_distance
    B = distance_typical_group_hc / total_distance

    scores = np.tan((A-B)*np.pi / 2)
    # print('final_scores: ', scores)

    return scores


def get_centroids(
    subject_data: np.ndarray,
    subject_label_count: np.ndarray,
    config: Mapping[str, Any],
) -> Centroids:

    col = subject_data.shape[1]
    typ_group1, typ_group2 = find_typical_subjects(
        subject_data[:, -1], subject_label_count, config)

    typical_group1_data = subject_data[typ_group1, :-1]
    typical_group2_data = subject_data[typ_group2, :-1]

    t_stat, p_val = stats.ttest_ind(
        typical_group1_data, typical_group2_data, axis=0, equal_var=True)

    significant_threshold = 0.01 / (col - 1)

    selected_features = cumulative_features_selection(
        p_val, significant_threshold)

    center_group1: np.ndarray = np.mean(
        typical_group1_data[:, selected_features], axis=0).reshape(1, -1)
    center_group2: np.ndarray = np.mean(
        typical_group2_data[:, selected_features], axis=0).reshape(1, -1)

    return {
        "group1_center": center_group1,
        "group2_center": center_group2,
        "selected_features": selected_features,
        'group1_typ_subjects': typ_group1,
        'group2_typ_subjects': typ_group2
    }
