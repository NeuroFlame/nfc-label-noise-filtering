import os
import numpy as np
import pandas as pd
import random

from scipy.spatial.distance import cdist

from executor.crf import crf
from app.code.Models.CentroidModel import Centroids
from utils import utils
from app.code.executor.find_scores import get_centroids
from app.code.executor.data_loaders import load_data_matfile

from numpy.random import SeedSequence, PCG64, Generator

# PARAMETERS
SamplingThs = 0.7
iter = 101
ntree = 201
NI_threshold = 2
TypThs = 0.8
TruncationParameter = 0.2
parameters = {
    'sampling_threshold': SamplingThs,
    'iter': iter,
    'ntree': ntree,
    'label_threshold': NI_threshold,
    'typical_threshold': TypThs
}

group_names = { 1: 'SZ', 2: 'HC'}

# INPUTSPEC VALUES
BASE_INPUT_DATA = "test_data/{site}"
BASE_OUTPUT_DIR = "test_output/round_{number}/{site}"
BASE_CACHE_DIR = "test/cache"
SUB_DATA = "data.mat"

# VALUES which are not in inputspec but qualified for it.
LAMBDA = 0

AGGREGATOR_CACHE: dict[str, any] = {}  # REMOTE CACHE
EXECUTOR_CACHE: dict[str, dict] = {}  # LOCAL CACHE


def perform_local_step_1(site_name: str, data_path: str, output_path: str, rng: Generator):
    """
    Perform local step 1
    """

    file_path = os.path.join(data_path, 'data.mat')
    original_dataset = load_data_matfile(
        file_path,
        name=[
            utils.SourceDataKeys.SFNC.value,
            utils.SourceDataKeys.FILE_ID.value,
            utils.SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )

    data = utils.convert_fnc_to_features(original_dataset, output_path, site_name)

    # data_path = os.path.join(output_path, f'{site_name}.mat')
    # data = data_loaders.load_result_matfile(data_path)[site_name]

    X = original_dataset[utils.SourceDataKeys.SFNC.value]
    y = data[:, -1]

    """ Two Sample ttest of Original labels: """
    t_values, p_values = utils.compute_two_sample_ttest(X, y)

    corrected_t_values = utils.upper_triangle_bonferroni(t_values, p_values)

    utils.fnc_heatmap(
        corrected_t_values,
        {
            'colorbar_name' : 'T Values',
            'title': 'Original HC Vs SZ T-test values',
            'path': output_path,
            'name': 'original_labels_ttest.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        }
    )

    """ Average FNC matrix """
    unique_labels = np.unique(y).astype(int) # TODO: Need to replace with configurations
    avg_fnc, aggregated_fnc_result = utils.average_fnc_by_label(X, y, unique_labels)

    for label in unique_labels:
        utils.fnc_heatmap(avg_fnc[label], {
            'colorbar_name' : "Avg FNC Values",
            'title': f'Average FNC of Original {group_names[label]} Subjects',
            'path': output_path,
            'name': f'local_original_avg_fnc_{group_names[label]}.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        })

    subject_noise_counts = crf.perform_crf(
        data, output_path, site_name, parameters, rng=rng)

    # crf_file = os.path.join(output_path, f'{site_name}_CRF.mat')
    # subject_noise_counts = data_loaders.load_result_matfile(crf_file)['count'][:]

    centroids = get_centroids(
        data, subject_noise_counts, parameters["typical_threshold"])

    selected_features_file = os.path.join(output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)

    EXECUTOR_CACHE[site_name] = {
        'data': data
    }

    return {
        'centroids': centroids,
        'aggregated_fnc_result': aggregated_fnc_result
    }

def perform_local_step_2(site_name: str, site_results: dict[str, Centroids], output_path: str):
    """
    Docstring for perform_local_step_2

    :param site_name: Description
    :param site_results: Description
    :param output_path: str
    """
    ind_site_data: np.ndarray = EXECUTOR_CACHE[site_name]['data'] # TODO: Don't store it as cache, better to load again.
    site_scores = pd.DataFrame(columns=list(site_results.keys()))

    for site in site_results:
        if site == site_name:
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
        include=[np.number]).columns.drop(site_name)
    site_scores["average"] = site_scores[num_cols].replace(
        0, np.nan).mean(axis=1, skipna=True)
    site_scores["average"] = site_scores["average"].fillna(0)

    output_file = os.path.join(output_path, f'{site_name}.csv')
    site_scores.to_csv(output_file)

    return np.array(site_scores['average'])

def perform_remote_step_1(site_results: dict[str, any], output_path: str):
    """
    Collect the n models from all the sites, then sent to all sites
    :param site_results: Description
    :type site_results: dict[str, any]
    """

    site_centroids: dict[str, Centroids] = {}
    global_avg_per_label = {
        1: [],
        2: []
    }

    for site in site_results:
        site_centroids[site] = site_results[site]['centroids']

        for label in (1,2):
            global_avg_per_label[label].append(site_results[site]['aggregated_fnc_result'][label])

    # print(len(global_avg_per_label[1]), len(global_avg_per_label[2]))

    global_result_path = os.path.join(output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)

    global_avg_fnc = utils.global_mean_from_sites(global_avg_per_label)
    for label in (1,2):
        utils.fnc_heatmap(global_avg_fnc[label], {
            'colorbar_name': 'Avg FNC Values',
            'title': f'Average FNC of Original {group_names[label]} Subjects',
            'path': global_result_path,
            'name': f'global_original_avg_fnc_{group_names[label]}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

    return site_centroids

def perform_remote_step_2(site_results: dict[str, np.ndarray]):
    """
    collect scores from all the sites and then compute adaptive threshold (t)

    :param site_results: Description
    :type site_results: dict[str, any]
    """

    global_scores = np.concatenate(list(site_results.values()))

    # ------------- prepare sorted view (for computing t only) -------------
    valid_mask = ~np.isnan(global_scores)
    valid_idx = np.nonzero(valid_mask)[0]
    s_valid = global_scores[valid_mask]

    order = np.argsort(s_valid, kind="mergesort")
    com = s_valid[order]          # sorted valid scores (ascending)

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
        i_neg_1b = round_clamp(N_neg * (1.0 - TruncationParameter), 1, N_neg)          # 1..N_neg
        i_neg0 = i_neg_1b - 1                                        # 0-based in com
    if N_pos > 0:
        i_pos_1b = N_neg + round_clamp(N_pos * TruncationParameter, 1, N_pos)          # (N_neg+1)..(N_neg+N_pos)
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
    print(t)
    
    return t

def perform_remote_step_3(site_results: dict[str, np.ndarray], output_path: str):
    global_avg_per_label = {
        1: [],
        2: []
    }

    for site in site_results:
        for label in (1,2):
            global_avg_per_label[label].append(site_results[site][label])

    global_result_path = os.path.join(output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)

    global_avg_fnc = utils.global_mean_from_sites(global_avg_per_label)
    for label in (1,2):
        utils.fnc_heatmap(global_avg_fnc[label], {
            'colorbar_name': 'Avg FNC Values',
            'title': f'Average FNC of Relabeled {group_names[label]} Subjects',
            'path': global_result_path,
            'name': f'global_relabeled_avg_fnc_{group_names[label]}.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        })

def perform_local_step_3(site: str, data_path: str, output_path: str, adaptive_score):
    """
    Compute Relabeled for every subject
    """
    scores_path = os.path.join(output_path, f'{site}.csv')
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
    file_path = os.path.join(data_path, 'data.mat')
    original_dataset = data_loaders.load_data_matfile(
        file_path,
        name=[
            utils.SourceDataKeys.SFNC.value,
        ],
    )
    X = original_dataset[utils.SourceDataKeys.SFNC.value]
    t_values, p_values = utils.compute_two_sample_ttest(X, re_labels)
    corrected_t_values = utils.upper_triangle_bonferroni(t_values, p_values)
    utils.fnc_heatmap(
        corrected_t_values,
        {
            'colorbar_name': 'T Values',
            'title': 'Relabeled HC Vs SZ T-test values',
            'path': output_path,
            'name': 're_labeled_ttest.png',
            'domain_names': [0, 5, 7, 16, 25, 42, 49, 53],
        }
    )

    """ Avg FNC Matrix of Relabeled Subjects """
    unique_labels = np.unique(re_labels).astype(int)
    avg_fnc, relabeled_aggregated_fnc_result = utils.average_fnc_by_label(X, re_labels, unique_labels)

    for label in unique_labels:
        if label == -1:
            continue
        utils.fnc_heatmap(avg_fnc[label], {
            'colorbar_name' : "Avg FNC Values",
            'title': f'Average FNC of Relabeled {group_names[label]} Subjects',
            'path': output_path,
            'name': f'local_relabeled_avg_fnc_{group_names[label]}.png',
            'domain_names': [0,5,7,16,25,42,49,53],
        })

    return relabeled_aggregated_fnc_result

def run_federated_de_noise(round_id: int):

    round_ss = SeedSequence(20250908)
    global_seed = int(round_ss.generate_state(1)[0])

    random.seed(global_seed)
    np.random.seed(global_seed)

    rng_round = Generator(PCG64(round_ss))

    """Controller"""
    sites = ['site1', 'site2']  # check for consistency

    # Iteration - 1
    site_results = {}
    for site in sites:
        base_path = BASE_INPUT_DATA.format(site=site)
        output_path = BASE_OUTPUT_DIR.format(number=round_id, site=site)
        os.makedirs(output_path, exist_ok=True)
        site_results[site] = perform_local_step_1(
            site, base_path, output_path, rng=rng_round)

    agg_results = perform_remote_step_1(site_results, "test_output/round_5")

    score_site_results: dict[str, np.ndarray] = {}
    for site in sites:
        output_path = BASE_OUTPUT_DIR.format(number=round_id, site=site)
        score_site_results[site] = perform_local_step_2(
            site,
            agg_results,
            output_path
        )

    adaptive_score = perform_remote_step_2(score_site_results)
    print(adaptive_score)

    site_avg_fnc_data = {}
    for site in sites:
        base_path = BASE_INPUT_DATA.format(site=site)
        output_path = BASE_OUTPUT_DIR.format(number=round_id, site=site)
        site_avg_fnc_data[site] = perform_local_step_3(site, base_path, output_path, adaptive_score)
    perform_remote_step_3(site_avg_fnc_data, "test_output/round_5")

CURRENT_EXECUTION_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "5"))

if CURRENT_EXECUTION_ID == 0:
    raise Exception("Failed to get execution id")

run_federated_de_noise(CURRENT_EXECUTION_ID)
