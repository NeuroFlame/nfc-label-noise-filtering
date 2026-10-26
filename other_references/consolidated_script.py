import os
import numpy as np
import pandas as pd
import random

from scipy.spatial.distance import cdist

from app.code.computation.crf import crf
from app.code.computation.types import CentroidSet, ValidatedInputs
from app.code.computation.local_math import (compute_two_sample_ttest, upper_triangle_bonferroni,
                                             average_fnc_by_label,get_centroids,
                                             compute_local_crf)
from app.code.computation.reports import render_fnc_heatmap
from app.code.computation.remote_math import global_mean_from_sites

from consolidated_data_loaders import load_data_matfile, load_result_matfile, SourceDataKeys

from numpy.random import SeedSequence, PCG64, Generator




# PARAMETERS
SamplingThs = 0.7
iter = 2
ntree = 201
NI_threshold = 2
TypThs = 0.8
TruncationParameter = 0.2
# parameters = {
#     'sampling_threshold': SamplingThs,
#     'iter': iter,
#     'ntree': ntree,
#     'label_threshold': NI_threshold,
#     'typical_threshold': TypThs
# }

# parameters.update({
#     'SamplingThreshold': SamplingThs,
#     'Iteration': iter,
#     'NTree': ntree,
#     'LabelThreshold': NI_threshold,
#     'TypicalThreshold': TypThs
# })

parameters = {
  "SamplingThreshold": SamplingThs,
  "Iteration": iter,
  "NTree": ntree,
  "LabelThreshold": NI_threshold,
  "TypicalThreshold": TypThs,
  "TruncationParameter": 0.2,
  "LabelDefinition": {
    "1": {
      "name": "SZ",
      "label": 1
    },
    "2": {
      "name": "HC",
      "label": 2
    }
  },
  "isControlLabel": "HC",
  "FNCDomainNames" :  [0, 5, 7, 16, 25, 42, 49, 53],
  "IgnoreSubjectsWithMissingData" : True,
  "LogLevel": "info"
}

group_names = {1: 'SZ', 2: 'HC'}

# INPUTSPEC VALUES
BASE_INPUT_DATA = "../test_data_Old/{site}"
BASE_OUTPUT_DIR = "test_output_cen_test/round_{number}/{site}"
BASE_CACHE_DIR = "test/cache"
SUB_DATA = "data.mat"

"""Controller"""
sites = ['site1', 'site2']  # check for consistency

# VALUES which are not in inputspec but qualified for it.
LAMBDA = 0

AGGREGATOR_CACHE: dict[str, any] = {}  # REMOTE CACHE
EXECUTOR_CACHE: dict[str, dict] = {}  # LOCAL CACHE


def convert_fnc_to_features_from_mat(original_dataset, dest_path: str, name: str):
    from scipy.io import savemat

    # --- find diagnosis column (case-insensitive) ---
    file_ids = original_dataset[SourceDataKeys.FILE_ID.value]
    label_index = next(
        (i for i, col in enumerate(file_ids) if "diagnosis" in col.lower()),
        None
    )
    if label_index is None:
        raise ValueError(f'No "diagnosis" column found in FILE_ID for {name}')

    labels = original_dataset[SourceDataKeys.ANALYSIS_SCORE.value][:, label_index]
    labels = labels.reshape(-1, 1)

    fnc_matrices = original_dataset[SourceDataKeys.SFNC.value]  # shape: (N, P, P)
    N, P, _ = fnc_matrices.shape

    # --- build the lower-triangle mask excluding diagonal (k=-1) ---
    mask = np.tril(np.ones((P, P), dtype=bool), k=-1)  # P x P

    linear_idx = np.where(mask.ravel(order='F'))[0]           # size: P*(P-1)/2
    fnc_flat_F = fnc_matrices.reshape(N, P * P, order='F')    # N x (P*P)

    source_data = fnc_flat_F[:, linear_idx]                   # N x (P*(P-1)/2)

    # append labels as the last column
    out = np.hstack([source_data, labels])

    # save with variable name = dataset name (like MATLAB)
    out_path = os.path.join(dest_path, f'{name}.mat')
    savemat(out_path, {name: out}, do_compression=True)

    return out


def perform_local_step_1(site_name: str, data_path: str, output_path: str, rng: Generator):
    """
    Perform local step 1
    """

    file_path = os.path.join(data_path, 'data.mat')
    original_dataset = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
            SourceDataKeys.FILE_ID.value,
            SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )

    data = convert_fnc_to_features_from_mat(original_dataset, output_path, site_name)

    # data_path = os.path.join(output_path, f'{site_name}.mat')
    # data = data_loaders.load_result_matfile(data_path)[site_name]

    X = original_dataset[SourceDataKeys.SFNC.value]
    y = data[:, -1]

    """ Two Sample ttest of Original labels: """
    t_values, p_values = compute_two_sample_ttest(X, y)

    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)

    render_fnc_heatmap(corrected_t_values, output_path,
                       name='original_labels_ttest.png',
                       title= 'Original HC Vs SZ T-test values',
                       domain_names=[0, 5, 7, 16, 25, 42, 49, 53]
                       )

    """ Average FNC matrix """
    unique_labels = np.unique(y).astype(int)  # TODO: Need to replace with configurations
    avg_fnc, aggregated_fnc_result = average_fnc_by_label(X, y, unique_labels)

    for label in unique_labels:
        render_fnc_heatmap(avg_fnc[label], output_path,
                           name=f'local_original_avg_fnc_{group_names[int(label)]}.png',
                           title=f'Average FNC of Original {group_names[int(label)]} Subjects',
                           domain_names=[0, 5, 7, 16, 25, 42, 49, 53]
                           )



    crf_file = os.path.join(output_path, f'crf.mat')

    if not os.path.isfile(crf_file):
        subject_noise_counts = crf.perform_crf(data, output_path, parameters, rng=rng)

    subject_noise_counts = load_result_matfile(crf_file)['count'][:]

    centroids = get_centroids(
        data, subject_noise_counts, parameters["TypicalThreshold"])

    selected_features_file = os.path.join(output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)

    fnc_sum = {str(label): aggregated_fnc_result[label]["sum"] for label in unique_labels}
    fnc_count = {
        str(label): aggregated_fnc_result[label]["count"] for label in unique_labels
    }

    EXECUTOR_CACHE[site_name] = {
        'data': data
    }

    return {
        'centroids': centroids,
        'aggregated_fnc_result': aggregated_fnc_result,
        'fnc_sum': fnc_sum,
        'fnc_count':fnc_count
    }


def perform_local_step_1_test(site_name: str, data_path: str, output_path: str, rng: Generator):
    """
       Perform local step 1
       """

    file_path = os.path.join(data_path, 'data.mat')
    original_dataset = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
            SourceDataKeys.FILE_ID.value,
            SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )
    inputs = ValidatedInputs(data=convert_fnc_to_features_from_mat(original_dataset, output_path, site_name),
                             self_token=f"{site_name}")


    results = compute_local_crf(inputs, parameters = parameters, output_dir=output_path, logger=None)
    return results.payload

def perform_local_step_2(site_name: str, site_results: dict[str, CentroidSet], output_path: str):
    """
    Docstring for perform_local_step_2

    :param site_name: Description
    :param site_results: Description
    :param output_path: str
    """
    ind_site_data: np.ndarray = EXECUTOR_CACHE[site_name][
        'data']  # TODO: Don't store it as cache, better to load again.
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

        scores = np.tan((A - B) * np.pi / 2)

        site_scores[site] = scores

    num_cols = site_scores.select_dtypes(
        include=[np.number]).columns
    num_cols = num_cols.drop(site_name) if len(site_results.keys()) > 1 else num_cols

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

    site_centroids: dict[str, CentroidSet] = {}

    for site in site_results:
        site_centroids[site] = site_results[site]['centroids']

    """Combine FNC averages across sites into the final report payload."""
    sorted_names = sorted(site_results.keys())
    fnc_sums = {name: site_results[name]['fnc_sum'] for name in sorted_names}
    fnc_counts = {name: site_results[name]['fnc_count'] for name in sorted_names}
    global_avg_fnc = global_mean_from_sites(fnc_sums, fnc_counts)


    global_result_path = os.path.join(output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)

    for label in global_avg_fnc.keys():
        render_fnc_heatmap(global_avg_fnc[label], global_result_path,
                           name=f'global_original_avg_fnc_{group_names[int(label)]}.png',
                           title= f'Average FNC of Original {group_names[int(label)]} Subjects',
                           domain_names = [0, 5, 7, 16, 25, 42, 49, 53]
                           )

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
        i_neg_1b = round_clamp(N_neg * (1.0 - TruncationParameter), 1, N_neg)  # 1..N_neg
        i_neg0 = i_neg_1b - 1  # 0-based in com
    if N_pos > 0:
        i_pos_1b = N_neg + round_clamp(N_pos * TruncationParameter, 1, N_pos)  # (N_neg+1)..(N_neg+N_pos)
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
        for label in (1, 2):
            global_avg_per_label[label].append(site_results[site][label])

    global_result_path = os.path.join(output_path, 'global_results')
    os.makedirs(global_result_path, exist_ok=True)

    """Combine relabeled FNC averages across sites into the final report payload."""
    sorted_names = sorted(site_results.keys())
    fnc_sums ={}
    fnc_counts={}
    for name in sorted_names:
        fnc_sums[name] = {}
        fnc_counts[name] = {}
        for label in  site_results[site].keys():
            fnc_sums[name][label] = site_results[name][label]['sum']
            fnc_counts[name][label]  = site_results[name][label]['count']

    global_avg_fnc = global_mean_from_sites(fnc_sums, fnc_counts)
    #global_avg_fnc = global_mean_from_sites(global_avg_per_label)
    for label in (1, 2):
        render_fnc_heatmap(global_avg_fnc[label], global_result_path,
                           name=f'global_original_avg_fnc_{group_names[label]}.png',
                           title= f'Average FNC of Relabeled {group_names[label]} Subjects',
                           domain_names = [0, 5, 7, 16, 25, 42, 49, 53]
                           )

    for labels in [(1, 2), (2, 1)]:
        label_1 = labels[0]
        label_2 = labels[1]
        render_fnc_heatmap(global_avg_fnc[label_1] - global_avg_fnc[label_2], global_result_path,
                           name=f'federated_relabeled_group_diff_{group_names[label_1]}_{group_names[label_2]}.png',
                           title=f'Relabeled Group Difference {group_names[label_1]} vs {group_names[label_2]} (Federated)',
                           domain_names = [0, 5, 7, 16, 25, 42, 49, 53]
                           )

    print(f"Global results path: {global_result_path}")
    np.save("federated_output_fnc", global_avg_fnc)

    return global_avg_fnc


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
    original_dataset = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
        ],
    )
    X = original_dataset[SourceDataKeys.SFNC.value]
    t_values, p_values = compute_two_sample_ttest(X, re_labels)
    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)

    render_fnc_heatmap(corrected_t_values, output_path,
                       name='re_labeled_ttest.png',
                       title='Relabeled HC Vs SZ T-test values',
                       domain_names=[0, 5, 7, 16, 25, 42, 49, 53]
                       )

    """ Avg FNC Matrix of Relabeled Subjects """
    unique_labels = np.unique(re_labels).astype(int)
    avg_fnc, relabeled_aggregated_fnc_result = average_fnc_by_label(X, re_labels, unique_labels)

    for label in unique_labels:
        if label == -1:
            continue

        render_fnc_heatmap(avg_fnc[label], output_path,
                           name=f'local_relabeled_avg_fnc_{group_names[label]}.png',
                           title=f'Average FNC of Relabeled {group_names[label]} Subjects',
                           domain_names=[0, 5, 7, 16, 25, 42, 49, 53]
                           )

    return relabeled_aggregated_fnc_result


def run_federated_de_noise(round_id: int):
    round_ss = SeedSequence(20250908)
    global_seed = int(round_ss.generate_state(1)[0])

    random.seed(global_seed)
    np.random.seed(global_seed)

    rng_round = Generator(PCG64(round_ss))

    # Iteration - 1
    site_results = {}
    for site in sites:
        base_path = BASE_INPUT_DATA.format(site=site)
        output_path = BASE_OUTPUT_DIR.format(number=round_id, site=site)
        os.makedirs(output_path, exist_ok=True)
        site_results[site] = perform_local_step_1(
            site, base_path, output_path, rng=rng_round)

    # agg_results = perform_remote_step_1(site_results, "test_output/round_5")
    agg_results = perform_remote_step_1(site_results, output_path)

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
    # perform_remote_step_3(site_avg_fnc_data, "test_output/round_5")
    fed_group_diff = perform_remote_step_3(site_avg_fnc_data, output_path)

    # #Centralized display
    # cen_site={}
    # cen_site['site1' ] = perform_local_step_3('site1', '../test_data_centralized/site1',
    #                                             'test_output_centralized/round_5/site1', adaptive_score);
    # cen_group_diff = perform_remote_step_3(cen_site, "test_output_centralized/round_5/site1")
    #
    # import compare_centralized_federated_results as com
    #
    # com.plot_scatterplot_of_fed_ad_centralized_group_diff(cen_group_diff, fed_group_diff)


CURRENT_EXECUTION_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "5"))

if CURRENT_EXECUTION_ID == 0:
    raise Exception("Failed to get execution id")

run_federated_de_noise(CURRENT_EXECUTION_ID)