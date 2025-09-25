import os
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import cdist

from crf import crf
from utils import utils
from utils import find_scores


from numpy.random import SeedSequence, PCG64, Generator


# PARAMETERS
SamplingThs = 0.7
iter = 101
ntree = 201
NI_threshold = 2
TypThs = 0.8

parameters = {
    'sampling_threshold': SamplingThs,
    'iter': iter,
    'ntree': ntree,
    'label_threshold': NI_threshold,
    'typical_threshold': TypThs
}


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
    Docstring for perform_local_step_1

    :param site_name: Description
    :type site_name: str
    :param data_path: Description
    :type data_path: str
    """
    data = utils.convert_fnc_to_features(data_path, output_path, site_name)

    EXECUTOR_CACHE[site_name] = {
        'data': data
    }

    # data_path = os.path.join(output_path, f'{site_name}.mat')
    # data = data_loaders.load_result_matfile(data_path)[site_name]

    subject_noise_counts = crf.perform_crf(
        data, output_path, site_name, parameters, rng=rng)

    # crf_file = os.path.join(output_path, f'{site_name}_CRF.mat')
    # subject_noise_counts = data_loaders.load_result_matfile(crf_file)['count'][:]

    centroids = find_scores.get_centroids(
        data, subject_noise_counts, parameters["typical_threshold"])

    selected_features_file = os.path.join(output_path, 'centers.npz')
    np.savez(selected_features_file, **centroids)
    return centroids


def perform_local_step_2(site_name, site_results: dict[str, any], output_path: str):
    """
    Docstring for perform_local_step_2

    :param site_name: Description
    :param site_results: Description
    :type site_results: dict[str, any]
    """
    ind_site_data: np.ndarray = EXECUTOR_CACHE[site_name]['data']
    site_scores = pd.DataFrame(columns=site_results.keys())

    for site in site_results:
        if site == site_name:
            site_scores[site] = ind_site_data[:, -1]
            continue

        main_sz = site_results[site]['center_sz']
        main_hc = site_results[site]['center_hc']
        main_selected_features = site_results[site]['selected_features']

        ind_selected_data = ind_site_data[:, main_selected_features]
        dist1 = cdist(ind_selected_data, main_sz)
        dist2 = cdist(ind_selected_data, main_hc)

        distance_typical_group_sz = dist1.mean(axis=1)
        distance_typical_group_hc = dist2.mean(axis=1)

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

# def perform_remote_step_1(site_results: dict[str, any]):
#     pass


def run_federated_deNoise(round_id: int):

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

    # agg_results = perform_remote_step_1(site_results)

    for site in sites:
        output_path = BASE_OUTPUT_DIR.format(number=round_id, site=site)
        perform_local_step_2(site, site_results, output_path)


CURRENT_EXECUTION_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

if CURRENT_EXECUTION_ID == 0:
    raise Exception("Failed to get execution id")

run_federated_deNoise(CURRENT_EXECUTION_ID)
