import numpy as np
# from . import build_crf as crf
from . import iterative_crf as crf
from . import crf_helpers as helper

from joblib import Parallel, delayed

class CRF:
    def __init__(self, ntree: int, label_noise_threshold: int):
        self.rf_1 = []
        self.rf_2 = []
        self.label_noise_threshold = label_noise_threshold
        self.ntree = ntree

    def crf_v1(self, train_data: np.ndarray, ntree: int):
        subjects_count, _ = train_data.shape
        is_continuous_data = helper.is_continuous(train_data[:, 1:])
        train_data = np.hstack(
            (train_data, np.arange(1, subjects_count + 1).reshape(-1, 1))
        )

        # final_output_1= np.empty(ntree, dtype=np.ndarray)
        # final_output_2= np.empty(ntree, dtype=np.ndarray)
        
        final_output_1 = Parallel(
                n_jobs=8,
                prefer="processes",
                batch_size="auto"
        )(
            delayed(crf.build_crf_results_iter)(
                train_data, is_continuous_data, 1
            )
            for _ in range(ntree)
        )
        
        final_output_2 = Parallel(
                n_jobs=8,
                prefer="processes",
                batch_size="auto"
        )(
            delayed(crf.build_crf_results_iter)(
                train_data, is_continuous_data, 2
            )
            for _ in range(ntree)
        )
        
        # for i in range(ntree):
        #     final_output_1[i] = crf.build_crf_results_iter(train_data, is_continuous_data, 1)
        #     final_output_2[i] = crf.build_crf_results_iter(train_data, is_continuous_data, 2)

        final_output_1 = np.hstack(final_output_1)
        final_output_2 = np.hstack(final_output_2)

        final_decisions = np.hstack([final_output_1, final_output_2])

        final_decisions[final_decisions < self.label_noise_threshold] = 0 # non-noise
        final_decisions[final_decisions >= self.label_noise_threshold] = 1 # noise

        noise_subjects = (final_decisions.sum(axis=1) > 0.5 * ntree * 2).astype(
            int
        )

        non_noise_subject_id: np.ndarray = np.where(noise_subjects == 0)[0]

        # non_noise_data = train_data[non_noise_subject_id, :]

        return non_noise_subject_id, final_decisions

    def compute_nltc_sequence(self, train_data: np.ndarray):
        # removing id's from the train_data:
        train_data = train_data[:, :-1]
        subject_count, _ = train_data.shape

        nltc_label_seq1 = np.zeros((subject_count, self.ntree), dtype=int)
        nltc_label_seq2 = np.zeros((subject_count, self.ntree), dtype=int)

        for tree_id in range(self.ntree):
            tree_result_1 = helper.check_tree_leaf(self.rf_1[tree_id])
            tree_result_1 = tree_result_1[tree_result_1[:, 0].argsort()]
            nltc_label_seq1[:, tree_id] = tree_result_1[:, 1]

            tree_result_2 = helper.check_tree_leaf(self.rf_2[tree_id])
            tree_result_2 = tree_result_2[tree_result_2[:, 0].argsort()]
            nltc_label_seq2[:, tree_id] = tree_result_2[:, 1]

        nltc_labels = np.hstack((nltc_label_seq1, nltc_label_seq2))

        nltc_labels[nltc_labels < self.label_noise_threshold] = 0  # not noise
        nltc_labels[nltc_labels >= self.label_noise_threshold] = 1  # yes noise

        return nltc_labels
