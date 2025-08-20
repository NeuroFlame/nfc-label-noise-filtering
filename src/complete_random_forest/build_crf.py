from collections import Counter
import numpy as np

from .node_definition import Node, NodeValues
from .crf_helpers import node_label, otsu4_thres


def build_crf(
    data: np.ndarray, is_continous_data: np.ndarray, up_labels: list, flag: int
):
    if data.size == 0:
        raise ValueError("The data is empty")
    
    subjects, features = data.shape

    if subjects < 2:
        value = NodeValues()
        value.datas = np.array(data[0, -1]).astype(int)  # id of the last subject
        value.labels = [int(data[0, 0]), *up_labels]
        new_node = Node(value)
        return new_node

    if len(np.unique(data[:, 0])) == 1:
        value = NodeValues()
        value.datas = np.array(data[:, -1]).astype(int)
        value.labels = [int(data[0, 0]), *up_labels]
        new_node = Node(value)
        return new_node

    # leaving Id's and label columns
    actual_features_length = features - 2
    random_features = np.random.permutation(actual_features_length)
    flagtemp = 0
    best_attribute = 0
    for rf in random_features:
        if np.unique(data[:, rf + 1]).size > 1:
            flagtemp = 1  # found a partition
            best_attribute = rf
            break

    if flagtemp == 0:
        # feature values are the same but the labels are different
        if len(data[:, 0]) > 5:
            print(data[0:3, 0:5])
        else:
            print(data[:, 0:5])

        unique_labels = np.unique(data[:, 0])

        if len(unique_labels) == 2:
            value = NodeValues()
            value.datas = data[:, features - 1].astype(int)
            value.labels = up_labels
            new_node = Node(value)

            left_value = NodeValues()
            left_value.datas = data[data[:, 0] == unique_labels[0], -1].astype(int)
            left_value.labels = [int(unique_labels[0]), *up_labels]

            right_value = NodeValues()
            right_value.datas = data[data[:, 0] == unique_labels[1], -1].astype(int)
            right_value.labels = [int(unique_labels[1]), *up_labels]

            left_node = Node(left_value)
            right_node = Node(right_value)

            new_node.left_node = left_node
            new_node.right_node = right_node
        else:
            value = NodeValues()
            value.datas = data[:, -1]
            majority_class_label = Counter(data[:, 0]).most_common(1)[0][0]
            value.labels = [int(majority_class_label), *up_labels]
            new_node = Node(value)

        return new_node

    # non-leaf nodes divide data in the following manner
    col = data[:, best_attribute + 1]
    if is_continous_data[best_attribute]:
        _, best_value = otsu4_thres(col)

        left_data = data[col <= best_value, :]
        right_data = data[col > best_value, :]
    else:
        unique_subjects_classes = np.unique(col)
        perm = np.random.permutation(len(unique_subjects_classes))
        best_value = unique_subjects_classes[perm[0]]

        left_data  = data[col == best_value, :]
        right_data = data[col != best_value, :]

    value = NodeValues()
    value.datas = data[:, -1]
    max_freq_labels = node_label(data, flag, up_labels)
    value.labels = [int(max_freq_labels), *up_labels]
    new_node = Node(value)
    new_node.left_node = build_crf(left_data, is_continous_data, value.labels, flag)
    new_node.right_node = build_crf(right_data, is_continous_data, value.labels, flag)

    return new_node
