import numpy as np
from scipy.spatial.distance import pdist
from .node_definition import Node

def is_continuous(data: np.ndarray) -> np.ndarray:
    """
    Determine if each column in the dataset is continuous.
    A column is considered continuous if it has more than 1/3 unique values.
    
    Parameters:
    - data: 2D NumPy array of shape (m, n)
    
    Returns:
    - is_conti: 1D NumPy array of shape (n,), with 1 for continuous, 0 otherwise
    """
    m, n = data.shape
    threshold = m / 3
    is_conti = np.zeros(n, dtype=int)

    for i in range(n):
        unique_vals = np.unique(data[:, i])
        if len(unique_vals) > threshold:
            is_conti[i] = 1

    return is_conti

def otsu4_thres(array: np.ndarray):
    """
    Python port of MATLAB Otsu4Thres.
    array: 1D (or Nx1) NumPy array of feature values.

    Returns:
        fea_id (int): 0-based index of the chosen threshold position
        thres (float): threshold value = array[fea_id]
    """
    arr = np.asarray(array).ravel()
    n = arr.size
    fun = np.full(n, -np.inf, dtype=float)  # sentinel so empty-group splits aren't chosen

    for v in range(n):
        te = arr[v]
        g1 = arr[arr <= te]
        g2 = arr[arr > te]

        size_n1 = g1.size
        size_n2 = g2.size
        if size_n1 == 0 or size_n2 == 0:
            continue  # matches intent; MATLAB would produce NaN here

        mean_n1 = g1.mean()
        mean_n2 = g2.mean()
        agg_mean = size_n1 * mean_n1 + size_n2 * mean_n2  # NOTE: mirrors your MATLAB (not normalized)

        fun[v] = size_n1 * (mean_n1 - agg_mean) ** 2 + size_n2 * (mean_n2 - agg_mean) ** 2  # between-class variance score

    fea_id = int(np.argmax(fun))
    thres = arr[fea_id]
    return fea_id, thres

def parallel_sort(frequency: np.ndarray) -> np.ndarray:
    """
    ParallelSort(Frequency) -> output (indices)

    In the original usage, Frequency = SortNum(:,2) is already sorted
    in DESCENDING order (counts). The function returns the indices
    (0-based) of entries that tie for the TOP frequency.
    """
    freq = np.asarray(frequency).ravel()
    if freq.size == 0:
        return np.array([], dtype=int)
    max_val = np.max(freq)
    return np.flatnonzero(freq == max_val)

def _mean_pdist(data: np.ndarray) -> float:
    """
    Mean pairwise Euclidean distance without SciPy.
    Returns +inf if fewer than 2 rows (mirrors MATLAB pdist() mean edge-case handling).
    """
    data = np.asarray(data, dtype=float)
    m = data.shape[0]
    if m < 2:
        return np.inf
    # squared distances via (x^2 + y^2 - 2x·y)
    sq_norms = np.sum(data * data, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (data @ data.T)
    iu = np.triu_indices(m, k=1)
    dists = np.sqrt(np.maximum(sq_dists[iu], 0.0))
    return float(dists.mean())

def node_label(data: np.ndarray, flag: int, up_labels: list) -> int:
    """
    Input:
    - data: ndarray with labels in FIRST column (col 0) and subject IDs in LAST column.
    - flag: int (1, 2, or 3) used to choose among ties.
    - up_labels: list of ancestor labels (first element used for a special tie-break).

    Returns:
    - label (int/float): the chosen class label.
    """
    labels_col = data[:, 0]

    # --- tabulate + sort by frequency DESC (tie-break by label ASC, like sortrows(...,-2)) ---
    uniq_vals, counts = np.unique(labels_col, return_counts=True)  # uniq_vals ascending
    # Stable sort by -counts; because uniq_vals is ascending, ties remain in ascending label order.
    order = np.argsort(-counts, kind="mergesort")
    sorted_vals = uniq_vals[order]
    sorted_cnts = counts[order]

    # --- find the set of labels tied at the TOP frequency ---
    top_idx = parallel_sort(sorted_cnts)  # indices into sorted arrays where count == max
    rank_top = sorted_vals[top_idx]       # the label values tied for most frequent
    rank_top_num = rank_top.size

    if rank_top_num == 1:
        # Single most frequent label
        return rank_top[0]

    # MATLAB uses 1-based indexing for flag; Python is 0-based.
    # For direct "pick #flag among ties", use (flag-1).
    if rank_top_num == 2 and flag <= 2:
        return rank_top[flag - 1]

    if rank_top_num == 3:
        # pick by flag position 1..3
        # clamp just in case flag is out of range
        pos = max(0, min(2, flag - 1))
        return rank_top[pos]

    if rank_top_num == 2 and flag == 3:
        # Special rule: choose the class whose feature cluster is more compact
        lab1, lab2 = rank_top[0], rank_top[1]
        # features are columns 1..-2 (exclude first label col and last subject ID col)
        fstart, fend = 1, data.shape[1] - 1
        group1 = data[labels_col == lab1, fstart:fend]
        group2 = data[labels_col == lab2, fstart:fend]

        d1 = np.mean(pdist(group1))
        d2 = np.mean(pdist(group2))

        chosen = lab1 if d1 <= d2 else lab2

        # if group1 has exactly 1 row and upLabels(1) is among the tied labels,
        # override with the ancestor's first label.
        if group1.shape[0] == 1 and len(up_labels) > 0 and up_labels[0] in rank_top:
            chosen = up_labels[0]
        return chosen

    # If there are >3 tied labels (rare), fall back to the first one (most frequent,
    # then by ascending label). You can adjust this to your needs.
    return rank_top[0]

def check_label_sequence(labels) -> int:
    n = len(labels)
    if n < 2:
        return 0

    ch1 = None
    for i in range(n - 1):
        if labels[i] != labels[i + 1]:
            ch1 = i
            break
    if ch1 is None:
        return 0

    for j in range(ch1 + 1, n - 1):
        if labels[j] != labels[j + 1]:
            return j - ch1
    return (n - 1) - ch1

def check_tree_leaf(sr_tree: Node) -> np.ndarray:
    """
    Returns: NumPy array of shape (k, 2)
            column 0 = datas (subject IDs)
            column 1 = subresult from check_label_sequence(labels)
    """
    # Treat as internal if either child exists (MATLAB used field count > 2)
    left = sr_tree.left_node
    right = sr_tree.right_node
    is_internal = (left is not None) or (right is not None)

    if is_internal:
        parts = []
        if left is not None:
            parts.append(check_tree_leaf(left))
        if right is not None:
            parts.append(check_tree_leaf(right))
        # Stack child results vertically
        return parts[0] if len(parts) == 1 else np.vstack(parts)

    # Leaf node: build [datas, subresult] where subresult repeats for each data item
    datas = np.atleast_1d(np.asarray(sr_tree.value.datas, dtype=int))
    labels = np.atleast_1d(np.asarray(sr_tree.value.labels, dtype=int))
    subresult = check_label_sequence(labels)
    return np.column_stack((datas, np.full(datas.shape[0], subresult, dtype=int)))
