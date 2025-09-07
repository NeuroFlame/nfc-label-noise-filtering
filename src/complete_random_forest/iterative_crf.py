import numpy as np
from collections import Counter

from .crf_helpers import node_label, otsu4_thres, check_label_sequence

# You already have these:
# from .crf_helpers import node_label, otsu4_thres, check_label_sequence

def build_crf_results_iter(
    data: np.ndarray,
    is_continuous: np.ndarray,
    flag: int,
) -> np.ndarray:

    if data.size == 0:
        raise ValueError("The data is empty")

    sub_count, features = data.shape
    if features < 2:
        raise ValueError("Expected at least 2 columns: label and id")

    # `data` layout assumed: col0=label, cols 1..F-2 = features, col F-1=id
    # We avoid copying sub-matrices by maintaining a permutation over rows.
    perm = np.arange(sub_count, dtype=np.int32)

    # Each stack item = (lo, hi, up_labels_tuple)
    # - work on rows perm[lo:hi]
    # - pass down the ancestry labels (root-most is last in the tuple, like MATLAB’s [child, parent, ...])
    stack = [(0, sub_count, tuple())]
    out_blocks = []

    # Local helpers for emitting leaves
    def _emit_leaf(lo: int, hi: int, up: tuple, force_label=None):
        rows = perm[lo:hi]
        ids = data[rows, -1].astype(int).ravel()
        # If force_label is given, use it; else take the node’s sole/first label
        if force_label is None:
            leaf_label = int(data[rows[0], 0])
        else:
            leaf_label = int(force_label)
        labels_seq = np.array((leaf_label, *up), dtype=int)  # [node_label, ancestors...]
        sub = check_label_sequence(labels_seq)
        out_blocks.append(np.column_stack((ids, np.full(ids.shape[0], sub, dtype=int))))

    while stack:
        lo, hi, up = stack.pop()
        span = hi - lo
        rows = perm[lo:hi]

        # --- Leaf tests (same as recursion) ---
        if span < 2:
            _emit_leaf(lo, hi, up)
            continue

        node_labels = data[rows, 0]
        uniq = np.unique(node_labels)
        if len(uniq) == 1:
            # Pure label leaf
            _emit_leaf(lo, hi, up, force_label=int(uniq[0]))
            continue

        # --- Pick a splittable feature (features are cols 1..F-2) ---
        best_attr = None
        for rf in np.random.permutation(features - 2):
            # need >1 distinct values to split
            if np.unique(data[rows, 1 + rf]).size > 1:
                best_attr = rf
                break

        if best_attr is None:
            # Degenerate: cannot split but labels differ — mirror MATLAB fallback
            # If exactly 2 classes: emit parent leaf with two child leaves’ labels inherited,
            # but in direct-output mode we just choose the majority and emit once.
            # This keeps NLTCLables identical to the recursion+check_tree_leaf path.
            uniq = np.unique(node_labels.astype(int))
            if len(uniq) == 2:
                # Match Code 2 behavior: treat this like a parent with two child leaves,
                # functionally meaning each subject keeps its *own* label at the head of the sequence.
                rows = perm[lo:hi]
                ids_block = data[rows, -1].astype(int)
                labels_block = data[rows, 0].astype(int)

                subs = np.fromiter(
                    (check_label_sequence(np.array((lbl, *up), dtype=int)) for lbl in labels_block),
                    dtype=int,
                    count=rows.size,
                )
                out_blocks.append(np.column_stack((ids_block, subs)))
            else:
                # >2 classes: Code 2 collapses to majority label → keep existing majority-leaf behavior
                maj = Counter(node_labels.astype(int)).most_common(1)[0][0]
                _emit_leaf(lo, hi, up, force_label=int(maj))
            continue

        # --- Node label for children’s ancestry ---
        # Pass the *current node*’s chosen label down to children (matches MATLAB NodeLabel usage)
        nod_lab = int(node_label(data[rows, :], flag, list(up)))
        new_up = (nod_lab, *up)

        # --- Partition by chosen feature ---
        feat = data[rows, 1 + best_attr]
        if is_continuous[best_attr]:
            # Otsu threshold for continuous feature
            _, thr = otsu4_thres(feat)
            mask = (feat <= thr)
        else:
            # Random category for discrete feature
            vals = np.unique(feat)
            pick = np.random.choice(vals)
            mask = (feat == pick)

        left_idx = rows[mask]
        right_idx = rows[~mask]

        # If split degenerate (one side empty), fall back to a leaf (avoid infinite loops)
        if left_idx.size == 0 or right_idx.size == 0:
            uniq = np.unique(node_labels.astype(int))
            rows = perm[lo:hi]
            if len(uniq) == 2:
                ids_block = data[rows, -1].astype(int)
                labels_block = data[rows, 0].astype(int)
                subs = np.fromiter(
                    (check_label_sequence(np.array((lbl, *up), dtype=int)) for lbl in labels_block),
                    dtype=int,
                    count=rows.size,
                )
                out_blocks.append(np.column_stack((ids_block, subs)))
            else:
                maj = Counter(node_labels.astype(int)).most_common(1)[0][0]
                _emit_leaf(lo, hi, up, force_label=int(maj))
            continue

        # In-place partition of the permutation segment
        perm[lo:hi] = np.concatenate((left_idx, right_idx))
        mid = lo + left_idx.size

        # Push children; order (right then left) if you want a DFS that processes left first on pop
        stack.append((mid, hi, new_up))   # right
        stack.append((lo,  mid, new_up))  # left

    # One concatenate at the end (fast)
    result = np.vstack(out_blocks)
    ids = result[:, 0].astype(int)
    vals = result[:, 1].astype(int)
    
    scores = np.empty(sub_count, dtype=int)
    scores[ids-1]=vals # becasue of id's are 1-based
    
    return scores.reshape(-1,1)