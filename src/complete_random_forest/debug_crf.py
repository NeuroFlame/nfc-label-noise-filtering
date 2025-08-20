from typing import Any, Optional

def _get(obj: Any, name: str, default: Optional[Any] = None):
    """Dict-or-attr getter."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _get_value(obj: Any):
    """Return the node's value payload (dict or object) under 'values' or 'value'."""
    v = _get(obj, "values", None)
    if v is None:
        v = _get(obj, "value", None)
    return v

def _get_child(node: Any, left: bool):
    """Return left or right child supporting multiple field names."""
    if left:
        return (_get(node, "left_node")
                or _get(node, "leftLeaf")
                or _get(node, "left"))
    else:
        return (_get(node, "right_node")
                or _get(node, "rightLeaf")
                or _get(node, "right"))

def log_tree(root: Any, indent: int = 0, path: str = "•", _seen=None):
    """
    Recursively log the entire tree.
    Expects each node to have:
      - values/value -> { 'datas': List[int], 'labels': List[int] }
      - left_node/leftLeaf/left
      - right_node/rightLeaf/right
    """
    if root is None:
        return
    if _seen is None:
        _seen = set()
    # prevent accidental cycles
    obj_id = id(root)
    if obj_id in _seen:
        print("  " * indent + f"{path} (cycle detected, skipping)")
        return
    _seen.add(obj_id)

    val = _get_value(root) or {}
    datas = _get(val, "datas")
    labels = _get(val, "labels")

    print("  " * indent + f"{path} Node")
    print("  " * indent + f"  labels: {labels}")
    print("  " * indent + f"  datas : {datas}")

    left = _get_child(root, left=True)
    right = _get_child(root, left=False)

    if left is not None:
        log_tree(left, indent + 1, path + " → L", _seen)
    if right is not None:
        log_tree(right, indent + 1, path + " → R", _seen)
