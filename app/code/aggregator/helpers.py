import os
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from utils.types import HeatMapOptions


def fnc_heatmap(fnc_matrix: np.ndarray, options: HeatMapOptions):
    # Symmetric color range

    colorbar_name = options.get('colorbar_name', "")
    title = options.get('title', "Heat Map")
    name = options.get('name', "heatmap.png")
    domain_breaks: list = options.get('domain_names', [])
    output_path = options.get('path', "")

    max_value = np.nanmax(np.abs(fnc_matrix))

    max_value = max_value if np.isfinite(max_value) and max_value > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    im = ax.imshow(fnc_matrix, cmap="RdYlBu_r", vmin=-max_value, vmax=max_value, origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name, rotation=270, labelpad=10)

    # Optional domain gridlines (edit to your atlas boundaries)
    if domain_breaks:
        for g in domain_breaks:
            ax.axhline(g - 0.5, color='k', lw=0.6)
            ax.axvline(g - 0.5, color='k', lw=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    output_filename = os.path.join(output_path, name)

    plt.savefig(output_filename)
    # plt.show()

def global_mean_from_sites(site_packets: Dict, labels_map: Dict[str, str]):
    """
    site_packets: list of dicts returned by site_stats_z for the SAME group.
    Supports either upper-triangle or full-matrix mode (must match across sites).
    Returns global average correlation matrix (p,p) for that group.
    """
    result = {}
    for label in labels_map.keys():
        aggregated_sum = np.zeros((53, 53), dtype=float)
        aggregated_total = np.zeros((53, 53), dtype=float)

        for pkt in site_packets[label]:
            aggregated_sum += pkt["sum"]
            aggregated_total += pkt["count"]

        avg_fnc = np.divide(aggregated_sum, aggregated_total, out=np.zeros_like(aggregated_sum),
                            where=aggregated_total > 0)

        inverse_avg_fnc = np.tanh(avg_fnc)
        inverse_avg_fnc = (inverse_avg_fnc + inverse_avg_fnc.T) / 2.0
        np.fill_diagonal(inverse_avg_fnc, 1.0)

        result[label] = inverse_avg_fnc

    return result
