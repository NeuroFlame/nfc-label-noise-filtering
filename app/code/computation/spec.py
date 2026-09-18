"""Declare the federated label-noise-filtering workflow.

Three fixed local/remote round trips (local CRF, dimensional scoring,
relabeling), each followed by a server-side aggregation, plus a final
site-local report step. There is no convergence loop, so this is a
``stepped_workflow`` rather than an ``iterative_workflow``.
"""

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
)

from .inputs import load_inputs
from .local_math import compute_dimensional_scores, compute_local_crf, relabel_subjects
from .remote_math import (
    aggregate_centroids,
    aggregate_relabeled_metrics,
    compute_adaptive_threshold,
)
from .results import build_site_report

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=compute_local_crf, input_fn=load_inputs),
        remote_step(fn=aggregate_centroids),
        local_step(fn=compute_dimensional_scores),
        remote_step(fn=compute_adaptive_threshold),
        local_step(fn=relabel_subjects),
        remote_step(fn=aggregate_relabeled_metrics),
        site_output_step(fn=build_site_report),
    ),
)
