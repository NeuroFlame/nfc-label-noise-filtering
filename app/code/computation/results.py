"""Build the final per-site HTML report (the site_output_step)."""

from . import constants
from .reports import generate_report_html, render_fnc_heatmap
from .types import GlobalReportData


def build_site_report(
    global_report: GlobalReportData, *, state, parameters, output_dir
):
    """Render the global heatmaps and assemble this site's self-contained report."""
    domain_names = parameters.get("FNCDomainNames", constants.DEFAULT_FNCDomainNames)
    label_def = parameters.get("LabelDefinition", {})

    for label, matrix in global_report.global_avg_fnc_original.items():
        label_name = label_def.get(label, {}).get("name", label)
        render_fnc_heatmap(
            matrix,
            output_dir,
            f"global_original_avg_fnc_{label_name}.png",
            title=f"Global Average FNC of Original {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    for label, matrix in global_report.global_avg_fnc_relabeled.items():
        label_name = label_def.get(label, {}).get("name", label)
        render_fnc_heatmap(
            matrix,
            output_dir,
            f"global_relabeled_avg_fnc_{label_name}.png",
            title=f"Global Average FNC of Relabeled {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    html = generate_report_html(
        state["my_name"],
        output_dir,
        state.get("scores_df"),
        parameters,
        global_report.adaptive_threshold,
    )
    return {"index.html": html}
