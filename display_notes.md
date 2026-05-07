### Overview

This paper presents Fed LAMP, a label noise filtering based dimensional prediction method called LAMP to improve biomarker discovery and prediction for mental disorders using fMRI data. The method uses a complete random forest model to identify typical subjects whose clinical labels are consistent with brain functional connectivity patterns. These reliable subjects are then used to build a dimensional model that assigns continuous scores reflecting disease severity to unseen subjects. Experiments on multi site schizophrenia and autism datasets show improved group separability, more stable biomarkers, and better generalization than traditional label based approaches. 

### Example Settings

```json
{
    "SamplingThreshold": 0.7,
    "Iteration": 101,
    "NTree": 201,
    "LabelThreshold": 2,
    "TypicalThreshold": 0.8,
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
    "LogLevel": "info"
}
```

### Settings Specification

| Variable Name | Type      | Description                                                                                                                       | Allowed Options | Default | Required |
| --- |-----------|-----------------------------------------------------------------------------------------------------------------------------------| --- |---------| --- |
| `SamplingThreshold` | `float`   | To decide how often a sample must look non noisy across repeated random sampling.                                                 | float | 0.7     | ✅ true |
| `Iteration` | `int`     | Controls how many times the full CRF process is repeated.                                                                         | integer | 101     | ✅ true |
| `NTree` | `int`     | Number of decision trees in each complete random forest.                                                                          | integer | 201     | ✅ true |
| `LabelThreshold` | `int`     | Number of change sequence happens in leaf nodes of CRF to detect tree label                                                       | integer | 2       | ✅ true |
| `TypicalThreshold` | `float`   | To decide if a subject is typical or noisy. If score ≥ this threshold, subject is kept as typical subject                         | float | 0.8     | ✅ true |
| `TruncationParameter` | `float`   | It trims the extreme x% from both sides of the negative and positive score distributions before computing the boundary threshold. | float | 0.2     | ✅ true |
| `LabelDefinition` | `object`  | Maps integer label keys to `{name, label}` objects defining each class (e.g. `"1": {"name": "SZ", "label": 1}`)                  | — | —       | ✅ true |

### Input Description

Each site is required to provide two CSV files:

- `data.csv` — rows are subjects, columns are FNC features followed by a label column
- `labels.csv` — subject labels (one per row, matching the order in `data.csv`)

Sample files are available in `test_data/site1/` and `test_data/site2/`.

### Algorithm Description

The key steps of the algorithm include:

1.  **Local Identification of Typical Subjects**:
    
    * Each site begins by filtering the subjects using complete random forest models. This step measures the stability of the original label across the labels predicted by these models. 
        
    * This produces a reduced feature set that better reflects the group differences. Using only the typical subjects with their selected features, each site computes an average feature vector for both the healthy and the patient group. These local summaries that represent the site-specific information are then shared with the aggregator (remote) site
        
2.  **Forming Global Models**:
    
    * The remote aggregator receives two group averages sent by each of the participating sites. These are aggregated across sites to form a map of centriods for healthy and patient groups.

3. **Computing Dimensional Scores at Each Site**:

    * After receiving the global models, each site computes a dimensional score for every subject. This score quantifies how similar a subject is to the patient versus a healthy reference pattern.

    * This score places the subject along a continuous healthy–patient axis and reduces thousands of imaging features to a single interpretable value.

4. **Federated Adaptive Thresholding**:

    * Dimensional score ranges vary across sites and iterations. In the federated method, each site sends only its dimensional scores to a remote aggregator. The aggregator combines all scores from all sites and computes one global threshold. This global threshold is returned to all sites.

    * Each site then applies the same thresholding rule to determine whether a subject is strongly healthy, strongly patient, or falls in the boundary region.

5. **Producing Updated Labels**:

    * Once the new labels are assigned, each site updates its typical subjects and recomputes its group averages. 

    * Over iterations, typical subjects become cleaner, intra-group compactness improves, and inter-group differences become more stable.

6. **Federated Report Generation**:

    * After relabeling, the aggregator combines the per-site relabeled FNC averages into global group averages (one per label) and broadcasts them back to all sites along with the adaptive threshold value.

    * Each site uses this global data together with its local results to generate a self-contained `index.html` report containing: label distribution KPIs, Bonferroni-corrected T-test heatmaps (original vs. relabeled), local and global average FNC heatmaps, and a per-subject dimensional score table with dark-mode support.

### Output Description

Each site produces the following files:

| File | Description |
| --- | --- |
| `{site_name}_relabeled.csv` | Per-subject dimensional scores and re-assigned labels (−1 = uncertain) |
| `index.html` | Self-contained HTML report with KPIs, heatmaps, and scores table |
| `original_labels_ttest.png` | Bonferroni-corrected T-test heatmap using original labels |
| `re_labeled_ttest.png` | Bonferroni-corrected T-test heatmap using relabeled subjects |
| `local_original_avg_fnc_{label}.png` | Local average FNC per group before relabeling |
| `local_relabeled_avg_fnc_{label}.png` | Local average FNC per group after relabeling |
| `global_original_avg_fnc_{label}.png` | Federated average FNC per group before relabeling |
| `global_relabeled_avg_fnc_{label}.png` | Federated average FNC per group after relabeling |