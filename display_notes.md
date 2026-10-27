**Source code:** [https://github.com/NeuroFlame/nfc-label-noise-filtering](https://github.com/NeuroFlame/nfc-label-noise-filtering)

&nbsp;

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
  "ControlLabel": "HC",
  "PatientLabel": "SZ",
  "LabelFile" : "labels.csv",
  "ExistingDataFile" : "data.csv",
  "ReadDataFileFromHalfpipeOutputs": {
    "atlas": "schaefer2011",
    "corr": "corrMatrix"
  },
  "FNCDomainNames" :  [0, 5, 7, 16, 25, 42, 49, 53],
  "IgnoreSubjectsWithMissingData" : true,
  "LogLevel": "error"
}
```

### Settings Specification

| Variable Name | Type      | Description                                                                                                                       | Allowed Options   | Default | Required                                                          |
| --- |-----------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------|---------|-------------------------------------------------------------------|
| `SamplingThreshold` | `float`   | To decide how often a sample must look non noisy across repeated random sampling.                                                 | float             | 0.7     | ✅ true                                                            |
| `Iteration` | `int`     | Controls how many times the full CRF process is repeated.                                                                         | integer           | 101     | ✅ true                                                            |
| `NTree` | `int`     | Number of decision trees in each complete random forest.                                                                          | integer           | 201     | ✅ true                                                            |
| `LabelThreshold` | `int`     | Number of change sequence happens in leaf nodes of CRF to detect tree label                                                       | integer           | 2       | ✅ true                                                            |
| `TypicalThreshold` | `float`   | To decide if a subject is typical or noisy. If score ≥ this threshold, subject is kept as typical subject                         | float             | 0.8     | ✅ true                                                            |
| `TruncationParameter` | `float`   | It trims the extreme x% from both sides of the negative and positive score distributions before computing the boundary threshold. | float             | 0.2     | ✅ true                                                            |
| `ControlLabel`        | `str or int` | Label used to indicate healthy controls in the dataset| String or integer | -       | ✅ true                                                            |
| `PatientLabel`        | `str or int` | Label used to indicate patients or unhealthy subjects| integer           | -       | ✅ true                                                            |
| `LabelFile`    | `str`        | Name of the label file in csv format. This file should contain headers "subject_id" and "diagnosis"| string            | -       | ✅ true                                                            |
| `ExistingDataFile`        | `str`        | Name of the csv file containing data of all the subjects (subjects * data). For FNC, the data is the linearized upper traingular data of the FNC matrix. Either this or the Halfpipe outputs should be provided for data. Should not contain any header. | string            | -       | ✅ true (if halfpipe outputs are not provided) ❌ false (otherwise) |
| `ReadDataFileFromHalfpipeOutputs`        | `dict`       | Reads the output files directly from halfpipe output folder based on the atlas and correlation type provided. Either this or  "ExistingDataFile" should be provided for data| dict              | -       | ✅ true (if data file is not provided) ❌ false (otherwise)         |
| `IgnoreSubjectsWithMissingData`    | bool        | Ignore or fail the run if any of the subjects with empty or missing values       | bool              | false   | ✅ true                                                            |
| `FNCDomainNames`    | `list`        | Domain names of FNC data used for displaying while rendering FNC data  | string            | []     | ✅ true                                                            |
| `LogLevel`    | `str`        | Logs the errors. Value either "info" or "error"       | string            | "info"   | ❌false                                                            |

### Input Description

Each site is required to data and labels files:

**Format -1 :** 

- `data.csv` —  filename can vary and can be configured in _parameters.json_. Each row has vectorized upper triangular FNC data for that subject. The subject ordering in this file should match exactly with labels.csv file.
- `labels.csv` —  filename can vary and can be configured in _parameters.json_. File with headers "_subject_id_" and "_diagnosis_" and contain label for each subject and its diagnosis as per '_ControlLabel_' or '_PatientLabel_' mentioned in parameters. The subject ordering in this file and the data csv file such match exactly. )

Sample files are available in `test_data/site1/` and `test_data/site2/`.


**Format - 2 :** 
- `Halfpipe output folder` - This folder should contain FNC files for each subject in tsv format. The subject id with which the file name starts with should be mentioned in _labels.csv_. The atlas and correlation matrix that are needed for filename selction should be provided _ReadDataFileFromHalfpipeOutputs_ in parameters.json.
- `labels.csv` —  filename can vary and can be configured in _parameters.json_. File with headers "_subject_id_" and "_diagnosis_" and contain label for each subject and its diagnosis as per '_ControlLabel_' or '_PatientLabel_' mentioned in parameters. Make sure the 'subject_id' contains the subject id with which the filename in halfpipe outputs starts with. )

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

| File | Description |
| --- | --- |
| `relabeled.csv` | Per-subject dimensional scores and re-assigned labels (−1 = uncertain) |
| `index.html` | Self-contained HTML report with KPIs, heatmaps, and scores table |
| `original_labels_ttest.png` | Bonferroni-corrected T-test heatmap using original labels |
| `re_labeled_ttest.png` | Bonferroni-corrected T-test heatmap using relabeled subjects |
| `local_original_avg_fnc_{label}.png` | Local average FNC per group before relabeling |
| `local_relabeled_avg_fnc_{label}.png` | Local average FNC per group after relabeling |
| `global_original_avg_fnc_{label}.png` | Federated average FNC per group before relabeling |
| `global_relabeled_avg_fnc_{label}.png` | Federated average FNC per group after relabeling |
| `orig.mat`, `crf.mat`, `centers.npz` | Diagnostic intermediates (validated input data, CRF results, typical-subject centroids) |

### Boilerplate Migration

This computation runs on the `computation-nvflare-boilerplate` `framework`/`runtime` architecture; authors work only inside `app/code/computation/`. See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for what changed from the prior hand-written NVFlare architecture and how the migration was verified.
