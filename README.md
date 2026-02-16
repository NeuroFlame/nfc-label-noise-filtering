### Overview

This paper presents Fed LAMP, a label noise filtering based dimensional prediction method called LAMP to improve biomarker discovery and prediction for mental disorders using fMRI data. The method uses a complete random forest model to identify typical subjects whose clinical labels are consistent with brain functional connectivity patterns. These reliable subjects are then used to build a dimensional model that assigns continuous scores reflecting disease severity to unseen subjects. Experiments on multi site schizophrenia and autism datasets show improved group separability, more stable biomarkers, and better generalization than traditional label based approaches. 

### Example Settings

```json
{
    "SamplingThreshold": "float",
    "Iteration": "int",
    "NTree": "int",
    "LabelThreshold": "float",
    "TypicalThreshold": "float",
    "TruncationParameter": "float",
    "LabelDefinition": {
        "1": {
          "name": "str",
          "label": "any"
        },
        "2": {
          "name": "str",
          "label": "any"
        }
    },
    "LogLevel": "str"
}
```

### Settings Specification

| Variable Name | Type      | Description                                                                                                                       | Allowed Options | Default | Required |
| --- |-----------|-----------------------------------------------------------------------------------------------------------------------------------| --- |---------| --- |
| `SamplingThreshold` | `float`   | To decide how often a sample must look non noisy across repeated random sampling.                                                 | float | 0.7     | ✅ true |
| `Iteration` | `int`     | Controls how many times the full CRF process is repeated.                                                                         | integer | 101     | ✅ true |
| `NTree` | `int`     | Number of decision trees in each complete random forest.                                                                          | integer | 201     | ✅ true |
| `LabelThreshold` | `boolean` | Number of change sequence happens in leaf nodes of CRF to detect tree label                                                       | integer | 2       | ✅ true |
| `TypicalThreshold` | `boolean` | To decide if a subject is typical or noisy. If score ≥ this threshold, subject is kept as typical subject                         | float | 0.8     | ✅ true |
| `TruncationParameter` | `float`   | It trims the extreme x% from both sides of the negative and positive score distributions before computing the boundary threshold. | float | 0.2     | ✅ true |

### Input Description

Each site required to provide a path to `.mat` file as input for this computation and must be in the below format:

    - FILE_ID = 'FILE_ID':
    - ANALYSIS_ID = 'analysis_ID'
    - ANALYSIS_SCORE = 'analysis_SCORE'
    - SFNC = 'sFNC'

Please check the sample files in the `nvflare_code/test_data/site1/data.mat` which FBIRN data format.

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

### Output Description

*   **Output files:**  site1.csv, `Original group Global Avg: HC.png`, `Original group Global Avg: SZ.png`, `Relabeled group Global Avg: HC.png`, `Relabeled group Global Avg: SZ.png`

    * Two-sample t-tests on FNC (Original vs. Relabeled)

### Developer Setup:

1. Clone the Repo
2. Build and run Docker file with: `./dockerRun.sh`
3. Inside the docker terminal opened from above command, run the app with: `nvflare simulator -c site1,site2 ./app`