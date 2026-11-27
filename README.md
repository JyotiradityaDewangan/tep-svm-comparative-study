# A Comparative Study of Kernel-Based Support Vector Machines and Convex Optimization for Fault Detection in the Tennessee Eastman Process

This beginner-friendly terminal project keeps a Scikit-learn RBF SVC as the primary TEP fault detector and adds separate linear SVM experiments: a from-scratch hinge-loss optimizer, a CVXOPT dual quadratic program, and Scikit-learn `LinearSVC`. These experimental models never replace the primary model. Each new run writes to a unique `results/runs/<run_id>/` folder; previously saved results and the legacy `models/` folder are preserved.

## Dataset inspection and what to download

Training inputs belong in `data/raw/`: `TEP_FaultFree_Training.csv` and `TEP_Faulty_Training.csv`. Put the separate official testing files in `data/testing/`: `TEP_FaultFree_Testing.csv` and `TEP_Faulty_Testing.csv`. Training runs sample only files under `data/raw/`; official testing files are never included in training. Each TEP CSV has 52 process variables: 41 measured variables (`xmeas_1`–`xmeas_41`) and 11 manipulated variables (`xmv_1`–`xmv_11`), plus target/run/sample metadata. This implementation reads CSV; the Harvard DOI release also distributes RData files.

Sources: [Harvard Dataverse DOI](https://doi.org/10.7910/DVN/6C3JR1) · [CSV release documentation](https://github.com/Mowriyan-2/TEP-fault-detection)

The multi-gigabyte TEP CSVs are not included in this repository. Download the four files listed above from the cited source and place them in `data/raw/` and `data/testing/` as documented.

Keep only training CSVs in `data/raw/`. The loader scans every CSV under that folder; placing official testing CSVs there would mix them into the training sample. Files are read in chunks and uniformly reservoir sampled. The default 20,000-row sample and 50,000-row safety cap keep kernel SVC bounded. Missing values and duplicates are checked on the sampled rows; the loader scans all source rows to obtain row counts. The main model is evaluated with an internal grouped split, not the official testing files.

### Official testing evaluation

After a successful training run has saved an RBF model, evaluate both official files without retraining or refitting preprocessing:

```powershell
python main.py --official-test
```

The command processes every row in chunks using the saved model, median imputer, scaler, and exact saved 52-feature order. It maps `faultNumber=0` to NORMAL and other numeric fault labels to FAULTY; recognized filenames provide labels if there is no target column. Results are separate from the internal holdout under a new `results/official_testing/evaluation_<timestamp>/` folder, including metrics, confusion matrix CSV/plot, source-file hashes, and a manifest. Select a particular saved training run with `python main.py --official-test --run-id <run_id>`.

Labels are checked from `faultNumber` when present: configured `NORMAL_LABEL=0` maps to NORMAL; observed other numeric fault IDs map to FAULTY after verifying at least two distinct source labels. Fault metadata, simulation run, sample index, timestamps, and other recognized metadata never become features. If no label exists, only filenames clearly indicating normal/fault-free and faulty data are accepted. No data is silently replaced.

## What the process represents

TEP is a simulated chemical manufacturing plant benchmark, not live measurements from a real plant. Gaseous reactants are processed through connected units and chemical products are separated. The five commonly described units are:

- **Reactor:** chemical reactions occur under controlled temperature and pressure.
- **Condenser:** cools process vapor so material can condense.
- **Separator:** divides product liquid from remaining gas.
- **Compressor:** raises the pressure of process gas for reuse or transfer.
- **Stripper:** removes remaining light components from the liquid product.

The 41 `xmeas` values represent measured process variables; the 11 `xmv` values represent manipulated control variables. Their engineering descriptions and units depend on the data dictionary supplied with the dataset.

## SVM and convex optimization

An SVM chooses a decision boundary that leaves the widest possible margin between classes. The observations nearest the boundary are support vectors and determine its position. A kernel computes similarity in a richer feature space, allowing curved boundaries without explicitly creating all transformed features. The project experiments use linear and RBF kernels; the primary model remains RBF.

For soft-margin classification, with labels `y_i ∈ {-1,+1}`, feature vector `x_i`, weight vector `w`, intercept `b`, and slack `xi_i`, the primal problem is:

```text
minimize    1/2 ||w||² + C Σ xi_i
subject to  y_i (wᵀ x_i + b) ≥ 1 - xi_i
            xi_i ≥ 0
```

The hard-margin problem minimizes `1/2 ||w||²` subject to `y_i(wᵀx_i+b)>=1`; it requires perfect separation and is unsuitable for noisy, overlapping classes. Slack variables make the soft-margin version more practical. The norm-squared term is convex and the constraints are linear, so it is a convex quadratic program. `C` controls the penalty on violations. Lagrange multipliers create the dual problem; the KKT conditions (primal/dual feasibility, stationarity, and complementary slackness) characterize its optimum. Scikit-learn's SVC uses its own optimizer; this project does not claim SVC calls CVXOPT.

`src/optimization.py` solves the dual QP. For `n` rows, `P=(yyᵀ)⊙(XXᵀ)`, `q=-1`, `G=[-I;I]`, `h=[0;C]`, `A=yᵀ`, and `b=0`; minimizing `1/2 αᵀPα+qᵀα` converts the dual maximization into CVXOPT form. The comparison uses a configurable small subset (default 1,000 total rows) because dense matrices grow quadratically in storage. A separate four-point QP is a short teaching example. CVXOPT is optional; its failure is reported while the other experiments continue.

## Project layout

```text
Chemical_Process_Fault_Detection/
├── config.py                 # All user-facing settings
├── main.py                   # Train, evaluate, report, or predict
├── requirements.txt
├── PROJECT_REPORT.md         # Research review, mathematics, and measured experiments
├── data/raw/                  # Place training CSV files here
├── data/testing/               # Place official testing CSV files here
├── data/processed/            # Dataset inspection summary
├── src/                       # Loading, preparation, SVM, reports, plots, prediction, QP
├── models/                    # Saved model and fitted transformers
├── results/
│   ├── main_model/  ├── svm_from_scratch/  ├── cvxopt_svm/
│   ├── svm_comparison/  ├── c_parameter_experiment/
│   ├── kernel_experiment/  ├── convergence/  ├── pca/
│   ├── runs/<run_id>/  # New run outputs, manifest, report, and saved models
│   ├── plots/  ├── heatmaps/  ├── excel_reports/  ├── csv_reports/
│   ├── predictions/  └── summary/
└── logs/
```

## Install and run (Windows PowerShell)

From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

CVXOPT is optional and is deliberately not included in `requirements.txt`. Install it separately with `python -m pip install cvxopt` to run the dual-QP experiment. Without it, the CVXOPT experiment is clearly skipped while the other SVM workflows continue. For package compatibility, Python 3.10–3.12 is a practical choice for this project.

Before real training, inspect `data/processed/dataset_summary.json` after a successful load. At startup the program prints actual sampled dimensions, columns, first rows, label counts, missing values, and duplicates. It saves dataset summary, actual evaluation reports, high-resolution plots, and the fitted model artifacts into their named folders.

## Settings in config.py

- `DATASET_PATH`: folder containing CSV inputs.
- `SAMPLE_SIZE`: maximum total loaded sample. `None` means full read and is guarded against unsafe SVC sizes.
- `NORMAL_LABEL`: source label representing normal operation (TEP faultNumber convention: 0).
- `TEST_SIZE`: test fraction, such as `0.30` for a 70:30 split.
- `RANDOM_STATE`: repeatable sample/split seed.
- `SVM_KERNEL`: `linear`, `rbf`, or `poly`.
- `SVM_C`: margin violation penalty; must be positive.
- `SVM_GAMMA`: `scale`, `auto`, or a positive number; controls nonlinear kernel reach.
- `USE_ALL_FEATURES` / `SELECTED_FEATURES`: include every numeric sensor or a specified subset.
- `NORMALIZE_DATA`: standardize values. The scaler is fit only on training rows.
- `USE_CLASS_BALANCING`: make SVC adjust for unequal training class counts.
- `RUN_SVM_EXPERIMENTS`: enable or skip the additional SVM-only experiments.
- `RUN_PCA_EXPERIMENT`: enable or skip the PCA versus original-feature study.
- `PCA_EXPLAINED_VARIANCE_TARGET`: training variance to retain (default `0.95`).
- `EXPERIMENT_KERNELS` / `EXPERIMENT_C_VALUES`: kernels and positive C values to evaluate.
- `EXPERIMENT_MAX_ROWS`: cap the training rows per C/kernel fit.
- `EXPERIMENT_SAMPLE_SIZES`: actual row counts used for the fit-time curve.
- `SCRATCH_COMPARE_ROWS` / `SCRATCH_ITERATIONS` / `SCRATCH_LEARNING_RATE`: comparison size and subgradient settings.
- `QP_MAX_ROWS`: total-row cap (default 1,000) for the dense CVXOPT comparison.
- `SAVE_MODEL`, `SAVE_PLOTS`, `SAVE_EXCEL_REPORTS`, `SAVE_CSV_REPORTS`, `SAVE_PREDICTIONS`: enable/disable their outputs.
- `MAX_CORRELATION_FEATURES`: feature count on correlation plot; `MAX_SVC_ROWS`: safety cap for the kernel model.

Invalid settings produce a clear error before training. The split is stratified by class and, when run metadata permits, grouped by source file plus simulation run so neighboring samples from a run are not divided between train and test. If this cannot retain both classes on both sides, the program reports and uses a stratified random split. This fallback can produce an optimistic result for related time-series rows; the split strategy is printed and included in the run summary.

## Predict without training again

Prediction needs a model saved by a successful `python main.py` run and a sensor CSV you create or obtain. The current local training files are `TEP_FaultFree_Training.csv` and `TEP_Faulty_Training.csv`; keep official testing CSVs outside `data/raw/` because the loader scans every CSV in that folder. For a prediction, create a CSV with the exact sensor feature names (for this version: `xmeas_1`–`xmeas_41` and `xmv_1`–`xmv_11`) and rows of actual readings. Do not pass `faultNumber`, `simulationRun`, or `sample` as input features. Then run:

```powershell
python main.py --predict --input data/new_sensor_readings.csv
```

Optional output path: `python main.py --predict --input readings.csv --output results/predictions/my_predictions.csv`. Required feature names and order come from the saved training artifacts. Extra columns are ignored, missing required features and nonnumeric values raise a clear error. The program applies the saved median imputer and scaler and writes NORMAL/FAULTY plus the SVM decision score.

## Outputs and metrics

- `results/plots/class_distribution.png`: sampled NORMAL/FAULTY counts.
- `results/heatmaps/confusion_matrix.png`: true class against predicted class.
- `results/heatmaps/sensor_correlation.png`: correlation of a readable feature subset.
- `results/plots/roc_curve.png`: true-positive versus false-positive rate and AUC.
- `results/plots/precision_recall_curve.png`: precision/recall trade-off.
- `results/plots/sensor_distribution_*.png`: selected sensor distributions by class.
- `results/plots/training_time_vs_sample_size.png`: measured SVC fit time at configured sample sizes.
- `results/svm_comparison/linear_implementation_comparison.png`: linear SVM comparison using the same small split.
- CSV/Excel experiment tables and charts are saved in their named subfolders under `results/`.
- Excel workbooks: dataset summary, sensor statistics, binary metrics, and per-class precision/recall/F1/support.
- CSV workbooks: metrics, class distribution, actual versus predicted, and test prediction rows with correctness and decision score.
- `results/summary/project_summary.txt`: computed run settings, scores, and output locations.

Accuracy is the fraction of correct predictions. Precision asks how many predicted faults were truly faulty. Recall asks how many actual faults were detected. F1 is the harmonic mean of precision and recall. ROC-AUC measures ranking quality across decision thresholds. Accuracy alone can hide missed faults, especially when classes are imbalanced, so use faulty-class recall and the confusion matrix too.

### Earlier recorded TEP run (historical example)

The 20,000-row sample (10,000 NORMAL / 10,000 FAULTY) gave a grouped internal holdout of 16,034 training and 3,966 test rows. The primary RBF SVC scored accuracy 0.7774, fault precision 0.9983, fault recall 0.5736, F1 0.7286, and ROC-AUC 0.8323. This is an internal split of the training CSVs, not a score on the official testing files. The report at [PROJECT_REPORT.md](PROJECT_REPORT.md) contains the bounded C/kernel/implementation results and the convergence analysis, including solver stopping limitations.

## What each Python module does

| File | Purpose |
|---|---|
| `config.py` | Central settings for paths, data sampling, model, and experiments. |
| `main.py` | CLI workflow for training, experiments-only runs, and predictions. |
| `src/data_loader.py` | Chunked CSV loading, target/metadata detection, sampling, and dataset summary. |
| `src/preprocessing.py` | Group-aware split and training-only imputation/scaling. |
| `src/svm_model.py` | Primary RBF SVC construction and artifact save/load. |
| `src/linear_svm_scratch.py` | Explicit mini-batch hinge-loss subgradient implementation and per-update loss trace. |
| `src/optimization.py` | CVXOPT dual QP matrices, solver, model recovery, and teaching example. |
| `src/experiments.py` | C/kernel/sample-size experiments and same-split linear implementation comparison. |
| `src/pca_experiment.py` | Training-only PCA fit and same-split original-feature versus PCA-feature RBF SVC comparison. |
| `src/evaluation.py` | Main-model metrics and CSV/Excel reporting. |
| `src/visualization.py` | Main-model plots and heatmaps. |
| `src/prediction.py` | Saved-model inference on a sensor CSV. |

## Synthetic pipeline check

`python main.py --self-test` runs a small generated dataset through fitting, prediction, save/load, plots, and reports inside a temporary folder. The generated data are strictly a software check, are not TEP, and are not evidence of model performance on TEP. The temporary model and reports are removed when the self-test ends.

## SVM implementations and experiments

The normal training command runs bounded SVM-only experiments. The C study uses `LinearSVC(loss="hinge")` at C=0.1, 1, 10, and 100. The kernel study compares linear and RBF SVC on the same fit/validation rows. The implementation study compares scratch subgradient, CVXOPT dual QP, and Scikit-learn `LinearSVC` on the same balanced subset and split, with matching `class_weight="balanced"` penalties (per-row bounds for CVXOPT and weighted hinge terms for scratch). `LinearSVC` does not expose support-vector indices; the scratch model reports margin-active points separately and does not claim they are exact dual support vectors. The convergence study records the normalized primal objective after every scratch-SVM update; subgradient loss may not decrease at every step, and a smooth curve alone does not prove convergence.

Each new run keeps reports in named subfolders under `results/runs/<run_id>/`, including `run_manifest.json` and an `experiment_report.md` generated from that run's measured result objects. The manifest records dataset filenames and sampled class counts, ordered features, split sizes/method, preprocessing, model/experiment settings, software versions, structured metrics including the confusion matrix, and SHA-256 hashes of available training and official testing CSVs. `results/main_model/`, other existing top-level result folders, and files already in `models/` are preserved as historical artifacts; they are not overwritten by new training runs. The new run's primary and experimental models live in its `models/` folder. The static tables in this report describe previously recorded runs; fit times vary by machine and run.

Run one experiment without fitting the primary RBF SVC again (the source CSVs are reloaded and the deterministic split is recreated). Only the selected experiment is executed:

```powershell
python main.py --experiments-only --experiment implementations
python main.py --experiments-only --experiment c
python main.py --experiments-only --experiment kernel
python main.py --experiments-only --experiment convergence
python main.py --experiments-only --experiment pca
```

Use `--experiment sample-size` for the fit-time curve, or `--experiments-only --experiment all` to run all studies without refitting the main SVC. The normal `python main.py` command runs the main model and all enabled experiments. Experiments use an inner group-aware training/validation split and leave the main holdout untouched. Each experiment now fits its imputer and scaler only on its own fitting rows; a clear message identifies any fallback to row splitting. The sample-size study measures fit time only, not predictive performance at each sample count.

The full mathematical derivation, literature review, references, research limitations, and experiment interpretation are in [PROJECT_REPORT.md](PROJECT_REPORT.md). The report distinguishes this educational implementation from a novel research contribution. Experiments make an inner validation split from training data; the separate main internal holdout is left untouched. The internal holdout and official testing evaluation are separate results.

### PCA feature reduction experiment

The default `python main.py` workflow runs the PCA comparison after the SVM experiments. It reuses the main run-aware train/test observations and their training-fitted median imputer and standard scaler. Scikit-learn PCA is fitted only on the standardized training matrix, retaining the configured `PCA_EXPLAINED_VARIANCE_TARGET` (default 95%); the held-out matrix is only transformed. Both original and reduced inputs are evaluated with the same RBF SVC hyperparameters, regardless of the configurable primary kernel. Principal components are new weighted combinations of sensor columns, not individual sensor readings. PCA may discard fault-relevant variation, so the experiment reports the measured result without assuming it improves.

Run PCA alone without refitting the primary saved model: `python main.py --experiments-only --experiment pca`. Its results are saved under the new run's `pca/` folder: `explained_variance.png`, `cumulative_explained_variance.png`, `dimensionality_reduction.png`, `pca_svm_performance_comparison.png`, and `pca_svm_comparison.csv` / `.xlsx`.

An earlier recorded local TEP run reduced 52 sensor features to 27 components (95.68% training variance retained; 48.08% dimensionality reduction). On its 16,034-row training and 3,966-row grouped test split, the original features scored accuracy 0.7774, precision 0.9983, recall 0.5736, F1 0.7286, ROC-AUC 0.8323. PCA scored accuracy 0.7637, precision 0.9956, recall 0.5489, F1 0.7076, ROC-AUC 0.8221. Treat these as historical internal-split results, not official-test results; use the run-specific report for the latest measured output.

## Automated tests

The synthetic suite uses Python's built-in `unittest`; it adds no package dependency. It validates training-only preprocessing, group splits, scratch-SVM objective math and class weights, CVXOPT constraints/objectives/KKT/bias recovery when CVXOPT is installed, PCA fit scope and component selection, prediction save/load, synthetic execution of each experiment, and missing-dataset CLI behavior. It does not replace testing on official TEP testing CSVs.

```powershell
python main.py --self-test
python -m unittest discover -s tests -v
```

`--self-test` and the test suite use synthetic data. They are software/math checks, not evidence of TEP model accuracy. Each complete run also writes a run-specific manifest and measured report. Prediction uses the latest completed run's saved primary model by default; `python main.py --predict --run-id <run_id> --input readings.csv` selects a particular run. Existing `models/` artifacts are used only when no completed run is available.

## Limitations and future scope

The first version is binary and depends on simulated benchmark data. A random sample of rows is not a substitute for evaluating every operating regime; the data source, sampling and split strategy should be described during review. Kernel SVC training grows quickly with sample count. Future work may identify individual fault types, add temporal early detection, monitor real-time industrial sensor streams, estimate fault severity, improve feature selection, or tune SVM parameters. None of these extensions is implemented here.
