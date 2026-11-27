# Chemical Process Fault Detection Using SVM and Convex Optimization

## Abstract

This project studies binary fault detection on the simulated Tennessee Eastman Process (TEP). It maps process sensor measurements to NORMAL or FAULTY labels. Alongside scikit-learn's SVC, the project includes a teaching implementation of a linear soft-margin SVM using hinge-loss subgradient descent and a small CVXOPT dual quadratic program. Bounded experiments examine C, linear versus RBF kernels, sample-size fit time, faulty-class recall, and agreement/performance of the three linear implementations. The experiments are educational and do not propose a new SVM algorithm.

## Problem and motivation

Chemical process variables interact, and disturbances can appear in several measurements at once. A classification model can provide a data-driven indication that a sampled operating point resembles faulty rather than normal behavior. The Tennessee Eastman process is a controlled simulation benchmark created for evaluating process control and monitoring methods; this study is not evidence of safety in a physical plant.

## Literature review

1. **Downs and Vogel (1993), TEP benchmark.** Describes the plant-wide process model and its measurements and manipulated variables. It provides the simulation context used by this project.
2. **Cortes and Vapnik (1995), support-vector networks.** Develops the maximum-margin classifier for nonseparable data using slack variables and kernel-based decision functions. This supplies the mathematical foundation of the classifier.
3. **Chiang, Russell, and Braatz (2000), chemical-process diagnosis.** Studies PCA, discriminant PLS, and Fisher discriminant analysis on chemical-process data including TEP. The work demonstrates that fault diagnosis is a classification problem where class-discriminative information matters.
4. **Kulkarni, Jayaraman, and Kulkarni (2005), knowledge-incorporated SVM.** Applies SVM to TEP fault detection and uses process-related structure in the representation. It illustrates how domain knowledge may complement a generic classifier.
5. **Yin et al. (2012), TEP methods comparison.** Compares data-driven process monitoring and fault diagnosis methods on the TEP benchmark, emphasizing that implementation assumptions and computational cost affect performance comparisons.
6. **Rieth et al. (2017), anomaly-detection evaluation.** Discusses evaluation issues in TEP simulation data. It motivates reporting the split strategy and per-class detection results instead of presenting a single accuracy value alone.
7. **Yin et al. (2014), SVM-based TEP detection.** Studies SVM and parameter selection for TEP binary fault detection. It supports SVM as a relevant model family while making clear that tuning and evaluation protocol affect the result.
8. **Onel, Kieslich, and Pistikopoulos (2019), nonlinear SVM feature selection.** Develops nonlinear SVM-based feature selection for TEP fault detection and diagnosis. It shows a research direction beyond this project's all-feature binary baseline.

### Synthesis, suitability, and limitations

Traditional multivariate-statistics methods such as PCA and PLS are useful process-monitoring tools and can model correlations among sensors. Discriminant methods add class information. SVM is suitable for this project because its maximum-margin objective gives a clear optimization interpretation, and kernels allow a nonlinear boundary for interacting sensor variables. Scaling is essential because the sensor channels have different ranges.

SVM also has limits. Kernel training can become slow as sample size grows; an RBF classifier can be difficult to explain sensor by sensor; supervised classification assumes representative labels; and aggregating all nonzero TEP fault IDs into FAULTY removes fault identity. A row-wise score can miss fault evolution over time. TEP is simulated, so results do not establish performance on a real plant. Published results are not directly comparable unless dataset versions, preprocessing, sample selection, and train/test protocols match.

### Project contribution

The contribution is a reproducible teaching and demonstration pipeline: binary TEP preprocessing, grouped train/test splitting where possible, training-only imputation and scaling, a from-scratch linear SVM, an explicit small dual QP, an SVC implementation, and bounded experiments with class-specific fault recall and measured fit times. These are implementation and learning contributions, not a claim of a novel detection method.

> **Result provenance:** The metric and timing tables below are historical artifacts from the earlier implementation. The experiment pipeline has since been changed to fit preprocessing inside each experiment split, apply class weights consistently, tighten and diagnose the CVXOPT solve, and write each new run to `results/runs/<run_id>/`. Do not treat the historical timings as current-run measurements. The current run's `run_manifest.json` and `experiment_report.md` are generated from its measured result objects; training times vary between runs and machines.

## Mathematical formulation

Let each sensor row be `x_i ∈ R^d`, with class `y_i ∈ {-1,+1}`. A linear decision function is `f(x)=w^T x+b`. Its decision boundary is `w^T x+b=0`; the predicted class is the sign of `f(x)`. The distance from a point to this boundary is `|w^T x+b|/||w||`.

For correctly placed support planes `w^T x+b=+1` and `w^T x+b=-1`, the margin width is `2/||w||`. Maximizing this width is equivalent to minimizing `1/2 ||w||^2`. The hard-margin problem minimizes this objective subject to `y_i(w^T x_i+b)>=1`; it is unsuitable if classes overlap or a noisy observation prevents perfect separation.

### Soft-margin primal problem

Real sensor classes can overlap. Introduce slack variables `xi_i >= 0` to permit margin violations and choose a penalty `C > 0`:

```text
minimize over w, b, xi:  1/2 ||w||² + C Σ_i xi_i
subject to:              y_i (wᵀ x_i + b) ≥ 1 - xi_i
                          xi_i ≥ 0
```

`C` trades a wider margin against training violations. Larger `C` penalizes violations more; smaller `C` allows more. The objective is convex: the squared norm has a positive-semidefinite Hessian, and the constraints are affine. A local minimum of this feasible convex program is also a global minimum.

### Lagrangian and dual derivation

Let `alpha_i >= 0` multiply the margin constraints and `mu_i >= 0` multiply `xi_i >= 0`. The Lagrangian is:

```text
L = 1/2 ||w||² + C Σ_i xi_i
    - Σ_i alpha_i [ y_i(wᵀx_i+b) - 1 + xi_i ]
    - Σ_i mu_i xi_i
```

Stationarity gives `w = Σ_i alpha_i y_i x_i`, `Σ_i alpha_i y_i = 0`, and `C - alpha_i - mu_i = 0`. Since `mu_i >= 0`, it follows that `0 <= alpha_i <= C`. Substituting these conditions into `L` gives the dual:

```text
maximize over alpha:  Σ_i alpha_i - 1/2 Σ_iΣ_j alpha_i alpha_j y_i y_j (x_iᵀx_j)
subject to:            0 <= alpha_i <= C
                       Σ_i alpha_i y_i = 0
```

For a linear model, `w = Σ_i alpha_i y_i x_i`. For a kernel model, dot products are replaced by `K(x_i,x_j)`, and predictions use `sign(Σ_i alpha_i y_i K(x_i,x)+b)`. This project's CVXOPT example uses a small **linear** dual problem; scikit-learn SVC runs its own optimizer and is not claimed to call CVXOPT.

### KKT conditions and support vectors

At an optimum the Karush-Kuhn-Tucker conditions require:

1. **Primal feasibility:** `y_i(wᵀx_i+b) >= 1-xi_i` and `xi_i >= 0`.
2. **Dual feasibility:** `alpha_i >= 0` and `mu_i >= 0` (equivalently `0 <= alpha_i <= C`).
3. **Stationarity:** the derivatives of the Lagrangian with respect to `w`, `b`, and each `xi_i` are zero, yielding the equations above.
4. **Complementary slackness:** `alpha_i[y_i(wᵀx_i+b)-1+xi_i]=0` and `mu_i xi_i=0`.

Observations with nonzero `alpha_i` are support vectors: they affect the fitted boundary. Points well outside the margin have `alpha_i=0` and do not directly set the boundary.

## Implementations and experiment plan

- **From scratch:** `src/linear_svm_scratch.py` uses mini-batch subgradient updates for regularized hinge loss. It is a transparent educational approximation with a finite update budget, not a production replacement for a specialized optimizer.
- **Quadratic programming:** `src/optimization.py` builds the dual matrices `P`, `q`, `G`, `h`, `A`, and `b` for a small linear SVM. `G=[-I; I]` and `h=[0; C]` encode `0<=alpha<=C`; `A=yᵀ`, `b=0` encode the equality.
- **Library baseline:** scikit-learn `LinearSVC(loss="hinge")` uses the same small comparison split as the scratch and CVXOPT models. LinearSVC does not expose support-vector indices/counts.
- **C experiment:** compares LinearSVC hinge-loss models at C=0.1, 1, 10, and 100 on the same bounded fit subset and validation rows.
- **Kernel experiment:** compares linear SVC and RBF SVC at the same C and with the same fit and validation rows.
- **Sample-size experiment:** fits the configured kernel at several actual training sizes and plots measured training time. Sizes are capped by available training rows.
- **Fault recall:** the report includes faulty-class recall and the confusion matrix. In this binary mapping, a false negative is a faulty row predicted NORMAL.

When `USE_CLASS_BALANCING=True`, all three linear implementations use the same class penalty (w_k=n/(K n_k)). Scratch uses the weighted hinge term (C\sum_i w_{y_i}\max(0,1-y_i f(x_i))); the dual QP uses the equivalent per-row upper bounds (0\leq\alpha_i\leq Cw_{y_i}); LinearSVC uses `class_weight="balanced"`. For a no-free-support-vector QP solution, bias recovery uses the KKT-feasible interval midpoint when the interval is consistent and records an explicit support-vector-average fallback otherwise. Solver diagnostics record primal/dual objectives, duality gap, equality residual, bound violations, and KKT margin/complementary-slackness residuals.

Experiments split a validation subset from the main training partition. The main test partition is not used to choose among C values or kernels. Since the working directory currently contains the two training CSVs, the reported main test score is an internal group-aware split, not the official TEP testing files. The values are specific to the sample seed, settings, and machine. The equal per-file sample cap makes the experimental sample balanced; it is not an estimate of the plant's natural fault frequency.

## Measured results from an earlier run (historical)

The two local training files contain 250,000 normal rows and 5,000,000 faulty rows (5,250,000 total streamed). The actual input schema has 55 columns: 52 numeric measurements/manipulated variables and three metadata fields. A uniform reservoir sample of 20,000 rows was used: 10,000 normal and 10,000 faulty because the sample budget was divided equally by file. Missing-value and duplicate checks found zero in that sampled set; they were not a full-file duplicate audit. The official testing files were not present in the project folder and were not used.

The run-aware split held out complete source-file/simulation-run groups: 16,034 training rows and 3,966 main test rows. Imputation and standardization were fitted only on training rows. Primary RBF SVC (`C=1`, `gamma=scale`) scores: accuracy **0.7774**, faulty precision **0.9983**, faulty recall **0.5736**, faulty F1 **0.7286**, ROC-AUC **0.8323**. Fit time was **31.52 s** and test prediction time **10.03 s** on the recorded run. Precision is high but fault recall is modest: around 43% of faulty test rows were missed at the default decision threshold. Accuracy alone would conceal that operationally important error type. The balanced sample is not an estimate of natural plant fault frequency.

### C and kernel experiment

Each candidate fit used 4,000 rows from the development portion of the main training partition and was scored on the same 3,103-row run-aware inner validation subset. The 3,966-row main test set remained untouched during experiment choices.

This C study uses LinearSVC with hinge loss. For C=10 and C=100, liblinear reached the configured 50,000 iteration cap before its stopping tolerance. Those results are reported, but they are not fully converged solver results.

| C | Accuracy | Fault precision | Fault recall | Fault F1 | Fit seconds | Solver converged? |
|---:|---:|---:|---:|---:|---:|:---|
| 0.1 | 0.7235 | 0.9577 | 0.4509 | 0.6132 | 0.131 | Yes |
| 1 | 0.7332 | 0.9077 | 0.5020 | 0.6465 | 0.764 | Yes |
| 10 | 0.7425 | 0.9024 | 0.5272 | 0.6656 | 4.036 | No; iteration cap |
| 100 | 0.7464 | 0.9046 | 0.5345 | 0.6719 | 19.379 | No; iteration cap |

Scores rose gradually with C in this split, but that does not imply that larger C always improves generalization. The two higher-C solutions stopped at the iteration cap, so their values should be interpreted with that limitation.

### Linear kernel vs RBF kernel

Both SVC fits use 4,000 training rows, the same 3,103 validation rows, and C=1.

| Kernel | Accuracy | Fault precision | Fault recall | Fault F1 | Fit seconds |
|:---|---:|---:|---:|---:|---:|
| Linear | 0.7341 | 0.9090 | 0.5033 | 0.6479 | 4.327 |
| RBF | 0.7599 | 0.9923 | 0.5099 | 0.6737 | 1.675 |

The linear kernel gives a hyperplane in the standardized sensor space. RBF can represent curved boundaries via similarities in an implicit feature space; it performed somewhat better on these validation metrics for this bounded setting. The comparison is not an official test-set result.

### Sample-size fit time

The configured RBF SVC (`C=1`) was refit at each size. Measurements are from this run's computer and can vary with background load.

| Training rows | Fit seconds |
|---:|---:|
| 500 | 0.0361 |
| 1,000 | 0.1062 |
| 2,000 | 0.4081 |
| 4,000 | 1.6054 |

### Linear implementation comparison

All implementations used the same 1,000-row balanced subset (736 fit rows and 264 held-out rows), the same feature scaling, labels, C=1, and group-aware split. The split is still small, so metrics are coarse. Fit times compare these implementations on this small educational QP subset, not optimizers at production scale.

| Implementation | Accuracy | Fault precision | Fault recall | Fault F1 | Fit seconds | Support vectors |
|---|---:|---:|---:|---:|---:|---:|
| From-scratch subgradient | 0.6515 | 0.8000 | 0.3520 | 0.4889 | 0.1520 | 552 (within/at margin) |
| CVXOPT dual QP | 0.6856 | 0.7692 | 0.4800 | 0.5911 | 5.728 | 550 (nonzero alpha) |
| Scikit-learn LinearSVC | 0.6894 | 0.7945 | 0.4640 | 0.5859 | 0.0679 | Not exposed by LinearSVC |

The scratch loss was recorded after all **1,000 subgradient updates**. The normalized objective went from **0.9867** to **0.7135**; 174 of 999 successive steps rose slightly, with a largest rise of about 0.000238. This is expected in a subgradient method and is not a mathematical convergence proof. Learning-rate decay, finite iteration count, and minibatch selection affect the trace. CVXOPT solved the 736-row fitting split (within the 1,000-row cap) and saved a separate linear model. Tables/plots and separate models are organized in the experiment folders described in README.

The loss trace is the normalized primal objective `0.5/(C*n)||w||² + mean(hinge loss)`, measured after every mini-batch update. A convergence trend means later updates tend toward a stable objective/solution; it is not proven merely by the curve looking smooth. A larger learning rate can make early updates faster but may overshoot; a smaller rate is steadier but slower. The configured schedule decays the step after each approximate pass through the sampled training rows. The 174 slight upward steps are consistent with minibatch subgradient behavior.

### PCA feature reduction

The added PCA experiment uses all 52 numeric TEP sensor/process input columns (`xmeas_1`–`xmeas_41`, `xmv_1`–`xmv_11`); fault label, simulation run, sample number, and loader metadata are excluded. It reuses the primary workflow's run-aware 16,034/3,966 train/test split and the median imputer and standard scaler already fit using training rows only. PCA is also fit on the scaled training rows only. The smallest component count reaching the configurable 95% training variance target was 27, retaining 95.68% and reducing the input dimension by 48.08%. Each component is a weighted combination of original sensor measurements.

Both rows below use the same configured RBF SVC (`C=1`, `gamma="scale"`, balanced class weights) and the identical observations and labels. Each SVC was fit separately; the original-feature row is therefore a repeatable baseline on the shared holdout, not a newly tuned variant.

| Input | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit seconds | Prediction seconds |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original standardized features | 52 | 0.7774 | 0.9983 | 0.5736 | 0.7286 | 0.8323 | 8.182 | 2.839 |
| PCA features (95% target) | 27 | 0.7637 | 0.9956 | 0.5489 | 0.7076 | 0.8221 | 7.271 | 3.236 |

PCA slightly reduced SVC fit time in this run, while the decision metrics also declined; this run does not support a claim that PCA improved fault-detection performance overall. The timing of a single SVC run can vary with the machine and system load. Outputs are in `results/pca/`. The internal split comes from sampled training CSV rows; these are not official TEP testing-file metrics.

Small metric differences among scratch, CVXOPT, and LinearSVC are expected: subgradient learning has a finite update budget, CVXOPT solves the specified dual QP to its numerical tolerance, and LinearSVC uses liblinear with its own stopping tolerance. The comparison is a learning demonstration on a small, deliberately balanced subset; the higher LinearSVC/CVXOPT scores here should not be confused with the primary RBF model or an independent official test.

## Limitations, novelty, and future work

This project does not introduce a new classifier. The from-scratch optimizer has finite-iteration error, two high-C LinearSVC fits hit their iteration cap, the CVXOPT QP is limited to a small sample, and kernel experiments are computationally bounded. Binary mapping treats every observed nonzero `faultNumber` as FAULTY. The run uses only the training CSVs and an internal group split; evaluate once on the official testing CSVs as a final independent check after adding an explicit testing-data path. Future extensions include that official-file evaluation, fault identification, temporal early warning, online monitoring, severity estimation, and careful parameter selection.

## References

1. J. J. Downs and E. F. Vogel, “A plant-wide industrial process control problem,” *Computers & Chemical Engineering*, 17(3), 245–255, 1993. [doi:10.1016/0098-1354(93)80018-I](https://doi.org/10.1016/0098-1354(93)80018-I)
2. C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, 20, 273–297, 1995. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
3. L. H. Chiang, E. L. Russell, and R. D. Braatz, “Fault diagnosis in chemical processes using Fisher discriminant analysis, discriminant partial least squares, and principal component analysis,” *Chemometrics and Intelligent Laboratory Systems*, 50(2), 243–252, 2000. [doi:10.1016/S0169-7439(99)00061-1](https://doi.org/10.1016/S0169-7439(99)00061-1)
4. A. Kulkarni, V. K. Jayaraman, and B. D. Kulkarni, “Knowledge incorporated support vector machines to detect faults in Tennessee Eastman Process,” *Computers & Chemical Engineering*, 29(10), 2128–2133, 2005. [doi:10.1016/j.compchemeng.2005.06.006](https://doi.org/10.1016/j.compchemeng.2005.06.006)
5. S. Yin, S. X. Ding, A. Haghani, H. Hao, and P. Zhang, “A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process,” *Journal of Process Control*, 22(9), 1567–1581, 2012. [doi:10.1016/j.jprocont.2012.06.009](https://doi.org/10.1016/j.jprocont.2012.06.009)
6. C. A. Rieth, B. D. Amsel, R. Tran, and M. B. Cook, “Issues and Advances in Anomaly Detection Evaluation for Joint Human-Machine Systems,” 2017. [Springer chapter DOI:10.1007/978-3-319-60384-1_6](https://doi.org/10.1007/978-3-319-60384-1_6)
7. S. Yin, X. Gao, H. R. Karimi, and X. Zhu, “Study on Support Vector Machine-Based Fault Detection in Tennessee Eastman Process,” *Abstract and Applied Analysis*, 2014, Article 836895. [doi:10.1155/2014/836895](https://doi.org/10.1155/2014/836895)
8. M. Onel, C. A. Kieslich, and E. N. Pistikopoulos, “A nonlinear support vector machine-based feature selection approach for fault detection and diagnosis: Application to the Tennessee Eastman process,” *AIChE Journal*, 65(3), 992–1005, 2019. [doi:10.1002/aic.16497](https://doi.org/10.1002/aic.16497)
