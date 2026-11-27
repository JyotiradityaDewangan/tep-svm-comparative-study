"""Run bounded experiments comparing only linear/RBF SVM implementations."""
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import config
from src.linear_svm_scratch import LinearSVMScratch
from src.optimization import solve_linear_svm_qp, predict_linear_svm_qp
from src.preprocessing import fit_preprocessor_on_training

def _stratified_subset(X, y, max_rows, seed):
    """Take a reproducible subset without losing either class."""
    y = np.asarray(y)
    if max_rows >= len(y):
        return np.asarray(X), y
    selected, _ = train_test_split(np.arange(len(y)), train_size=max_rows,
                                   random_state=seed, stratify=y)
    return np.asarray(X)[selected], y[selected]

def _metrics(y_true, predictions):
    """Return comparable binary metrics with FAULTY treated as positive."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "fault_precision": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "fault_recall": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "fault_f1": f1_score(y_true, predictions, pos_label=1, zero_division=0),
    }

def _class_counts(y):
    y = np.asarray(y)
    return {"NORMAL": int(np.sum(y == 0)), "FAULTY": int(np.sum(y == 1))}

def _print_class_counts(title, y):
    counts = _class_counts(y)
    print(f"  {title} class counts: NORMAL={counts['NORMAL']:,}, FAULTY={counts['FAULTY']:,}")

def _fit_linear_svc(model, X, y):
    """Fit and report whether liblinear reached its stopping tolerance."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        started=time.perf_counter()
        model.fit(X,y)
        elapsed=time.perf_counter()-started
    converged=not any(issubclass(item.category,ConvergenceWarning) for item in captured)
    warning_messages = [str(item.message) for item in captured
                        if issubclass(item.category, ConvergenceWarning)]
    if not converged:
        for message in warning_messages:
            print(f"  WARNING: {message}")
    return elapsed,converged,warning_messages

def _inner_split(X, y, groups):
    """Create a validation split using training rows only; prefer whole runs."""
    dev_idx = val_idx = None
    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=30, test_size=0.20,
                                     random_state=config.RANDOM_STATE + 17)
        for candidate_dev, candidate_val in splitter.split(X, y, groups):
            if len(np.unique(y[candidate_dev])) == 2 and len(np.unique(y[candidate_val])) == 2:
                dev_idx, val_idx = candidate_dev, candidate_val
                break
    split_strategy = "group-aware split by source file and simulationRun"
    if dev_idx is None:
        dev_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.20,
                                            random_state=config.RANDOM_STATE + 17,
                                            stratify=y)
        split_strategy = "stratified row split (group split could not keep both classes in each partition)"
        print("  Experiment validation uses a stratified row split; a two-class run split was unavailable.")
    dev_groups = None if groups is None else np.asarray(groups)[dev_idx]
    return (np.asarray(X)[dev_idx], np.asarray(y)[dev_idx],
            np.asarray(X)[val_idx], np.asarray(y)[val_idx], dev_groups,
            dev_idx, val_idx, split_strategy)

def _balanced_implementation_split(X, y, groups=None):
    """Build one balanced raw dataset and one group-aware split for all SVMs."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    normal = np.flatnonzero(y == 0)
    faulty = np.flatnonzero(y == 1)
    cap = min(config.SCRATCH_COMPARE_ROWS, config.QP_MAX_ROWS)
    per_class = min(len(normal), len(faulty), cap // 2)
    if per_class < 2:
        raise ValueError("Not enough normal and faulty rows for the implementation comparison.")
    selected = np.concatenate((rng.choice(normal, per_class, replace=False),
                               rng.choice(faulty, per_class, replace=False)))
    X_small = X[selected]
    y_small = y[selected]
    small_groups = None if groups is None else np.asarray(groups)[selected]
    X_train_idx = X_test_idx = None
    if small_groups is not None:
        splitter=GroupShuffleSplit(n_splits=30,test_size=0.25,random_state=config.RANDOM_STATE)
        for train_idx,test_idx in splitter.split(X_small,y_small,small_groups):
            if len(np.unique(y_small[train_idx]))==2 and len(np.unique(y_small[test_idx]))==2:
                X_train_idx,X_test_idx=train_idx,test_idx
                break
    if X_train_idx is None:
        X_train_idx,X_test_idx=train_test_split(np.arange(len(y_small)),test_size=0.25,
            random_state=config.RANDOM_STATE,stratify=y_small)
        if small_groups is not None:
            print("  Linear implementation comparison uses stratified rows because a two-class run split was unavailable.")
    X_train,X_test=X_small[X_train_idx],X_small[X_test_idx]
    y_train,y_test=y_small[X_train_idx],y_small[X_test_idx]
    return X_train, X_test, y_train, y_test

def _run_implementation_comparison(X, y, groups=None, run_qp=True, run_library=True):
    """Fit scratch, CVXOPT, and scikit-learn LinearSVC on identical data."""
    X_train_raw, X_test_raw, y_train, y_test = _balanced_implementation_split(X, y, groups)
    X_train, X_test, imputer, scaler = fit_preprocessor_on_training(X_train_raw, X_test_raw)
    print("Linear implementation comparison uses one shared preprocessing fit:")
    _print_class_counts("fit", y_train)
    _print_class_counts("validation", y_test)
    signed_train = np.where(y_train == 1, 1.0, -1.0)
    rows = []
    model_dir = config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Scratch model: w and b are updated explicitly from hinge-loss subgradients.
    scratch = LinearSVMScratch(C=config.SVM_C, iterations=config.SCRATCH_ITERATIONS,
        learning_rate=config.SCRATCH_LEARNING_RATE, random_state=config.RANDOM_STATE,
        class_weight="balanced" if config.USE_CLASS_BALANCING else None)
    started = time.perf_counter()
    scratch.fit(X_train, signed_train)
    elapsed = time.perf_counter() - started
    scratch_prediction = (scratch.predict(X_test) == 1).astype(int)
    scratch_metrics = _metrics(y_test, scratch_prediction)
    scratch_margins = signed_train * scratch.decision_function(X_train)
    scratch_support = int(np.sum(scratch_margins <= 1.0 + 1e-6))
    rows.append({"implementation":"From-scratch hinge subgradient", **scratch_metrics,
        "training_time_seconds":elapsed,"margin_active_points":scratch_support,
        "support_vectors":"not defined by primal scratch model",
        "training_rows":len(y_train),"test_rows":len(y_test),"C":config.SVM_C})
    if config.SAVE_MODEL:
        joblib.dump(scratch, model_dir / "linear_svm_from_scratch.pkl")
        result_dir=config.RESULTS_DIR/"svm_from_scratch"
        result_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(scratch,result_dir/"linear_svm_from_scratch.pkl")

    # The QP is capped at a small row count because P and G are dense matrices.
    qp = None
    qp_elapsed = np.nan
    if run_qp:
        try:
            started = time.perf_counter()
            qp = solve_linear_svm_qp(X_train, signed_train, C=config.SVM_C,
                class_weight="balanced" if config.USE_CLASS_BALANCING else None)
            qp_elapsed = time.perf_counter() - started
            qp_prediction = (predict_linear_svm_qp(qp, X_test) == 1).astype(int)
            qp_metrics = _metrics(y_test, qp_prediction)
            rows.append({"implementation":"CVXOPT dual quadratic program", **qp_metrics,
                "training_time_seconds":qp_elapsed,"support_vectors":qp["support_vector_count"],
                "bias_method":qp["diagnostics"]["bias_method"],
                "duality_gap":qp["diagnostics"]["duality_gap"],
                "max_kkt_margin_violation":qp["diagnostics"]["max_kkt_margin_violation"],
                "equality_residual":qp["diagnostics"]["equality_residual"],
                "solver_status":qp["status"],
                "training_rows":len(y_train),"test_rows":len(y_test),"C":config.SVM_C})
            qp_model={"weights":qp["weights"],"bias":qp["bias"],"alpha":qp["alpha"],
                      "support_vector_count":qp["support_vector_count"],
                      "upper_bounds":qp["upper_bounds"],"diagnostics":qp["diagnostics"]}
            if config.SAVE_MODEL:
                joblib.dump(qp_model, model_dir / "linear_svm_cvxopt.pkl")
                result_dir=config.RESULTS_DIR/"cvxopt_svm"
                result_dir.mkdir(parents=True,exist_ok=True)
                joblib.dump(qp_model,result_dir/"linear_svm_cvxopt.pkl")
        except ImportError as exc:
            print(f"CVXOPT comparison skipped: {exc}. Install it with 'python -m pip install cvxopt'.")
        except Exception as exc:
            print(f"CVXOPT comparison failed clearly; scratch and LinearSVC results are retained: {exc}")
            rows.append({"implementation":"CVXOPT dual quadratic program","accuracy":np.nan,
                "fault_precision":np.nan,"fault_recall":np.nan,"fault_f1":np.nan,
                "training_time_seconds":np.nan,"support_vectors":np.nan,
                "training_rows":len(y_train),"test_rows":len(y_test),"C":config.SVM_C,
                "status":str(exc)})

    # LinearSVC is the requested library primal/hinge-loss linear SVM.
    if run_library:
        library = LinearSVC(C=config.SVM_C, loss="hinge", dual=True,
            class_weight="balanced" if config.USE_CLASS_BALANCING else None,
            random_state=config.RANDOM_STATE, max_iter=50_000, tol=1e-3)
        elapsed,converged,warning_messages=_fit_linear_svc(library,X_train,y_train)
        library_prediction = library.predict(X_test)
        rows.append({"implementation":"scikit-learn LinearSVC (hinge)",
            **_metrics(y_test, library_prediction),"training_time_seconds":elapsed,
            # LinearSVC does not expose support-vector indices/counts.
            "support_vectors":"not exposed by LinearSVC","training_rows":len(y_train),
            "test_rows":len(y_test),"C":config.SVM_C,
            "solver_iterations":int(np.max(library.n_iter_)),"solver_converged":converged,
            "convergence_warnings":" | ".join(warning_messages)})
    table = pd.DataFrame(rows)
    return table, scratch

def _run_c_experiment(X_train, y_train, X_validation, y_validation):
    """Use LinearSVC with hinge loss to isolate the effect of C."""
    X_fit_raw, y_fit = _stratified_subset(X_train, y_train,
        min(config.EXPERIMENT_MAX_ROWS, len(y_train)), config.RANDOM_STATE)
    X_fit, X_validation, _, _ = fit_preprocessor_on_training(X_fit_raw, X_validation)
    _print_class_counts("C experiment fit", y_fit)
    _print_class_counts("C experiment validation", y_validation)
    rows = []
    for c_value in config.EXPERIMENT_C_VALUES:
        model = LinearSVC(C=c_value, loss="hinge", dual=True,
            class_weight="balanced" if config.USE_CLASS_BALANCING else None,
            random_state=config.RANDOM_STATE, max_iter=50_000, tol=1e-3)
        elapsed,converged,warning_messages=_fit_linear_svc(model,X_fit,y_fit)
        rows.append({"C":c_value, **_metrics(y_validation, model.predict(X_validation)),
                     "training_time_seconds":elapsed,"training_rows":len(y_fit),
                     "validation_rows":len(y_validation),"implementation":"LinearSVC hinge",
                      "solver_iterations":int(np.max(model.n_iter_)),"solver_converged":converged,
                      "convergence_warnings":" | ".join(warning_messages)})
        print(f"  LinearSVC C={c_value:g}; fault recall={rows[-1]['fault_recall']:.4f}")
    return pd.DataFrame(rows)

def _run_kernel_experiment(X_train, y_train, X_validation, y_validation):
    """Compare linear and RBF SVC on the exact same fit and validation rows."""
    X_fit_raw, y_fit = _stratified_subset(X_train, y_train,
        min(config.EXPERIMENT_MAX_ROWS, len(y_train)), config.RANDOM_STATE)
    X_fit, X_validation, _, _ = fit_preprocessor_on_training(X_fit_raw, X_validation)
    _print_class_counts("kernel experiment fit", y_fit)
    _print_class_counts("kernel experiment validation", y_validation)
    rows = []
    for kernel in config.EXPERIMENT_KERNELS:
        model = SVC(kernel=kernel, C=config.SVM_C,
            gamma=config.SVM_GAMMA if kernel == "rbf" else "scale",
            class_weight="balanced" if config.USE_CLASS_BALANCING else None,
            random_state=config.RANDOM_STATE)
        started = time.perf_counter()
        model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - started
        rows.append({"kernel":kernel, **_metrics(y_validation, model.predict(X_validation)),
                     "training_time_seconds":elapsed,"training_rows":len(y_fit),
                     "validation_rows":len(y_validation),"C":config.SVM_C,
                     "support_vectors":int(model.n_support_.sum())})
        print(f"  {kernel.upper()} kernel; fault recall={rows[-1]['fault_recall']:.4f}")
    return pd.DataFrame(rows)

def _run_sample_size_experiment(X_train, y_train):
    rows = []
    for requested_size in config.EXPERIMENT_SAMPLE_SIZES:
        actual_size = min(int(requested_size), len(y_train))
        if actual_size < 4:
            continue
        X_fit_raw, y_fit = _stratified_subset(X_train, y_train, actual_size,
                                          config.RANDOM_STATE + actual_size)
        X_fit, _, _, _ = fit_preprocessor_on_training(X_fit_raw, None)
        model = SVC(kernel=config.SVM_KERNEL, C=config.SVM_C,
                    gamma=config.SVM_GAMMA,
                    class_weight="balanced" if config.USE_CLASS_BALANCING else None,
                    random_state=config.RANDOM_STATE)
        started = time.perf_counter(); model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - started
        rows.append({"training_rows":len(y_fit),"kernel":config.SVM_KERNEL,
                     "C":config.SVM_C,"training_time_seconds":elapsed})
    return pd.DataFrame(rows)

def _save_table(table, csv_name, excel_name, sheet_name, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if config.SAVE_CSV_REPORTS:
        table.to_csv(folder / csv_name, index=False)
    if config.SAVE_EXCEL_REPORTS:
        with pd.ExcelWriter(folder / excel_name, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            sheet = writer.sheets[sheet_name[:31]]
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = min(max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 48)
                sheet.column_dimensions[column[0].column_letter].width = width

def _save_line_plot(table, x_name, y_names, title, xlabel, output):
    if not config.SAVE_PLOTS or table.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in y_names:
        ax.plot(table[x_name], table[name], marker="o", label=name.replace("_", " ").title())
    ax.set(title=title, xlabel=xlabel, ylabel="Score")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)

def run_svm_experiments(X_train, y_train, groups=None, selected="all"):
    """Run selected experiments using training data only; never touch main test rows."""
    print("\nSVM experiments (only SVM implementations; main holdout remains untouched)")
    X_train = np.asarray(X_train); y_train = np.asarray(y_train)
    X_development, y_development, X_validation, y_validation, development_groups, development_indices, validation_indices, split_strategy = _inner_split(X_train, y_train, groups)
    print("Experiment development/validation split:")
    _print_class_counts("development", y_development)
    _print_class_counts("validation", y_validation)
    output = {}
    run_metadata = {
        "split_strategy": split_strategy,
        "development_rows": int(len(y_development)),
        "validation_rows": int(len(y_validation)),
        "development_class_counts": _class_counts(y_development),
        "validation_class_counts": _class_counts(y_validation),
    }
    selected = {selected} if isinstance(selected, str) else set(selected)
    if "all" in selected or "implementations" in selected or "convergence" in selected:
        table, scratch = _run_implementation_comparison(
            X_development, y_development, development_groups,
            run_qp=("all" in selected or "implementations" in selected),
            run_library=("all" in selected or "implementations" in selected))
        if "all" in selected or "implementations" in selected:
            output["implementations"] = table
            folder = config.RESULTS_DIR / "svm_comparison"
            _save_table(table,"linear_implementation_comparison.csv","linear_implementation_comparison.xlsx","Implementations",folder)
            print("\nLinear SVM implementation comparison:")
            print(table.to_string(index=False))
            for method, folder_name, base_name in (
                ("From-scratch hinge subgradient","svm_from_scratch","scratch_svm"),
                ("CVXOPT dual quadratic program","cvxopt_svm","cvxopt_svm")):
                row=table.loc[table["implementation"]==method]
                if not row.empty:
                    _save_table(row,f"{base_name}_metrics.csv",f"{base_name}_metrics.xlsx",
                                "Metrics",config.RESULTS_DIR/folder_name)
            # Preserve the project's existing general report locations.
            if config.SAVE_CSV_REPORTS:
                config.CSV_DIR.mkdir(parents=True,exist_ok=True)
                table.to_csv(config.CSV_DIR/"svm_implementation_comparison.csv",index=False)
            if config.SAVE_PLOTS:
                fig, ax = plt.subplots(figsize=(8,5)); positions=np.arange(len(table))
                ax.bar(positions-.18,table["accuracy"].fillna(0),.36,label="Accuracy")
                ax.bar(positions+.18,table["fault_recall"].fillna(0),.36,label="Fault recall")
                ax.set_xticks(positions,table["implementation"],rotation=15,ha="right")
                ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Linear SVM Implementations")
                ax.legend(); fig.tight_layout(); folder.mkdir(parents=True,exist_ok=True)
                fig.savefig(folder/"linear_implementation_comparison.png",dpi=180); plt.close(fig)
                config.PLOTS_DIR.mkdir(parents=True,exist_ok=True)
                shutil.copy2(folder/"linear_implementation_comparison.png",
                             config.PLOTS_DIR/"svm_implementation_comparison.png")
        if "all" in selected or "convergence" in selected:
            convergence = pd.DataFrame({"iteration":np.arange(1,len(scratch.loss_history)+1),
                                        "training_objective":scratch.loss_history})
            output["convergence"] = convergence
            print(f"\nScratch objective trace: {len(convergence):,} updates; "
                  f"initial={convergence['training_objective'].iloc[0]:.6f}; "
                  f"final={convergence['training_objective'].iloc[-1]:.6f}. "
                  "This trace alone is not a proof of convergence.")
            folder=config.RESULTS_DIR/"convergence"
            _save_table(convergence,"scratch_svm_loss.csv","scratch_svm_loss.xlsx","Loss by Epoch",folder)
            if config.SAVE_PLOTS:
                fig,ax=plt.subplots(figsize=(7,5)); ax.plot(convergence["iteration"],convergence["training_objective"])
                ax.set(title="From-Scratch Linear SVM Convergence",xlabel="Subgradient update iteration",ylabel="Normalized primal objective")
                ax.grid(True,alpha=.3); fig.tight_layout(); folder.mkdir(parents=True,exist_ok=True)
                fig.savefig(folder/"scratch_svm_convergence.png",dpi=180); plt.close(fig)

    if "all" in selected or "c" in selected:
        c_table=_run_c_experiment(X_development,y_development,X_validation,y_validation)
        output["c_parameter"]=c_table
        print("\nC parameter experiment:")
        print(c_table.to_string(index=False))
        folder=config.RESULTS_DIR/"c_parameter_experiment"
        _save_table(c_table,"linear_svc_c_parameter.csv","linear_svc_c_parameter.xlsx","C Experiment",folder)
        if config.SAVE_CSV_REPORTS:
            config.CSV_DIR.mkdir(parents=True,exist_ok=True)
            c_table.to_csv(config.CSV_DIR/"svm_c_parameter_experiment.csv",index=False)
        if config.SAVE_PLOTS:
            # The configured C values span orders of magnitude.
            fig,ax=plt.subplots(figsize=(7,5))
            for metric in ("accuracy","fault_precision","fault_recall","fault_f1"):
                ax.plot(c_table["C"],c_table[metric],marker="o",label=metric.replace("_"," ").title())
            ax.set_xscale("log"); ax.set_ylim(0,1.05); ax.set(title="Effect of C on LinearSVC",xlabel="C (log scale)",ylabel="Score")
            ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); folder.mkdir(parents=True,exist_ok=True)
            fig.savefig(folder/"c_parameter_effect.png",dpi=180); plt.close(fig)

    if "all" in selected or "kernel" in selected:
        kernel_table=_run_kernel_experiment(X_development,y_development,X_validation,y_validation)
        output["kernel"]=kernel_table
        print("\nLinear versus RBF kernel experiment:")
        print(kernel_table.to_string(index=False))
        folder=config.RESULTS_DIR/"kernel_experiment"
        _save_table(kernel_table,"linear_vs_rbf.csv","linear_vs_rbf.xlsx","Kernel Experiment",folder)
        if config.SAVE_CSV_REPORTS:
            config.CSV_DIR.mkdir(parents=True,exist_ok=True)
            kernel_table.to_csv(config.CSV_DIR/"svm_kernel_experiment.csv",index=False)
        if config.SAVE_PLOTS:
            positions=np.arange(len(kernel_table)); fig,ax=plt.subplots(figsize=(7,5))
            for offset,metric in ((-.3,"accuracy"),(-.1,"fault_precision"),(.1,"fault_recall"),(.3,"fault_f1")):
                ax.bar(positions+offset,kernel_table[metric],.2,label=metric.replace("_"," ").title())
            ax.set_xticks(positions,kernel_table["kernel"].str.upper()); ax.set_ylim(0,1.05)
            ax.set(title="Linear vs RBF SVM",xlabel="Kernel",ylabel="Score")
            ax.legend(); fig.tight_layout(); folder.mkdir(parents=True,exist_ok=True)
            fig.savefig(folder/"linear_vs_rbf_comparison.png",dpi=180); plt.close(fig)

    if "all" in selected or "sample-size" in selected:
        size_table=_run_sample_size_experiment(X_development,y_development)
        output["sample_size"]=size_table
        print("Sample-size experiment records fitting time only; it does not measure predictive performance at each size.")
        print(size_table.to_string(index=False))
        folder=config.RESULTS_DIR/"svm_comparison"
        _save_table(size_table,"sample_size_fit_time.csv","sample_size_fit_time.xlsx","Sample Size",folder)
        if config.SAVE_PLOTS and len(size_table)>1:
            config.PLOTS_DIR.mkdir(parents=True,exist_ok=True)
            folder.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(7,5))
            ax.plot(size_table["training_rows"],size_table["training_time_seconds"],marker="o")
            ax.set(title="SVM Fit Time vs Sample Size",xlabel="Training observations",ylabel="Fit time (seconds)")
            ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(folder/"training_time_vs_sample_size.png",dpi=180); plt.close(fig)
            # Keep the previously used location too.
            fig,ax=plt.subplots(figsize=(7,5)); ax.plot(size_table["training_rows"],size_table["training_time_seconds"],marker="o")
            ax.set(title="SVM Fit Time vs Sample Size",xlabel="Training observations",ylabel="Fit time (seconds)")
            ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(config.PLOTS_DIR/"training_time_vs_sample_size.png",dpi=180); plt.close(fig)
        output["sample_size"]=size_table
    output["_run_metadata"] = run_metadata
    if selected == {"all"} and config.SAVE_EXCEL_REPORTS:
        summary_folder=config.RESULTS_DIR/"svm_comparison"
        summary_folder.mkdir(parents=True,exist_ok=True)
        sheet_names={"implementations":"Implementations","convergence":"Convergence",
                     "c_parameter":"C Parameter","kernel":"Kernel","sample_size":"Sample Size"}
        with pd.ExcelWriter(summary_folder/"svm_experiments.xlsx",engine="openpyxl") as writer:
            for key,table in output.items():
                if key.startswith("_"):
                    continue
                table.to_excel(writer,sheet_name=sheet_names[key],index=False)
                sheet=writer.sheets[sheet_names[key]]
                sheet.freeze_panes="A2"
                for column in sheet.columns:
                    width=min(max(max(len(str(cell.value or "")) for cell in column)+2,12),48)
                    sheet.column_dimensions[column[0].column_letter].width=width
        # Refresh the original workbook location for users who already had it.
        config.EXCEL_DIR.mkdir(parents=True,exist_ok=True)
        # The combined workbook is written only for a complete experiment run.
        config.EXCEL_DIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(summary_folder/"svm_experiments.xlsx",
                     config.EXCEL_DIR/"svm_experiments.xlsx")
    return output
