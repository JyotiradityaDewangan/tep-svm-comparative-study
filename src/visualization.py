"""Create the requested readable plots from the sampled data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import config

def create_plots(data, features, y_test, predicted, scores, confusion):
    if not config.SAVE_PLOTS:
        return []
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    counts = data["_binary_target"].value_counts().reindex([0,1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6,4)); ax.bar(["NORMAL","FAULTY"], counts.values, color=["#4c9f70","#d95f59"])
    ax.set(title="Sampled Class Distribution", xlabel="Operating condition", ylabel="Observations")
    for i,v in enumerate(counts.values): ax.text(i,v,f"{v:,}",ha="center",va="bottom")
    fig.tight_layout(); path=config.PLOTS_DIR/"class_distribution.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    fig, ax = plt.subplots(figsize=(5,4)); sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL","FAULTY"], yticklabels=["NORMAL","FAULTY"], ax=ax)
    ax.set(title="Confusion Matrix", xlabel="Predicted class", ylabel="True class")
    fig.tight_layout(); path=config.HEATMAPS_DIR/"confusion_matrix.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    chosen = features[:config.MAX_CORRELATION_FEATURES]
    fig, ax = plt.subplots(figsize=(max(8,len(chosen)*.45), max(6,len(chosen)*.4)))
    sns.heatmap(data[chosen].corr(), cmap="coolwarm", center=0, ax=ax, xticklabels=True, yticklabels=True)
    ax.set(title=f"Sensor Correlation (first {len(chosen)} features)")
    fig.tight_layout(); path=config.HEATMAPS_DIR/"sensor_correlation.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    fig, ax = plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test,scores,name="SVM",ax=ax); ax.set_title("ROC Curve"); fig.tight_layout()
    path=config.PLOTS_DIR/"roc_curve.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    fig, ax = plt.subplots(figsize=(6,5)); PrecisionRecallDisplay.from_predictions(y_test,scores,name="SVM",ax=ax); ax.set_title("Precision-Recall Curve"); fig.tight_layout()
    path=config.PLOTS_DIR/"precision_recall_curve.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    for feature in features[:3]:
        fig, ax = plt.subplots(figsize=(7,4))
        for label, name, color in [(0,"NORMAL","#4c9f70"),(1,"FAULTY","#d95f59")]:
            vals=data.loc[data["_binary_target"]==label,feature].dropna()
            if len(vals)>3000: vals=vals.sample(3000,random_state=config.RANDOM_STATE)
            ax.hist(vals,bins=35,alpha=.55,label=name,color=color,density=True)
        ax.set(title=f"{feature}: Normal vs Faulty",xlabel=feature,ylabel="Density"); ax.legend(); fig.tight_layout()
        path=config.PLOTS_DIR/f"sensor_distribution_{feature}.png"; fig.savefig(path,dpi=180); plt.close(fig); saved.append(path)
    return saved
