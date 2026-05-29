"""Generate result figures from benchmark CSV outputs."""

import matplotlib.pyplot as plt
import pandas as pd

FOLD_CSV = "results/per_fold_metrics.csv"
SUMMARY_CSV = "results/metrics_summary.csv"
COLORS = {"LR": "#2196F3", "RF": "#FF9800", "CNN": "#4CAF50"}
MODELS = ["LR", "RF", "CNN"]
METRICS = ["balanced_accuracy", "auroc", "sensitivity", "specificity"]
METRIC_LABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "auroc": "AUROC",
    "sensitivity": "Sensitivity (PD)",
    "specificity": "Specificity (HC)",
}


def plot_per_fold(df: pd.DataFrame) -> None:
    """Figure 1: per-fold metric strip chart with mean line per model."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Per-Fold Metrics — 5-Fold Subject-Independent Cross-Validation",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    x_positions = {m: i for i, m in enumerate(MODELS)}
    jitter = {"LR": -0.15, "RF": 0.0, "CNN": 0.15}

    for ax, metric in zip(axes, METRICS):
        for model in MODELS:
            fold_vals = df[df["model"] == model][metric].values
            x = x_positions[model] + jitter[model]
            xs = [x] * len(fold_vals)
            ax.scatter(xs, fold_vals, color=COLORS[model], s=50, zorder=3, alpha=0.9)
            ax.hlines(
                fold_vals.mean(),
                x - 0.12,
                x + 0.12,
                colors=COLORS[model],
                linewidths=2,
                zorder=4,
            )

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score" if metric == "balanced_accuracy" else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m],
                   markersize=8, label=m)
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig("results/per_fold_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/per_fold_metrics.png")


def plot_sens_spec(df: pd.DataFrame) -> None:
    """Figure 2: grouped bar chart of sensitivity vs specificity per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.3
    x = range(len(MODELS))

    sens = [df.loc[df["model"] == m, "mean_sensitivity"].values[0] for m in MODELS]
    spec = [df.loc[df["model"] == m, "mean_specificity"].values[0] for m in MODELS]
    sens_err = [df.loc[df["model"] == m, "sd_sensitivity"].values[0] for m in MODELS]
    spec_err = [df.loc[df["model"] == m, "sd_specificity"].values[0] for m in MODELS]

    bars1 = ax.bar(
        [i - bar_width / 2 for i in x], sens, bar_width,
        yerr=sens_err, capsize=4,
        label="Sensitivity (PD)", color="#E57373", alpha=0.9,
    )
    bars2 = ax.bar(
        [i + bar_width / 2 for i in x], spec, bar_width,
        yerr=spec_err, capsize=4,
        label="Specificity (HC)", color="#64B5F6", alpha=0.9,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score (mean ± std across folds)", fontsize=11)
    ax.set_title(
        "Sensitivity vs Specificity per Model\n"
        "5-Fold Subject-Independent Cross-Validation",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.03,
            f"{h:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig("results/sensitivity_specificity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/sensitivity_specificity.png")


if __name__ == "__main__":
    fold_df = pd.read_csv(FOLD_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)
    plot_per_fold(fold_df)
    plot_sens_spec(summary_df)
