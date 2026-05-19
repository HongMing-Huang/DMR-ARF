"""
Generate publication-quality figures for the DMR-ARF paper.

Outputs:
- figures/fig1_distribution_shift.pdf
- figures/fig2_us_performance.pdf
- figures/fig3_multi_dataset.pdf
- figures/fig4_rolling_f1.pdf
- figures/fig5_ablation.pdf
- results/ablation/ablation_3x3.pdf
- results/ablation/ablation_3x3.png
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/paper_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

COLORS = {
    "DMR-ARF": "#D62728",
    "ARF_NoMemory": "#1F77B4",
    "OzaBagging_ADWIN": "#FF7F0E",
    "LeveragingBagging": "#2CA02C",
    "Static_XGBoost": "#8C564B",
    "SMOTE_XGBoost": "#9467BD",
}

MODEL_ORDER = [
    "DMR-ARF",
    "ARF_NoMemory",
    "OzaBagging_ADWIN",
    "LeveragingBagging",
    "Static_XGBoost",
    "SMOTE_XGBoost",
]

MODEL_LABELS = {
    "DMR-ARF": "DMR-ARF",
    "ARF_NoMemory": "ARF NoMemory",
    "OzaBagging_ADWIN": "OzaBagging ADWIN",
    "LeveragingBagging": "LeveragingBagging",
    "Static_XGBoost": "Static XGBoost",
    "SMOTE_XGBoost": "SMOTE+XGBoost",
}

SMOTE_US_ACCIDENTS = {
    "model": "SMOTE_XGBoost",
    "macro_f1": 0.340,
    "gmean": np.nan,
    "auc": np.nan,
    "f1_class0": 0.08,
    "sev1_recall": 0.17,
}

SMOTE_MULTI_DATASET_MACRO_F1 = {
    "US_Accidents": 0.340,
    "Electricity": 0.7098,
    "KDDCup99": 0.6262,
    "CoverType": 0.8149,
}

TRAIN_DISTRIBUTION = {"Sev1": 0.63, "Sev2": 78.86, "Sev3": 19.10, "Sev4": 1.79}
TEST_DISTRIBUTION = {"Sev1": 1.82, "Sev2": 84.44, "Sev3": 7.72, "Sev4": 6.00}
SEV1_RECALL = {
    "DMR-ARF": 0.44,
    "ARF_NoMemory": 0.00,
    "OzaBagging_ADWIN": 0.00,
    "LeveragingBagging": 0.00,
    "Static_XGBoost": 0.00,
    "SMOTE_XGBoost": 0.17,
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, fontsize: int = 8, precision: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.{precision}f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_fig1() -> None:
    print("Generating Figure 1...")
    labels = [
        "Severity 1\n(Fatal)",
        "Severity 2\n(Serious)",
        "Severity 3\n(Moderate)",
        "Severity 4\n(Minor)",
    ]
    train_values = np.array(list(TRAIN_DISTRIBUTION.values()))
    test_values = np.array(list(TEST_DISTRIBUTION.values()))
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_train = ax.bar(x - width / 2, train_values, width, label="Training set", color="#4C72B0")
    bars_test = ax.bar(x + width / 2, test_values, width, label="Test set", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    add_bar_labels(ax, bars_train)
    add_bar_labels(ax, bars_test)

    sev4_index = 3
    ax.annotate(
        "Drift +4.2pp",
        xy=(x[sev4_index] + width / 2, test_values[sev4_index]),
        xytext=(x[sev4_index] + 0.6, test_values[sev4_index] + 12),
        color="#D62728",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#D62728", "lw": 1.5},
    )
    sev3_index = 2
    ax.annotate(
        "Drift -11.4pp",
        xy=(x[sev3_index] + width / 2, test_values[sev3_index]),
        xytext=(x[sev3_index] - 0.75, test_values[sev3_index] + 18),
        color="#D62728",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#D62728", "lw": 1.5},
    )

    fig.tight_layout()
    save_figure(fig, "fig1_distribution_shift")


def plot_fig2() -> None:
    print("Generating Figure 2...")
    summary = pd.read_csv(RESULTS_DIR / "US_Accidents" / "summary.csv")
    summary = pd.concat([summary, pd.DataFrame([SMOTE_US_ACCIDENTS])], ignore_index=True)
    summary = summary.set_index("model").reindex(MODEL_ORDER).reset_index()
    summary["sev1_recall"] = summary["model"].map(SEV1_RECALL)
    summary.loc[summary["model"] == "SMOTE_XGBoost", "sev1_recall"] = SMOTE_US_ACCIDENTS["sev1_recall"]

    metric_specs = [
        ("macro_f1", "Macro F1"),
        ("gmean", "G-mean"),
        ("auc", "AUC"),
        ("f1_class0", "Severity-1 F1"),
        ("sev1_recall", "Severity-1 Recall"),
    ]

    x = np.arange(len(MODEL_ORDER))
    width = 0.65

    fig, axes = plt.subplots(1, 5, figsize=(16, 5))

    for col_idx, (metric, metric_label) in enumerate(metric_specs):
        ax = axes[col_idx]
        values = summary[metric].to_numpy()
        valid = ~pd.isna(values)

        ax.bar(
            x[valid],
            values[valid],
            width,
            color=[COLORS[m] for m in np.array(MODEL_ORDER)[valid]],
            edgecolor=["black" if m == "DMR-ARF" else "none" for m in np.array(MODEL_ORDER)[valid]],
            linewidth=[1.2 if m == "DMR-ARF" else 0 for m in np.array(MODEL_ORDER)[valid]],
            zorder=3,
        )
        for missing_idx in x[~valid]:
            ax.text(
                missing_idx,
                0.05,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
                rotation=90,
                zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(metric_label, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

        if col_idx == 0:
            ax.set_ylabel("Score")

        if metric == "gmean":
            ax.text(
                0.5, 0.87,
                "Only DMR-ARF\nachieves G-mean > 0",
                ha="center", va="center",
                color="#B22222", fontsize=8,
                transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FDE0DD",
                      "edgecolor": "none", "alpha": 0.9},
                zorder=4,
            )
            smote_idx = MODEL_ORDER.index("SMOTE_XGBoost")
            ax.text(
                smote_idx, -0.28,
                "†not applicable",
                ha="center", va="top",
                fontsize=7, color="gray", fontstyle="italic",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(pad=2.0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig2_us_performance.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_us_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3() -> None:
    print("Generating Figure 3...")
    master = pd.read_csv(RESULTS_DIR / "master_summary.csv")
    smote_rows = [
        {"dataset": dataset, "model": "SMOTE_XGBoost", "macro_f1": macro_f1}
        for dataset, macro_f1 in SMOTE_MULTI_DATASET_MACRO_F1.items()
    ]
    master = pd.concat([master, pd.DataFrame(smote_rows)], ignore_index=True)
    pivot = (
        master.pivot(index="dataset", columns="model", values="macro_f1")
        .reindex([
            "US_Accidents",
            "Electricity",
            "KDDCup99",
            "CoverType",
        ])
        .reindex(columns=MODEL_ORDER)
    )

    dataset_labels = [
        "US Accidents\n(385K, 4-class, traffic)",
        "Electricity\n(45K, binary, concept drift)",
        "KDDCup99\n(489K, 5-class, network)",
        "CoverType\n(581K, 7-class, forest)",
    ]

    x = np.arange(len(dataset_labels))
    width = 0.13
    fig, ax = plt.subplots(figsize=(16, 5.8))

    for model_idx, model_name in enumerate(MODEL_ORDER):
        offset = (model_idx - (len(MODEL_ORDER) - 1) / 2) * width
        values = pivot[model_name].to_numpy()
        valid = ~pd.isna(values)
        ax.bar(
            x[valid] + offset,
            values[valid],
            width,
            color=COLORS[model_name],
            label=MODEL_LABELS[model_name],
            zorder=3,
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1.2, label="F1=0.5 reference", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    fig.tight_layout()
    save_figure(fig, "fig3_multi_dataset")


def plot_fig4() -> None:
    print("Generating Figure 4...")
    dmr = pd.read_csv(RESULTS_DIR / "US_Accidents" / "rolling_DMR-ARF.csv").iloc[:, 0].to_numpy()
    arf = pd.read_csv(RESULTS_DIR / "US_Accidents" / "rolling_ARF_NoMemory.csv").iloc[:, 0].to_numpy()
    steps = np.arange(1, len(dmr) + 1) * 10000

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        steps,
        dmr,
        color=COLORS["DMR-ARF"],
        marker="o",
        markersize=5,
        linewidth=2.2,
        label="DMR-ARF",
        zorder=3,
    )
    ax.fill_between(steps, dmr - 0.008, dmr + 0.008, color=COLORS["DMR-ARF"], alpha=0.12, zorder=2)
    ax.plot(
        steps,
        arf,
        color=COLORS["ARF_NoMemory"],
        marker="s",
        markersize=4,
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        label=MODEL_LABELS["ARF_NoMemory"],
        zorder=3,
    )
    ax.axvline(150000, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(
        "Concept drift\n(distribution shift)",
        xy=(150000, dmr[14]),
        xytext=(170000, 0.34),
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.2},
        fontsize=10,
        color="gray",
    )

    ax.set_xticks([50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels(["50K", "100K", "150K", "200K", "250K", "300K"])
    ax.set_xlabel("Processed Samples")
    ax.set_ylabel("Rolling Macro F1 (window=10,000)")
    ax.set_ylim(0.25, 0.52)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig4_rolling_f1")


def plot_fig5() -> None:
    print("Generating Figure 5...")

    ablation_rows = [
        {
            "x": [50, 100, 200, 500, 1000],
            "row_label": "Buffer size M",
            "selected": 200,
            "selected_label": "Selected M=200",
            "macro_f1": [0.2693, 0.2668, 0.3091, 0.3212, 0.3307],
            "sev1_f1":  [0.0638, 0.0712, 0.1495, 0.1622, 0.1976],
            "sev1_rec": [0.0987, 0.1200, 0.4375, 0.6662, 0.5611],
        },
        {
            "x": [5, 10, 20, 50, 100],
            "row_label": "Replay interval T",
            "selected": 10,
            "selected_label": "Selected T=10",
            "macro_f1": [0.2789, 0.3091, 0.2782, 0.2519, 0.2378],
            "sev1_f1":  [0.1158, 0.1495, 0.1115, 0.0323, 0.0014],
            "sev1_rec": [0.7557, 0.4375, 0.1534, 0.0213, 0.0007],
        },
        {
            "x": [1, 3, 5, 10, 20],
            "row_label": "Replay batch B",
            "selected": 5,
            "selected_label": "Selected B=5",
            "macro_f1": [0.2449, 0.2779, 0.3091, 0.3069, 0.1391],
            "sev1_f1":  [0.0077, 0.1173, 0.1495, 0.1376, 0.0930],
            "sev1_rec": [0.0043, 0.1776, 0.4375, 0.6989, 0.8104],
        },
    ]

    metric_keys   = ["macro_f1",    "sev1_f1",        "sev1_rec"]
    metric_labels = ["Macro F1",    "Severity-1 F1",  "Severity-1 Recall"]
    ylim_per_col  = [(0.10, 0.35),  (0.00, 0.22),     (0.00, 0.90)]

    # sharey='col': each metric column shares one y-scale across all three rows
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey="col")

    for row_idx, row in enumerate(ablation_rows):
        x_vals = row["x"]
        x_pad  = (x_vals[-1] - x_vals[0]) * 0.08

        for col_idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row_idx, col_idx]

            ax.plot(
                x_vals,
                row[mkey],
                color="#4C72B0",
                linewidth=2,
                marker="o",
                markersize=6,
                zorder=3,
            )

            ax.axvline(
                row["selected"],
                color="red",
                alpha=0.6,
                linestyle="--",
                linewidth=1.5,
                zorder=2,
            )
            ax.text(
                row["selected"], 0.97,
                row["selected_label"],
                color="red", fontsize=7,
                ha="center", va="top", rotation=90,
                transform=ax.get_xaxis_transform(),
            )

            ax.set_xticks(x_vals)
            ax.set_xlim(x_vals[0] - x_pad, x_vals[-1] + x_pad)
            ax.tick_params(axis="x", labelrotation=35, labelsize=9)
            ax.grid(alpha=0.3, linestyle="--", zorder=0)
            ax.set_title(
                f"Effect of {row['row_label']} on {mlabel}",
                fontsize=10,
            )

            if row_idx == len(ablation_rows) - 1:
                ax.set_xlabel("Hyperparameter value")

    # Apply explicit ylim per column (propagates to all rows via sharey='col')
    for col_idx, (ylo, yhi) in enumerate(ylim_per_col):
        axes[0, col_idx].set_ylim(ylo, yhi)

    plt.tight_layout(rect=[0.07, 0, 1, 1], pad=2.0)

    # Bold row labels on the far left after layout is settled
    for row_idx, row in enumerate(ablation_rows):
        row_box = axes[row_idx, 0].get_position()
        fig.text(
            0.035,
            (row_box.y0 + row_box.y1) / 2,
            row["row_label"],
            ha="center", va="center",
            rotation=90, fontsize=12, fontweight="bold",
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ablation_dir = RESULTS_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig5_ablation.pdf",    dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_ablation.png",    dpi=300, bbox_inches="tight")
    fig.savefig(ablation_dir / "ablation_3x3.pdf",    dpi=300, bbox_inches="tight")
    fig.savefig(ablation_dir / "ablation_3x3.png",    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plotters = [plot_fig1, plot_fig2, plot_fig3, plot_fig4, plot_fig5]
    for index, plotter in enumerate(plotters, start=1):
        try:
            plotter()
        except Exception as exc:
            print(f"Error while generating Figure {index}: {exc}")
    print("All figures saved to figures/")


if __name__ == "__main__":
    main()
