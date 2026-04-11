from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# Paths
# =========================================================
base = Path(__file__).resolve().parent

agg_path = base / "../aggregated/aggregated_results.csv"
all_path = base / "../plots_metrics_cv/all_results.csv"
out_dir = base / "../plots_per_metrics"
out_dir.mkdir(parents=True, exist_ok=True)

agg_res = pd.read_csv(agg_path)
all_res = pd.read_csv(all_path)

# =========================================================
# Config
# =========================================================
test_metrics = [
    "test_accuracy@0.5",
    "test_auprc",
    "test_auroc",
    "test_balanced_accuracy@0.5",
    "test_f1@0.5",
    "test_f2@0.5",
    "test_minrp",
    "test_precision@0.5",
    "test_recall@0.5",
]

model_order = ["gru", "tcn", "sand", "grud", "strats"]
colors = [
    "#8172B2",  # GRU
    "#4C72B0",  # TCN
    "#55A868",  # SAND
    "#DD8452",  # GRUD
    "#C44E52",  # STRATS
]

datasets = ["mimic_iii", "physionet_2012"]
pct_order = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

sns.set_theme(style="whitegrid")


# =========================================================
# Helper
# =========================================================
def available_models_in_order(df):
    return [m for m in model_order if m in df["model"].unique()]


# =========================================================
# SUBSAMPLED
# =========================================================
def plot_subsampled(dataset):
    perturbation = "subsampled"
    metric = "test_auroc"

    sub_agg = agg_res[
        (agg_res["dataset"] == dataset) &
        (agg_res["perturbation"] == perturbation)
    ].copy()

    if sub_agg.empty:
        print(f"Skipping {dataset} - {perturbation}: no data")
        return

    grouped = (
        sub_agg
        .groupby(["model", "perturbation", "pct"])[test_metrics]
        .median()
        .reset_index()
    )

    models_here = available_models_in_order(grouped)

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4), sharey=True,
        gridspec_kw={"width_ratios": [2, 1]}
    )

    sns.lineplot(
        data=grouped,
        x="pct",
        y=metric,
        hue="model",
        hue_order=models_here,
        palette=colors[:len(models_here)],
        marker="o",
        dashes=False,
        ax=axes[0]
    )
    axes[0].set_xlabel("Perturbation (%)")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title(f"{dataset} - subsampled")
    axes[0].set_xticks(pct_order)
    axes[0].grid(alpha=0.3)

    sns.lineplot(
        data=grouped[grouped["pct"] <= 20],
        x="pct",
        y=metric,
        hue="model",
        hue_order=models_here,
        palette=colors[:len(models_here)],
        marker="o",
        dashes=False,
        ax=axes[1],
        legend=False
    )
    axes[1].set_xlabel("Perturbation (%)")
    axes[1].set_title("Zoomed (pct ≤ 20)")
    axes[1].set_xticks([1, 2, 5, 10, 20])
    axes[1].grid(alpha=0.3)

    axes[0].legend(title="Model")
    plt.tight_layout()

    save_path = out_dir / f"{dataset}_{perturbation}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================
# SPARSIFIED
# =========================================================
def plot_sparsified(dataset):
    perturbation = "sparsified-tsid-varid"
    metric = "test_auroc"

    sub_agg = agg_res[
        (agg_res["dataset"] == dataset) &
        (agg_res["perturbation"] == perturbation)
    ].copy()

    if sub_agg.empty:
        print(f"Skipping {dataset} - {perturbation}: no data")
        return

    sub_agg["pct"] = sub_agg["pct"].astype(int)

    # Keep your notebook behavior:
    # for mimic_iii, remove GRUD below 20 only
    if dataset == "mimic_iii":
        sub_agg = sub_agg[~((sub_agg["model"] == "grud") & (sub_agg["pct"] < 20))]

    mean_grouped = (
        sub_agg
        .groupby(["model", "perturbation", "pct"])[test_metrics]
        .mean()
        .reset_index()
    )
    median_grouped = (
        sub_agg
        .groupby(["model", "perturbation", "pct"])[test_metrics]
        .median()
        .reset_index()
    )

    sub_agg = sub_agg.sort_values(by="pct").copy()
    sub_agg["pct_str"] = sub_agg["pct"].astype(str)
    mean_grouped["pct_str"] = mean_grouped["pct"].astype(str)
    median_grouped["pct_str"] = median_grouped["pct"].astype(str)

    pct_order_str = [str(x) for x in pct_order if str(x) in sub_agg["pct_str"].unique()]
    models_here = available_models_in_order(sub_agg)

    plt.figure(figsize=(9, 6))

    ax = sns.stripplot(
        data=sub_agg,
        x="pct_str",
        y=metric,
        hue="model",
        hue_order=models_here,
        palette=colors[:len(models_here)],
        dodge=True,
        size=5,
        alpha=0.6,
        marker="o",
        jitter=True,
        legend=False,
        order=pct_order_str
    )

    x_positions = range(len(pct_order_str))
    pos_map = {cat: pos for cat, pos in zip(pct_order_str, x_positions)}

    median_grouped = median_grouped[median_grouped["pct_str"].isin(pct_order_str)].copy()
    mean_grouped = mean_grouped[mean_grouped["pct_str"].isin(pct_order_str)].copy()

    median_grouped["x_pos"] = median_grouped["pct_str"].map(pos_map)
    mean_grouped["x_pos"] = mean_grouped["pct_str"].map(pos_map)

    sns.lineplot(
        data=median_grouped,
        x="x_pos",
        y=metric,
        hue="model",
        hue_order=models_here,
        marker="o",
        linestyle="-",
        palette=colors[:len(models_here)],
        ax=ax,
        legend=False
    )

    sns.lineplot(
        data=mean_grouped,
        x="x_pos",
        y=metric,
        hue="model",
        hue_order=models_here,
        marker="o",
        linestyle="--",
        alpha=0.5,
        palette=colors[:len(models_here)],
        ax=ax,
        legend=False
    )

    ax.set_title(f"{dataset} - sparsified-tsid-varid")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(pct_order_str)
    ax.set_xlabel("Perturbation (%)")
    ax.set_ylabel("AUROC")
    ax.grid(alpha=0.3)

    model_handles = [
        Line2D([0], [0], color=colors[i], lw=2, label=models_here[i])
        for i in range(len(models_here))
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Median"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Mean"),
    ]

    legend1 = ax.legend(handles=model_handles, title="Model", loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, title="Statistic", loc="lower left")

    plt.tight_layout()

    save_path = out_dir / f"{dataset}_{perturbation}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================
# UNBALANCED
# =========================================================
def plot_unbalanced(dataset):
    perturbation = "unbalanced"
    metric = "test_auprc"

    sub_agg = agg_res[
        (agg_res["dataset"] == dataset) &
        (agg_res["perturbation"] == perturbation)
    ].copy()

    if sub_agg.empty:
        print(f"Skipping {dataset} - {perturbation}: no data")
        return

    sub_agg["pct"] = sub_agg["pct"].astype(int)

    grouped = (
        sub_agg
        .groupby(["model", "perturbation", "pct"])[test_metrics]
        .median()
        .reset_index()
    )

    grouped["random_baseline"] = grouped["pct"] / 100.0
    grouped["aupr_gain"] = grouped[metric] - grouped["random_baseline"]

    models_here = available_models_in_order(grouped)

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    sns.lineplot(
        data=grouped,
        x="pct",
        y=metric,
        hue="model",
        hue_order=models_here,
        marker="o",
        dashes=False,
        palette=colors[:len(models_here)],
        ax=axes[0]
    )

    unique_pct_baseline = grouped[["pct", "random_baseline"]].drop_duplicates().sort_values("pct")
    for _, row in unique_pct_baseline.iterrows():
        axes[0].scatter(
            row["pct"],
            row["random_baseline"],
            marker="x",
            color="black",
            s=80
        )

    axes[0].set_xlabel("Perturbation (%)")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_xticks(sorted(grouped["pct"].unique()))
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{dataset} - AUPRC")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), title="Model", loc="lower right")

    sns.lineplot(
        data=grouped,
        x="pct",
        y="aupr_gain",
        hue="model",
        hue_order=models_here,
        marker="o",
        dashes=False,
        palette=colors[:len(models_here)],
        ax=axes[1],
        legend=False
    )

    axes[1].set_xlabel("Perturbation (%)")
    axes[1].set_ylabel("AUPRC gain over random")
    axes[1].set_xticks(sorted(grouped["pct"].unique()))
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f"{dataset} - AUPRC gain")

    plt.tight_layout()

    save_path = out_dir / f"{dataset}_{perturbation}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("Available datasets:", agg_res["dataset"].unique())
    print("Available perturbations:", agg_res["perturbation"].unique())

    for dataset in datasets:
        plot_subsampled(dataset)
        plot_sparsified(dataset)
        plot_unbalanced(dataset)

    print(f"\nAll plots saved in: {out_dir}")


if __name__ == "__main__":
    main()