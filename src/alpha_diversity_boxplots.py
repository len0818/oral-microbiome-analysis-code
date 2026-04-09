from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_ORDER = ["mild", "moderate", "severe"]


def detect_sample_id_column(df: pd.DataFrame) -> str:
    candidates = ["Description", "SampleID", "sample-id", "sample_id", "#SampleID"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def load_alpha_table(table_path: str | Path, metadata_path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(table_path, sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t")

    if "Unnamed: 0" in table.columns and "Description" not in table.columns:
        table = table.rename(columns={"Unnamed: 0": "Description"})

    sample_col_table = detect_sample_id_column(table)

    if metadata.shape[1] > 1:
        metadata = metadata.iloc[:, 1:].copy()

    if "Unnamed: 0" in metadata.columns and "Description" not in metadata.columns:
        metadata = metadata.rename(columns={"Unnamed: 0": "Description"})

    sample_col_meta = detect_sample_id_column(metadata)

    merged = table.merge(
        metadata,
        left_on=sample_col_table,
        right_on=sample_col_meta,
        how="inner",
    )
    return merged


def style_boxplot(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_facecolor("white")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, pad=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)


def plot_boxplot(ax: plt.Axes, data: pd.DataFrame, y: str, order: Iterable[str], ylabel: str, title: str) -> None:
    sns.boxplot(
        x="Group",
        y=y,
        data=data,
        order=list(order),
        ax=ax,
        color="white",
        flierprops={"marker": "o"},
        linewidth=1.2,
    )
    style_boxplot(ax, ylabel=ylabel, title=title)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create alpha-diversity boxplots.")
    parser.add_argument("--shannon-table", required=True)
    parser.add_argument("--shannon-metadata", required=True)
    parser.add_argument("--observed-table", required=True)
    parser.add_argument("--observed-metadata", required=True)
    parser.add_argument("--output", required=True, help="Output figure path, e.g. results/alpha_boxplots.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    sns.set_theme(style="white")
    shannon = load_alpha_table(args.shannon_table, args.shannon_metadata)
    observed = load_alpha_table(args.observed_table, args.observed_metadata)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    plot_boxplot(
        axes[0],
        shannon,
        y="shannon_entropy",
        order=DEFAULT_ORDER,
        ylabel="Shannon",
        title="Shannon index",
    )
    plot_boxplot(
        axes[1],
        observed,
        y="observed_features",
        order=DEFAULT_ORDER,
        ylabel="Observed features",
        title="Observed features",
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
