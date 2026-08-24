"""Plot generation, isolated from analysis so numeric functions stay importable headlessly.

Uses the Agg backend explicitly -- the original scripts relied on whatever backend
happened to be installed and one of them (``src/visual/visualize.py``) called
``plt.show()``, which blocks forever / raises in any non-interactive (CI, API, Docker)
context.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")


def save_histogram(df: pd.DataFrame, column: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
    ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    out_path = out_dir / f"hist_{column}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_correlation_heatmap(corr: pd.DataFrame, out_dir: Path, filename: str = "correlation_matrix.png") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_bar_by_group(series: pd.Series, title: str, ylabel: str, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
