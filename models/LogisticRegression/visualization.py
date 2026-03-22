"""
visualization.py
─────────────────────────────────────────────────────────────
All matplotlib figure factories – purely functional, no Streamlit imports.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression

from models import Perceptron, logistic_boundary_params


# ──────────────────────────────────────────────────────────
# Shared style helpers
# ──────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0F1117",
    "surface": "#1A1D27",
    "grid": "#2A2D3A",
    "class_neg": "#FF6B6B",   # class –1
    "class_pos": "#4ECDC4",   # class +1
    "perceptron": "#FFD93D",
    "logistic": "#6BCB77",
    "neutral": "#A8AABD",
    "accent": "#C77DFF",
}

# Custom colormaps for decision regions
_cmap_neg = LinearSegmentedColormap.from_list("neg", ["#0F1117", "#FF6B6B"], N=2)
_cmap_pos = LinearSegmentedColormap.from_list("pos", ["#0F1117", "#4ECDC4"], N=2)
_region_cmap = LinearSegmentedColormap.from_list(
    "region", ["#FF6B6B33", "#4ECDC433"], N=256
)


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    """Apply consistent dark theme to a figure and its axes list."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["neutral"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["neutral"])
        ax.yaxis.label.set_color(PALETTE["neutral"])
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)


def _scatter_data(ax: plt.Axes, X: np.ndarray, y: np.ndarray, alpha: float = 0.75) -> None:
    """Scatter plot of two-class data on a given axes."""
    mask_neg = y == -1
    mask_pos = y == 1
    ax.scatter(
        X[mask_neg, 0], X[mask_neg, 1],
        c=PALETTE["class_neg"], s=30, alpha=alpha,
        edgecolors="white", linewidths=0.3, label="Class −1",
    )
    ax.scatter(
        X[mask_pos, 0], X[mask_pos, 1],
        c=PALETTE["class_pos"], s=30, alpha=alpha,
        edgecolors="white", linewidths=0.3, label="Class +1",
    )


def _draw_line(
    ax: plt.Axes,
    m: float | None,
    b: float,
    x_range: tuple[float, float],
    color: str,
    label: str,
    lw: float = 2.0,
    linestyle: str = "-",
) -> None:
    """Draw a decision boundary line (slope-intercept) on an axes."""
    x_min, x_max = x_range
    if m is None:
        # Vertical line
        ax.axvline(b, color=color, linewidth=lw, linestyle=linestyle, label=label)
    else:
        xs = np.linspace(x_min, x_max, 300)
        ys = m * xs + b
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=linestyle, label=label)


def _fill_decision_regions(
    ax: plt.Axes,
    m: float | None,
    b: float,
    feature_range: tuple[float, float, float, float],
    color_neg: str,
    color_pos: str,
    alpha: float = 0.12,
) -> None:
    """Shade the two half-planes of a linear decision boundary."""
    x_min, x_max, y_min, y_max = feature_range
    xx = np.linspace(x_min, x_max, 300)

    if m is None:
        boundary_x = b
        ax.fill_betweenx([y_min, y_max], x_min, boundary_x, color=color_neg, alpha=alpha)
        ax.fill_betweenx([y_min, y_max], boundary_x, x_max, color=color_pos, alpha=alpha)
    else:
        boundary_y = m * xx + b
        ax.fill_between(xx, y_min, boundary_y, color=color_neg, alpha=alpha)
        ax.fill_between(xx, boundary_y, y_max, color=color_pos, alpha=alpha)


# ──────────────────────────────────────────────────────────
# Public figure factories
# ──────────────────────────────────────────────────────────

def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Synthetic Dataset",
) -> plt.Figure:
    """Simple scatter plot of the full dataset."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _scatter_data(ax, X, y)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(framealpha=0.2, labelcolor="white", facecolor=PALETTE["surface"])
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_final_boundaries(
    X: np.ndarray,
    y: np.ndarray,
    perceptron: Perceptron,
    lr_model: LogisticRegression,
    feature_range: tuple[float, float, float, float],
) -> plt.Figure:
    """
    Side-by-side comparison of Perceptron vs Logistic Regression
    final decision boundaries.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x_min, x_max, y_min, y_max = feature_range
    x_range = (x_min, x_max)

    titles = ["Perceptron", "Logistic Regression"]
    colors = [PALETTE["perceptron"], PALETTE["logistic"]]

    # --- Perceptron boundary (last epoch)
    m_p, b_p, _ = perceptron.boundary_history[-1]
    # --- LR boundary
    m_l, b_l = logistic_boundary_params(lr_model)

    boundaries = [(m_p, b_p), (m_l, b_l)]

    for ax, (m, b), title, color in zip(axes, boundaries, titles, colors):
        _scatter_data(ax, X, y)
        _fill_decision_regions(ax, m, b, feature_range, PALETTE["class_neg"], PALETTE["class_pos"])
        _draw_line(ax, m, b, x_range, color, label="Decision Boundary", lw=2.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Accuracy badge
        if title == "Perceptron":
            acc = perceptron.boundary_history[-1][2]
        else:
            acc = lr_model.score(X, y)
        ax.text(
            0.97, 0.04, f"Acc: {acc:.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.75, edgecolor="none"),
        )

        patch_neg = mpatches.Patch(color=PALETTE["class_neg"], label="Class −1")
        patch_pos = mpatches.Patch(color=PALETTE["class_pos"], label="Class +1")
        patch_bd = mpatches.Patch(color=color, label="Boundary")
        ax.legend(
            handles=[patch_neg, patch_pos, patch_bd],
            framealpha=0.25, labelcolor="white",
            facecolor=PALETTE["surface"], fontsize=8,
        )

    _apply_dark_style(fig, axes)
    fig.suptitle(
        "Final Decision Boundaries", color="white",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_perceptron_epoch(
    X: np.ndarray,
    y: np.ndarray,
    perceptron: Perceptron,
    epoch: int,
    feature_range: tuple[float, float, float, float],
) -> plt.Figure:
    """
    Visualise the Perceptron decision boundary at a specific epoch.
    Also shows the training error curve up to that epoch.
    """
    fig, (ax_main, ax_err) = plt.subplots(
        1, 2, figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    x_min, x_max, y_min, y_max = feature_range

    # ── Left: data + boundary at this epoch ──────────────
    _scatter_data(ax_main, X, y, alpha=0.6)

    m, b, acc = perceptron.boundary_history[epoch]
    _fill_decision_regions(
        ax_main, m, b, feature_range,
        PALETTE["class_neg"], PALETTE["class_pos"], alpha=0.15,
    )
    _draw_line(
        ax_main, m, b, (x_min, x_max),
        PALETTE["perceptron"],
        label=f"Epoch {epoch + 1}", lw=2.5,
    )

    # Draw all prior boundaries faintly
    for prev_epoch in range(epoch):
        pm, pb, _ = perceptron.boundary_history[prev_epoch]
        _draw_line(
            ax_main, pm, pb, (x_min, x_max),
            PALETTE["perceptron"], label="",
            lw=0.6, linestyle="--",
        )

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_title(
        f"Perceptron – Epoch {epoch + 1}  |  Train Acc: {acc:.1%}",
        fontsize=11, fontweight="bold",
    )
    ax_main.set_xlabel("Feature 1")
    ax_main.set_ylabel("Feature 2")

    patch_neg = mpatches.Patch(color=PALETTE["class_neg"], label="Class −1")
    patch_pos = mpatches.Patch(color=PALETTE["class_pos"], label="Class +1")
    patch_bd = mpatches.Patch(color=PALETTE["perceptron"], label=f"Epoch {epoch + 1}")
    ax_main.legend(
        handles=[patch_neg, patch_pos, patch_bd],
        framealpha=0.25, labelcolor="white",
        facecolor=PALETTE["surface"], fontsize=8,
    )

    # ── Right: error curve ───────────────────────────────
    all_errors = perceptron.errors_per_epoch
    epochs_x = list(range(1, len(all_errors) + 1))
    ax_err.plot(epochs_x, all_errors, color=PALETTE["neutral"], lw=1.2, alpha=0.5, label="All epochs")
    ax_err.plot(
        epochs_x[: epoch + 1], all_errors[: epoch + 1],
        color=PALETTE["perceptron"], lw=2.0, label="Up to now",
    )
    ax_err.scatter(
        [epoch + 1], [all_errors[epoch]],
        color=PALETTE["accent"], s=80, zorder=5, label="Current epoch",
    )
    ax_err.set_title("Training Errors per Epoch", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Epoch")
    ax_err.set_ylabel("# Misclassified")
    ax_err.legend(
        framealpha=0.25, labelcolor="white",
        facecolor=PALETTE["surface"], fontsize=8,
    )

    _apply_dark_style(fig, [ax_main, ax_err])
    fig.tight_layout()
    return fig


def plot_accuracy_comparison(
    perceptron: Perceptron,
    lr_model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Bar chart comparing final test accuracy of both models."""
    perc_acc = perceptron.score(X_test, y_test)
    lr_acc = lr_model.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(
        ["Perceptron", "Logistic\nRegression"],
        [perc_acc, lr_acc],
        color=[PALETTE["perceptron"], PALETTE["logistic"]],
        width=0.45,
        edgecolor="none",
    )
    for bar, val in zip(bars, [perc_acc, lr_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="white",
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold")
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig