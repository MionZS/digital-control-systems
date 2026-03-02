"""Validation utilities for SINDy models."""

from __future__ import annotations

import numpy as np


def one_step_error(identifier, data: dict) -> float:
    """Mean absolute one-step prediction error.

    Compares the model's predicted derivatives with numerically differentiated
    ground-truth data.
    """
    model = identifier.model
    if model is None:
        raise RuntimeError("Model not fitted. Call fit() first.")

    x = np.asarray(data["x"], dtype=float)
    t = data.get("t")
    u = data.get("u")

    if u is not None:
        x_dot_pred = model.predict(x, u=np.asarray(u, dtype=float))
    else:
        x_dot_pred = model.predict(x)

    if t is not None:
        x_dot_true = model.differentiate(x, t=np.asarray(t, dtype=float))
    else:
        x_dot_true = model.differentiate(x)

    return float(np.mean(np.abs(x_dot_pred - x_dot_true)))


def rollout_error(identifier, data: dict) -> float:
    """RMSE of a full rollout versus ground-truth trajectory."""
    x_true = np.asarray(data["x"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    u = data.get("u")

    x0 = x_true[0]
    x_pred = identifier.predict(x0, u, t)

    n = min(len(x_true), len(x_pred))
    return float(np.sqrt(np.mean((x_true[:n] - x_pred[:n]) ** 2)))


def plot_validation(identifier, data: dict, title: str = ""):
    """Plot ground-truth vs SINDy-predicted trajectories.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    x_true = np.asarray(data["x"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    u = data.get("u")

    x0 = x_true[0]
    x_pred = identifier.predict(x0, u, t)
    n = min(len(t), len(x_pred))

    n_states = x_true.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 3 * n_states), squeeze=False)
    for i in range(n_states):
        ax = axes[i, 0]
        ax.plot(t[:n], x_true[:n, i], label="True", linewidth=2)
        ax.plot(t[:n], x_pred[:n, i], "--", label="SINDy", linewidth=2)
        ax.set_ylabel(f"$x_{i}$")
        ax.legend()
        ax.grid(True)
    axes[-1, 0].set_xlabel("Time (s)")
    if title:
        axes[0, 0].set_title(title)
    plt.tight_layout()
    return fig
