"""Plotting utilities for simulation results and model analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np  # noqa: F401 — re-exported for notebook convenience


def plot_step_response(result, r_final: float, title: str = "", save_path: Optional[str] = None):
    """Plot output and control-input time series for a step-response simulation.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    y = result.y[:, 0] if result.y.ndim > 1 else result.y.ravel()
    u = result.u[:, 0] if result.u.ndim > 1 else result.u.ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(result.t, y, linewidth=2, label="Output")
    ax1.axhline(r_final, color="r", linestyle="--", linewidth=1.5, label="Reference")
    ax1.set_ylabel("Output")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(result.t, u, linewidth=2, color="darkorange", label="Control")
    ax2.set_ylabel("Control Input")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bode(model, title: str = "", save_path: Optional[str] = None):
    """Bode plot for an LTIModel using python-control.

    Returns the matplotlib Figure.
    """
    import control  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys = control.ss(model.A, model.B, model.C, model.D)
    fig = plt.figure(figsize=(10, 8))
    control.bode_plot(sys, plot=True)
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectories(
    results_dict: dict,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Overlay multiple SimulationResult output trajectories.

    Parameters
    ----------
    results_dict : mapping of label → SimulationResult.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, result in results_dict.items():
        y = result.y[:, 0] if result.y.ndim > 1 else result.y.ravel()
        ax.plot(result.t, y, linewidth=2, label=str(label))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid(True)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
