"""Standalone automatic model-order identification from two LVM files.

This script is a pure Python fallback for the Marimo notebook workflow.

Example:
    uv run python Scripts/automatic_order_identification.py \
        --control input/degrausMotorTQ/controle.lvm \
        --output input/degrausMotorTQ/saida.lvm \
        --out-json input/impulse_response/identified_model.json \
        --out-plot input/impulse_response/identified_model_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import TransferFunction, cont2discrete, lfilter, step


@dataclass
class StepRegion:
    index: int
    step_sample: int
    t0: float
    t1: float
    u0: float
    y0: float
    u_ss: float
    y_ss: float
    du: float
    dy: float
    gain_section: float
    t_rel: list[float]
    y_rel: list[float]
    u_rel: list[float]
    fit_mask_count: int


@dataclass
class CandidateFit:
    name: str
    family: str
    order: int
    n_params: int
    params: dict[str, float]
    rmse: float
    sse: float
    bic: float
    aic: float
    converged: bool
    message: str
    y_hat_regions: list[list[float]]
    score_rank_key: tuple[float, float, int]
    gs_num: list[float] | None
    gs_den: list[float] | None
    gz_num: list[float] | None
    gz_den: list[float] | None
    dt: float
    notes: list[str]


class FeatureSummary(TypedDict):
    gain_mean: float
    gain_std: float
    overshoot_mean: float
    oscillation_score: float
    apparent_family: str
    candidate_order: list[str]


def read_lvm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if "***End_of_Header***" in line:
            end_idx = i
            break
    if end_idx is None:
        msg = f"Could not find LVM header end in {path}"
        raise ValueError(msg)

    rows: list[list[float]] = []
    for line in lines[end_idx + 1 :]:
        parts = [p for p in line.strip().split("\t") if p != ""]
        if len(parts) < 2:
            continue
        try:
            vals = [float(p.replace(",", ".")) for p in parts[:2]]
            rows.append(vals)
        except ValueError:
            continue

    if not rows:
        msg = f"No numeric data found in {path}"
        raise ValueError(msg)

    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0]
    x = arr[:, 1]
    return t, x


def align_timebase(
    tu: np.ndarray,
    u: np.ndarray,
    ty: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(tu) != len(ty):
        raise ValueError("Control and output files have different lengths.")
    if not np.allclose(tu, ty, atol=1e-9, rtol=0.0):
        raise ValueError("Control and output files do not share the same time base.")
    if len(tu) < 3:
        raise ValueError("Not enough samples.")
    dt = float(np.median(np.diff(tu)))
    return tu.copy(), u.copy(), y.copy(), dt


def detect_steps(u: np.ndarray, threshold: float) -> np.ndarray:
    du = np.diff(u)
    idx = np.where(np.abs(du) >= threshold)[0] + 1
    return idx.astype(int)


def build_regions(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    step_idx: np.ndarray,
    pre_samples: int,
    post_ignore: int,
    steady_tail: int,
    fit_horizon_s: float,
    min_region_len: int,
) -> list[StepRegion]:
    regions: list[StepRegion] = []
    n = len(t)
    for j, idx in enumerate(step_idx):
        end = int(step_idx[j + 1]) if j + 1 < len(step_idx) else n
        if end - idx < min_region_len:
            continue

        pre0 = max(0, idx - pre_samples)
        pre1 = idx
        post0 = min(end, idx + post_ignore)
        tail0 = max(post0, end - steady_tail)
        tail1 = end

        u0 = float(np.mean(u[pre0:pre1]))
        y0 = float(np.mean(y[pre0:pre1]))
        u_ss = float(np.mean(u[tail0:tail1]))
        y_ss = float(np.mean(y[tail0:tail1]))

        du_step = u_ss - u0
        dy_step = y_ss - y0

        if abs(du_step) < 1e-12:
            continue

        t_rel = t[idx:end] - t[idx]
        u_rel = u[idx:end] - u0
        y_rel = y[idx:end] - y0

        fit_mask = t_rel <= fit_horizon_s
        if int(np.sum(fit_mask)) < min_region_len:
            fit_mask = np.ones_like(t_rel, dtype=bool)

        gain_section = dy_step / du_step
        regions.append(
            StepRegion(
                index=len(regions) + 1,
                step_sample=int(idx),
                t0=float(t[idx]),
                t1=float(t[end - 1]),
                u0=u0,
                y0=y0,
                u_ss=u_ss,
                y_ss=y_ss,
                du=float(du_step),
                dy=float(dy_step),
                gain_section=float(gain_section),
                t_rel=t_rel.tolist(),
                y_rel=y_rel.tolist(),
                u_rel=u_rel.tolist(),
                fit_mask_count=int(np.sum(fit_mask)),
            )
        )
    return regions


def normalized_regions(
    regions: list[StepRegion], fit_horizon_s: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for r in regions:
        t = np.asarray(r.t_rel, dtype=float)
        y = np.asarray(r.y_rel, dtype=float)
        mask = t <= fit_horizon_s
        if np.sum(mask) < 8:
            mask = np.ones_like(t, dtype=bool)
        yn = y[mask] / r.du
        out.append((t[mask], yn))
    return out


def region_features(regions: list[StepRegion]) -> FeatureSummary:
    gains = np.asarray([r.gain_section for r in regions], dtype=float)
    gain_mean = float(np.mean(gains))
    gain_std = float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0

    overshoots: list[float] = []
    osc_counts: list[int] = []
    for r in regions:
        yn = np.asarray(r.y_rel, dtype=float) / r.du
        final = float(r.gain_section)
        if abs(final) < 1e-9:
            overshoots.append(0.0)
            osc_counts.append(0)
            continue

        peak = float(np.max(yn))
        overshoot = max(0.0, (peak - final) / max(abs(final), 1e-9))
        overshoots.append(float(overshoot))

        dy = np.diff(yn)
        if len(dy) < 5:
            osc_counts.append(0)
        else:
            s = np.sign(dy)
            s[s == 0] = 1
            changes = int(np.sum(s[1:] * s[:-1] < 0))
            osc_counts.append(changes)

    overshoot_mean = float(np.mean(overshoots)) if overshoots else 0.0
    osc_mean = float(np.mean(osc_counts)) if osc_counts else 0.0

    if overshoot_mean < 0.02 and osc_mean < 1.0:
        family = "first_order_like"
        candidates = ["first_order", "second_order_underdamped", "third_order_real_poles"]
    elif overshoot_mean < 0.15 and osc_mean < 3.0:
        family = "second_order_like"
        candidates = ["second_order_underdamped", "third_order_real_poles", "first_order"]
    else:
        family = "third_or_higher_like"
        candidates = ["third_order_real_poles", "second_order_underdamped"]

    return {
        "gain_mean": gain_mean,
        "gain_std": gain_std,
        "overshoot_mean": overshoot_mean,
        "oscillation_score": osc_mean,
        "apparent_family": family,
        "candidate_order": candidates,
    }


def step_first_order(t: np.ndarray, K: float, tau: float, delay: float) -> np.ndarray:
    tau = max(float(tau), 1e-9)
    delay = max(float(delay), 0.0)
    y = np.zeros_like(t, dtype=float)
    m = t >= delay
    te = t[m] - delay
    y[m] = K * (1.0 - np.exp(-te / tau))
    return y


def step_second_order_underdamped(
    t: np.ndarray,
    K: float,
    zeta: float,
    wn: float,
    delay: float,
) -> np.ndarray:
    zeta = float(np.clip(zeta, 1e-6, 0.999999))
    wn = max(float(wn), 1e-6)
    delay = max(float(delay), 0.0)

    y = np.zeros_like(t, dtype=float)
    m = t >= delay
    te = t[m] - delay

    wd = wn * math.sqrt(max(1e-12, 1.0 - zeta**2))
    phi = math.atan2(math.sqrt(max(1e-12, 1.0 - zeta**2)), zeta)
    y[m] = K * (
        1.0
        - (1.0 / math.sqrt(max(1e-12, 1.0 - zeta**2)))
        * np.exp(-zeta * wn * te)
        * np.sin(wd * te + phi)
    )
    return y


def _third_order_tf(K: float, tau1: float, tau2: float, tau3: float):
    tau1 = max(float(tau1), 1e-6)
    tau2 = max(float(tau2), 1e-6)
    tau3 = max(float(tau3), 1e-6)
    den = np.polymul([tau1, 1.0], np.polymul([tau2, 1.0], [tau3, 1.0]))
    num = [float(K)]
    return TransferFunction(num, den), num, den.tolist()


def step_third_order_real_poles(
    t: np.ndarray,
    K: float,
    tau1: float,
    tau2: float,
    tau3: float,
    delay: float,
) -> np.ndarray:
    delay = max(float(delay), 0.0)
    y = np.zeros_like(t, dtype=float)
    m = t >= delay
    if not np.any(m):
        return y
    te = t[m] - delay
    sys, _, _ = _third_order_tf(K, tau1, tau2, tau3)
    _, y_step = step(sys, T=te)
    y[m] = y_step
    return y


def discrete_from_continuous(
    num: list[float], den: list[float], dt: float
) -> tuple[list[float], list[float]]:
    # scipy.signal.cont2discrete has multiple overloads (2, 3, 4 or 5 items);
    # we only need numerator and denominator at index 0 and 1.
    result = cont2discrete((num, den), dt=dt, method="zoh")
    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError("Unexpected cont2discrete return format")

    numd = np.asarray(result[0], dtype=float)
    dend = np.asarray(result[1], dtype=float)

    numd_list = np.squeeze(numd).astype(float).tolist()
    dend_list = np.squeeze(dend).astype(float).tolist()
    return numd_list, dend_list


def first_order_tf(K: float, tau: float) -> tuple[list[float], list[float]]:
    return [float(K)], [float(tau), 1.0]


def second_order_ud_tf(K: float, zeta: float, wn: float) -> tuple[list[float], list[float]]:
    num = [float(K) * float(wn) ** 2]
    den = [1.0, 2.0 * float(zeta) * float(wn), float(wn) ** 2]
    return num, den


def third_order_real_tf(
    K: float, tau1: float, tau2: float, tau3: float
) -> tuple[list[float], list[float]]:
    _, num, den = _third_order_tf(K, tau1, tau2, tau3)
    return num, den


def concat_residuals(
    regions_norm: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    p: np.ndarray,
) -> tuple[np.ndarray, list[list[float]]]:
    residuals: list[np.ndarray] = []
    preds: list[list[float]] = []
    for t, y in regions_norm:
        if model_name == "first_order":
            y_hat = step_first_order(t, *p)
        elif model_name == "second_order_underdamped":
            y_hat = step_second_order_underdamped(t, *p)
        elif model_name == "third_order_real_poles":
            y_hat = step_third_order_real_poles(t, *p)
        else:
            raise ValueError(f"Unknown model {model_name}")
        residuals.append(y_hat - y)
        preds.append(y_hat.tolist())
    return np.concatenate(residuals), preds


def score_from_residuals(residuals: np.ndarray, n_params: int) -> tuple[float, float, float, float]:
    n = int(len(residuals))
    sse = float(np.sum(residuals**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    sse_safe = max(sse, 1e-18)
    aic = float(n * np.log(sse_safe / n) + 2 * n_params)
    bic = float(n * np.log(sse_safe / n) + n_params * np.log(n))
    return sse, rmse, aic, bic


def fit_candidate(
    name: str, regions_norm: list[tuple[np.ndarray, np.ndarray]], dt: float
) -> CandidateFit:
    notes: list[str] = []
    if name == "first_order":
        p0 = np.array([1.0, 10.0, 0.0], dtype=float)
        lb = np.array([0.0, 1e-4, 0.0], dtype=float)
        ub = np.array([20.0, 1e5, 10.0], dtype=float)
        n_params = 3
        family = "monotonic"
    elif name == "second_order_underdamped":
        p0 = np.array([1.0, 0.7, 0.2, 0.0], dtype=float)
        lb = np.array([0.0, 1e-4, 1e-4, 0.0], dtype=float)
        ub = np.array([20.0, 0.999, 50.0, 10.0], dtype=float)
        n_params = 4
        family = "underdamped"
    elif name == "third_order_real_poles":
        p0 = np.array([1.0, 4.0, 4.0, 4.0, 0.0], dtype=float)
        lb = np.array([0.0, 1e-4, 1e-4, 1e-4, 0.0], dtype=float)
        ub = np.array([20.0, 1e4, 1e4, 1e4, 10.0], dtype=float)
        n_params = 5
        family = "higher_order_real_poles"
    else:
        raise ValueError(name)

    def objective(p):
        res, _ = concat_residuals(regions_norm, name, p)
        return res

    try:
        result = least_squares(objective, p0, bounds=(lb, ub), max_nfev=20000)
        residuals, preds = concat_residuals(regions_norm, name, result.x)
        sse, rmse, aic, bic = score_from_residuals(residuals, n_params)

        if name == "first_order":
            K, tau, delay = [float(v) for v in result.x]
            params = {"K": K, "tau": tau, "delay": delay}
            gs_num, gs_den = first_order_tf(K, tau)
        elif name == "second_order_underdamped":
            K, zeta, wn, delay = [float(v) for v in result.x]
            params = {"K": K, "zeta": zeta, "omega_n": wn, "delay": delay}
            gs_num, gs_den = second_order_ud_tf(K, zeta, wn)
            if zeta > 0.97:
                notes.append("The fitted second-order model is very close to critical damping.")
        else:
            K, tau1, tau2, tau3, delay = [float(v) for v in result.x]
            params = {"K": K, "tau1": tau1, "tau2": tau2, "tau3": tau3, "delay": delay}
            gs_num, gs_den = third_order_real_tf(K, tau1, tau2, tau3)

        gz_num, gz_den = discrete_from_continuous(gs_num, gs_den, dt)

        return CandidateFit(
            name=name,
            family=family,
            order={"first_order": 1, "second_order_underdamped": 2, "third_order_real_poles": 3}[
                name
            ],
            n_params=n_params,
            params=params,
            rmse=rmse,
            sse=sse,
            bic=bic,
            aic=aic,
            converged=bool(result.success),
            message=str(result.message),
            y_hat_regions=preds,
            score_rank_key=(bic, rmse, n_params),
            gs_num=[float(v) for v in gs_num],
            gs_den=[float(v) for v in gs_den],
            gz_num=[float(v) for v in gz_num],
            gz_den=[float(v) for v in gz_den],
            dt=float(dt),
            notes=notes,
        )
    except Exception as exc:
        return CandidateFit(
            name=name,
            family=family,
            order={"first_order": 1, "second_order_underdamped": 2, "third_order_real_poles": 3}[
                name
            ],
            n_params=n_params,
            params={},
            rmse=float("inf"),
            sse=float("inf"),
            bic=float("inf"),
            aic=float("inf"),
            converged=False,
            message=f"Fit failed: {exc}",
            y_hat_regions=[],
            score_rank_key=(float("inf"), float("inf"), n_params),
            gs_num=None,
            gs_den=None,
            gz_num=None,
            gz_den=None,
            dt=float(dt),
            notes=["Fit failed."],
        )


def choose_model(
    candidates: list[CandidateFit],
    apparent_family: str,
    rmse_improvement_guard: float,
    bic_margin_guard: float,
):
    valid = [c for c in candidates if c.converged and c.rmse < float("inf")]
    if not valid:
        return None, {"reason": "No candidate converged."}

    by_bic = sorted(valid, key=lambda c: c.score_rank_key)
    best = by_bic[0]

    simpler = [c for c in valid if c.order < best.order]
    decision_notes = [f"Apparent family: {apparent_family}"]

    for s in sorted(simpler, key=lambda c: c.order):
        rmse_improvement = (s.rmse - best.rmse) / max(s.rmse, 1e-12)
        bic_advantage = s.bic - best.bic
        if rmse_improvement < rmse_improvement_guard or bic_advantage < bic_margin_guard:
            if s.bic <= best.bic + bic_margin_guard:
                decision_notes.append(
                    f"Selected simpler model {s.name} because the more complex alternative did not"
                    f" clearly improve RMSE/BIC."
                )
                return s, {
                    "reason": "complexity_guard",
                    "decision_notes": decision_notes,
                }

    decision_notes.append(f"Selected {best.name} as the best-scoring candidate.")
    return best, {"reason": "best_score", "decision_notes": decision_notes}


def _poly_to_text(coeffs: list[float], var: str) -> str:
    parts: list[str] = []
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        power = n - i - 1
        c_str = f"{c:.6g}"
        if power == 0:
            term = c_str
        elif power == 1:
            term = f"{c_str}{var}"
        else:
            term = f"{c_str}{var}^{power}"
        parts.append(term)
    return " + ".join(parts)


def tf_to_text(num: list[float], den: list[float], var: str) -> str:
    return f"({_poly_to_text(num, var)}) / ({_poly_to_text(den, var)})"


def save_plot(
    out_plot: Path,
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    step_idx: np.ndarray,
    regions: list[StepRegion],
    selected: CandidateFit | None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    axes[0].plot(t, u, linewidth=1.2)
    axes[0].set_title("Control input u(t)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)
    for idx in step_idx:
        axes[0].axvline(t[idx], linestyle="--", linewidth=0.8)

    axes[1].plot(t, y, linewidth=1.2, label="measured y")
    if selected is not None:
        y_hat_full = np.full_like(y, np.nan, dtype=float)
        for r, yhat in zip(regions, selected.y_hat_regions):
            idx0 = r.step_sample
            idx1 = idx0 + len(yhat)
            y_hat_full[idx0:idx1] = r.y0 + r.du * np.asarray(yhat, dtype=float)
        axes[1].plot(
            t, y_hat_full, linewidth=1.2, linestyle="--", label=f"selected: {selected.name}"
        )

    axes[1].set_title("Measured output and selected-model reconstruction")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for r in regions:
        tr = np.asarray(r.t_rel, dtype=float)
        yr = np.asarray(r.y_rel, dtype=float) / r.du
        axes[2].plot(tr, yr, linewidth=1.0, alpha=0.8, label=f"region {r.index}")
    if selected is not None and selected.y_hat_regions:
        axes[2].plot(
            np.asarray(regions[0].t_rel[: len(selected.y_hat_regions[0])], dtype=float),
            np.asarray(selected.y_hat_regions[0], dtype=float),
            linewidth=2.0,
            linestyle="--",
            label="selected normalized fit",
        )

    axes[2].set_title("Normalized incremental step responses")
    axes[2].set_xlabel("Time since step [s]")
    axes[2].set_ylabel("Δy / Δu")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)


def simulate_output_from_input(
    selected: CandidateFit | None,
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    step_idx: np.ndarray,
) -> np.ndarray | None:
    if selected is None or selected.gz_num is None or selected.gz_den is None:
        return None

    dt = float(np.median(np.diff(t)))
    if len(step_idx) > 0:
        pre_n = max(1, int(step_idx[0]))
    else:
        pre_n = min(20, len(t))

    u0 = float(np.mean(u[:pre_n]))
    y0 = float(np.mean(y[:pre_n]))

    u_rel = np.asarray(u - u0, dtype=float)
    delay_s = float(selected.params.get("delay", 0.0)) if selected.params else 0.0
    delay_samples = max(0, int(round(delay_s / dt)))

    if delay_samples > 0:
        u_rel_delayed = np.zeros_like(u_rel)
        u_rel_delayed[delay_samples:] = u_rel[:-delay_samples]
    else:
        u_rel_delayed = u_rel

    b = np.asarray(selected.gz_num, dtype=float)
    a = np.asarray(selected.gz_den, dtype=float)
    # scipy lfilter typing can be ndarray or tuple depending on overloads; normalize to ndarray.
    y_rel_sim_raw = lfilter(b, a, u_rel_delayed)
    if isinstance(y_rel_sim_raw, tuple):
        y_rel_sim = np.asarray(y_rel_sim_raw[0], dtype=float)
    else:
        y_rel_sim = np.asarray(y_rel_sim_raw, dtype=float)
    return np.asarray(y0 + y_rel_sim, dtype=float)


def save_overlay_plots(
    out_plot: Path,
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    y_sim: np.ndarray | None,
) -> Path:
    overlay_path = out_plot.with_name(f"{out_plot.stem}_control_overlays{out_plot.suffix}")

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    # Top: control + measured output on SAME axis (no twinx)
    ax_top.plot(t, u, color="tab:blue", linewidth=1.2, label="control u(t)")
    ax_top.plot(t, y, color="tab:red", linewidth=1.2, label="measured y(t)")
    ax_top.set_title("Overlay 1: control input and measured output")
    ax_top.set_xlabel("Time [s]")
    ax_top.set_ylabel("Value")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")

    # Bottom: control + simulated model response on SAME axis (no twinx)
    ax_bottom.plot(t, u, color="tab:blue", linewidth=1.2, label="control u(t)")
    if y_sim is not None:
        ax_bottom.plot(
            t,
            y_sim,
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            label="simulated y_model(t)",
        )
    ax_bottom.set_title("Overlay 2: control input and model simulated response")
    ax_bottom.set_xlabel("Time [s]")
    ax_bottom.set_ylabel("Value")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="best")

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(overlay_path, dpi=140)
    plt.close(fig)
    return overlay_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic model-order identification from two LVM files"
    )
    parser.add_argument("--control", type=Path, required=True, help="Path to control LVM file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output LVM file")
    parser.add_argument("--out-json", type=Path, required=True, help="Where to write JSON result")
    parser.add_argument("--out-plot", type=Path, default=None, help="Optional PNG diagnostics plot")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.1,
        help=(
            "Scale factor applied to time column to convert to seconds. "
            "Use 0.1 for deciseconds (10 samples per second), 1.0 if already in seconds."
        ),
    )
    parser.add_argument("--step-threshold", type=float, default=0.5)
    parser.add_argument("--pre-samples", type=int, default=15)
    parser.add_argument("--post-ignore", type=int, default=3)
    parser.add_argument("--steady-tail", type=int, default=20)
    parser.add_argument("--fit-horizon-s", type=float, default=80.0)
    parser.add_argument("--min-region-len", type=int, default=40)
    parser.add_argument("--rmse-improvement-guard", type=float, default=0.08)
    parser.add_argument("--bic-margin-guard", type=float, default=2.0)
    args = parser.parse_args()

    if not args.control.exists() or not args.output.exists():
        print(f"Input file not found. control={args.control} output={args.output}")
        return 1

    t_u, u = read_lvm(args.control)
    t_y, y = read_lvm(args.output)

    # Convert raw time units to seconds before all processing.
    t_u = np.asarray(t_u, dtype=float) * float(args.time_scale)
    t_y = np.asarray(t_y, dtype=float) * float(args.time_scale)

    t, u, y, dt = align_timebase(t_u, u, t_y, y)

    step_idx = detect_steps(u, threshold=args.step_threshold)
    regions = build_regions(
        t=t,
        u=u,
        y=y,
        step_idx=step_idx,
        pre_samples=args.pre_samples,
        post_ignore=args.post_ignore,
        steady_tail=args.steady_tail,
        fit_horizon_s=args.fit_horizon_s,
        min_region_len=args.min_region_len,
    )

    if not regions:
        print("No valid step regions detected.")
        return 2

    regions_norm = normalized_regions(regions, fit_horizon_s=args.fit_horizon_s)
    features = region_features(regions)

    candidates: list[CandidateFit] = []
    for name in features["candidate_order"]:
        candidates.append(fit_candidate(name, regions_norm, dt))

    for name in ["first_order", "second_order_underdamped", "third_order_real_poles"]:
        if name not in [c.name for c in candidates]:
            candidates.append(fit_candidate(name, regions_norm, dt))

    selected, selection_info = choose_model(
        candidates=candidates,
        apparent_family=str(features["apparent_family"]),
        rmse_improvement_guard=args.rmse_improvement_guard,
        bic_margin_guard=args.bic_margin_guard,
    )

    payload = {
        "input": {
            "control_file": str(args.control),
            "output_file": str(args.output),
            "time_scale_to_seconds": float(args.time_scale),
            "sample_time_s": dt,
            "detected_steps": int(len(step_idx)),
            "valid_regions": int(len(regions)),
        },
        "selection": {
            "apparent_family": features["apparent_family"],
            "reason": selection_info.get("reason") if isinstance(selection_info, dict) else None,
            "decision_notes": selection_info.get("decision_notes", [])
            if isinstance(selection_info, dict)
            else [],
        },
        "model": None
        if selected is None
        else {
            "name": selected.name,
            "family": selected.family,
            "order": selected.order,
            "parameters": selected.params,
            "fit": {
                "rmse": selected.rmse,
                "bic": selected.bic,
                "aic": selected.aic,
            },
            "continuous_s_domain": {
                "numerator": selected.gs_num,
                "denominator": selected.gs_den,
                "transfer_function": tf_to_text(selected.gs_num or [], selected.gs_den or [], "s")
                if selected.gs_num and selected.gs_den
                else None,
            },
            "discrete_z_domain": {
                "method": "zoh",
                "numerator": selected.gz_num,
                "denominator": selected.gz_den,
                "transfer_function": tf_to_text(selected.gz_num or [], selected.gz_den or [], "z")
                if selected.gz_num and selected.gz_den
                else None,
            },
            "notes": selected.notes,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    y_sim = simulate_output_from_input(selected, t, u, y, step_idx)

    if args.out_plot is not None:
        save_plot(args.out_plot, t, u, y, step_idx, regions, selected)
        overlay_file = save_overlay_plots(args.out_plot, t, u, y, y_sim)

    if selected is None:
        print("No candidate converged.")
        return 3

    print("Selected model")
    print(f"- family: {features['apparent_family']}")
    print(f"- name: {selected.name}")
    print(f"- order: {selected.order}")
    print(f"- rmse: {selected.rmse:.6f}")
    print(f"- bic: {selected.bic:.3f}")
    print(f"- out_json: {args.out_json}")
    if args.out_plot is not None:
        print(f"- out_plot: {args.out_plot}")
        print(f"- overlay_subplots: {overlay_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
