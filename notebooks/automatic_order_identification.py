"""Notebook - Automatic model-order selection from two LVM files.

Reads:
- control input u(t) from one .lvm
- output y(t) from another .lvm

What it does:
1. Parses the two files
2. Detects the step regions automatically
3. Builds normalized incremental step responses
4. Classifies the apparent shape:
   - first-order-like
   - second-order-like (underdamped)
   - third-order-or-higher-like
5. Fits candidate models with guarded logic
6. Selects a final model with a complexity penalty
7. Writes the result to a JSON file

Run with:
    uv run marimo edit automatic_order_identification.py
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from dataclasses import asdict, dataclass
    from pathlib import Path as PathType

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.signal import TransferFunction, cont2discrete, step

    return (
        PathType,
        TransferFunction,
        asdict,
        cont2discrete,
        dataclass,
        json,
        least_squares,
        math,
        mo,
        np,
        plt,
        step,
    )


@app.cell
def _(PathType):
    # ===== User-editable paths =====
    control_file = PathType("controle.lvm")
    output_file = PathType("saida.lvm")
    output_json = PathType("identified_model.json")

    # ===== Identification settings =====
    step_threshold = 0.5  # change detector on u[k]
    pre_samples = 15  # baseline window before each detected step
    post_ignore = 3  # ignore a few samples right after the step for baseline/steady windows
    steady_tail = 20  # steady-state averaging tail
    fit_horizon_s = 80.0  # only the first N seconds of each region are used in the fit
    min_region_len = 40  # minimum number of samples required per region

    # ===== Model selection guards =====
    rmse_improvement_guard = 0.08  # higher-order model must improve RMSE by at least 8%
    bic_margin_guard = 2.0  # and have meaningfully better BIC
    return (
        bic_margin_guard,
        control_file,
        fit_horizon_s,
        min_region_len,
        output_file,
        output_json,
        post_ignore,
        pre_samples,
        rmse_improvement_guard,
        steady_tail,
        step_threshold,
    )


@app.cell
def _(control_file, mo, output_file, output_json):
    mo.md(f"""
    # Automatic model-order identification from two LVM files

    **Files**
    - control: `{control_file}`
    - output: `{output_file}`
    - output json: `{output_json}`

    This notebook:
    - detects the step regions automatically
    - tests candidate model orders
    - chooses a final model with a guarded decision tree
    - exports the result to JSON
    """)
    return


@app.cell
def _(np):
    def read_lvm(path):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

        end_idx = None
        for i, line in enumerate(lines):
            if "***End_of_Header***" in line:
                end_idx = i
                break
        if end_idx is None:
            raise ValueError(f"Could not find LVM header end in {path}")

        rows = []
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
            raise ValueError(f"No numeric data found in {path}")

        arr = np.asarray(rows, dtype=float)
        t = arr[:, 0]
        x = arr[:, 1]
        return t, x

    def align_timebase(tu, u, ty, y):
        if len(tu) != len(ty):
            raise ValueError("Control and output files have different lengths.")
        if not np.allclose(tu, ty, atol=1e-9, rtol=0.0):
            raise ValueError("Control and output files do not share the same time base.")
        if len(tu) < 3:
            raise ValueError("Not enough samples.")
        dt = float(np.median(np.diff(tu)))
        return tu.copy(), u.copy(), y.copy(), dt

    return align_timebase, read_lvm


@app.cell
def _(dataclass):
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
        t_rel: list
        y_rel: list
        u_rel: list
        fit_mask_count: int

    @dataclass
    class CandidateFit:
        name: str
        family: str
        order: int
        n_params: int
        params: dict
        rmse: float
        sse: float
        bic: float
        aic: float
        converged: bool
        message: str
        y_hat_regions: list
        score_rank_key: tuple
        gs_num: list | None
        gs_den: list | None
        gz_num: list | None
        gz_den: list | None
        dt: float
        notes: list

    return CandidateFit, StepRegion


@app.cell
def _(StepRegion, np):
    def detect_steps(u, threshold):
        du = np.diff(u)
        idx = np.where(np.abs(du) >= threshold)[0] + 1
        return idx.astype(int)

    def build_regions(
        t, u, y, step_idx, dt, pre_samples, post_ignore, steady_tail, fit_horizon_s, min_region_len
    ):
        regions = []
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
                    u0=float(u0),
                    y0=float(y0),
                    u_ss=float(u_ss),
                    y_ss=float(y_ss),
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

    return build_regions, detect_steps


@app.cell
def _(np):
    def normalized_regions(regions, fit_horizon_s):
        out = []
        for r in regions:
            t = np.asarray(r.t_rel, dtype=float)
            y = np.asarray(r.y_rel, dtype=float)
            mask = t <= fit_horizon_s
            if np.sum(mask) < 8:
                mask = np.ones_like(t, dtype=bool)
            yn = y[mask] / r.du
            out.append((t[mask], yn))
        return out

    def region_features(regions):
        gains = np.asarray([r.gain_section for r in regions], dtype=float)
        gain_mean = float(np.mean(gains))
        gain_std = float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0

        overshoots = []
        osc_counts = []
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

    return normalized_regions, region_features


@app.cell
def _(TransferFunction, math, np, step):
    def step_first_order(t, K, tau, delay):
        tau = max(float(tau), 1e-9)
        delay = max(float(delay), 0.0)
        y = np.zeros_like(t, dtype=float)
        m = t >= delay
        te = t[m] - delay
        y[m] = K * (1.0 - np.exp(-te / tau))
        return y

    def step_second_order_underdamped(t, K, zeta, wn, delay):
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

    def _third_order_tf(K, tau1, tau2, tau3):
        tau1 = max(float(tau1), 1e-6)
        tau2 = max(float(tau2), 1e-6)
        tau3 = max(float(tau3), 1e-6)
        den = np.polymul([tau1, 1.0], np.polymul([tau2, 1.0], [tau3, 1.0]))
        num = [float(K)]
        return TransferFunction(num, den), num, den.tolist()

    def step_third_order_real_poles(t, K, tau1, tau2, tau3, delay):
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

    return (
        step_first_order,
        step_second_order_underdamped,
        step_third_order_real_poles,
    )


@app.cell
def _(cont2discrete, np):
    def discrete_from_continuous(num, den, dt):
        numd, dend, _ = cont2discrete((num, den), dt=dt, method="zoh")
        numd = np.squeeze(numd).astype(float).tolist()
        dend = np.squeeze(dend).astype(float).tolist()
        return numd, dend

    def first_order_tf(K, tau):
        return [float(K)], [float(tau), 1.0]

    def second_order_ud_tf(K, zeta, wn):
        num = [float(K) * float(wn) ** 2]
        den = [1.0, 2.0 * float(zeta) * float(wn), float(wn) ** 2]
        return num, den

    def third_order_real_tf(K, tau1, tau2, tau3):
        tau1 = max(float(tau1), 1e-6)
        tau2 = max(float(tau2), 1e-6)
        tau3 = max(float(tau3), 1e-6)
        den = np.polymul([tau1, 1.0], np.polymul([tau2, 1.0], [tau3, 1.0]))
        num = [float(K)]
        return num, den.tolist()

    return (
        discrete_from_continuous,
        first_order_tf,
        second_order_ud_tf,
        third_order_real_tf,
    )


@app.cell
def _(
    CandidateFit,
    discrete_from_continuous,
    first_order_tf,
    least_squares,
    np,
    second_order_ud_tf,
    step_first_order,
    step_second_order_underdamped,
    step_third_order_real_poles,
    third_order_real_tf,
):
    def concat_residuals(regions_norm, model_name, p):
        residuals = []
        preds = []
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

    def score_from_residuals(residuals, n_params):
        n = int(len(residuals))
        sse = float(np.sum(residuals**2))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        sse_safe = max(sse, 1e-18)
        aic = float(n * np.log(sse_safe / n) + 2 * n_params)
        bic = float(n * np.log(sse_safe / n) + n_params * np.log(n))
        return sse, rmse, aic, bic

    def fit_candidate(name, regions_norm, dt):
        notes = []
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

        def fun(p):
            res, _ = concat_residuals(regions_norm, name, p)
            return res

        try:
            res = least_squares(fun, p0, bounds=(lb, ub), max_nfev=20000)
            residuals, preds = concat_residuals(regions_norm, name, res.x)
            sse, rmse, aic, bic = score_from_residuals(residuals, n_params)

            if name == "first_order":
                K, tau, delay = [float(v) for v in res.x]
                params = {"K": K, "tau": tau, "delay": delay}
                gs_num, gs_den = first_order_tf(K, tau)
            elif name == "second_order_underdamped":
                K, zeta, wn, delay = [float(v) for v in res.x]
                params = {"K": K, "zeta": zeta, "omega_n": wn, "delay": delay}
                gs_num, gs_den = second_order_ud_tf(K, zeta, wn)
                if zeta > 0.97:
                    notes.append("The fitted second-order model is very close to critical damping.")
            else:
                K, tau1, tau2, tau3, delay = [float(v) for v in res.x]
                params = {"K": K, "tau1": tau1, "tau2": tau2, "tau3": tau3, "delay": delay}
                gs_num, gs_den = third_order_real_tf(K, tau1, tau2, tau3)

            gz_num, gz_den = discrete_from_continuous(gs_num, gs_den, dt)

            return CandidateFit(
                name=name,
                family=family,
                order={
                    "first_order": 1,
                    "second_order_underdamped": 2,
                    "third_order_real_poles": 3,
                }[name],
                n_params=n_params,
                params=params,
                rmse=rmse,
                sse=sse,
                bic=bic,
                aic=aic,
                converged=bool(res.success),
                message=str(res.message),
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
                order={
                    "first_order": 1,
                    "second_order_underdamped": 2,
                    "third_order_real_poles": 3,
                }[name],
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

    return (fit_candidate,)


@app.cell
def _():
    def choose_model(candidates, apparent_family, rmse_improvement_guard, bic_margin_guard):
        valid = [c for c in candidates if c.converged and c.rmse < float("inf")]
        if not valid:
            return None, {"reason": "No candidate converged."}

        by_bic = sorted(valid, key=lambda c: c.score_rank_key)
        best = by_bic[0]

        # Guard against unnecessary complexity:
        # prefer a lower-order model if the higher-order winner does not improve enough.
        simpler = [c for c in valid if c.order < best.order]
        decision_notes = [f"Apparent family: {apparent_family}"]

        for s in sorted(simpler, key=lambda c: c.order):
            rmse_improvement = (s.rmse - best.rmse) / max(s.rmse, 1e-12)
            bic_advantage = s.bic - best.bic
            if rmse_improvement < rmse_improvement_guard or bic_advantage < bic_margin_guard:
                # If the more complex model does not clearly earn its complexity, keep the simpler one.
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

    return (choose_model,)


@app.cell
def _(
    align_timebase,
    bic_margin_guard,
    build_regions,
    choose_model,
    control_file,
    detect_steps,
    fit_candidate,
    fit_horizon_s,
    min_region_len,
    mo,
    normalized_regions,
    output_file,
    post_ignore,
    pre_samples,
    read_lvm,
    region_features,
    rmse_improvement_guard,
    steady_tail,
    step_threshold,
):
    result_bundle = None
    message = ""

    if not control_file.exists() or not output_file.exists():
        message = f"Input file not found.\n\n- control: `{control_file}`\n- output: `{output_file}`"
    else:
        t_u, u = read_lvm(control_file)
        t_y, y = read_lvm(output_file)
        t, u, y, dt = align_timebase(t_u, u, t_y, y)

        step_idx = detect_steps(u, threshold=step_threshold)
        regions = build_regions(
            t=t,
            u=u,
            y=y,
            step_idx=step_idx,
            dt=dt,
            pre_samples=pre_samples,
            post_ignore=post_ignore,
            steady_tail=steady_tail,
            fit_horizon_s=fit_horizon_s,
            min_region_len=min_region_len,
        )
        regions_norm = normalized_regions(regions, fit_horizon_s=fit_horizon_s)
        features = region_features(regions)

        candidates = []
        for name in features["candidate_order"]:
            candidates.append(fit_candidate(name, regions_norm, dt))

        # Evaluate the remaining candidate too, so the JSON always stores all three
        for name in ["first_order", "second_order_underdamped", "third_order_real_poles"]:
            if name not in [c.name for c in candidates]:
                candidates.append(fit_candidate(name, regions_norm, dt))

        selected, selection_info = choose_model(
            candidates=candidates,
            apparent_family=features["apparent_family"],
            rmse_improvement_guard=rmse_improvement_guard,
            bic_margin_guard=bic_margin_guard,
        )

        result_bundle = {
            "t": t,
            "u": u,
            "y": y,
            "dt": dt,
            "step_idx": step_idx,
            "regions": regions,
            "features": features,
            "candidates": candidates,
            "selected": selected,
            "selection_info": selection_info,
        }

        if selected is None:
            message = "No model converged."
        else:
            message = (
                f"## Selected model\n\n"
                f"- apparent family: `{features['apparent_family']}`\n"
                f"- selected: `{selected.name}`\n"
                f"- order: `{selected.order}`\n"
                f"- RMSE: `{selected.rmse:.6f}`\n"
                f"- BIC: `{selected.bic:.3f}`\n"
                f"- parameters: `{selected.params}`"
            )

    mo.md(message)
    return (result_bundle,)


@app.cell
def _(mo, result_bundle):
    if result_bundle is None:
        mo.md("No result available.")
    else:
        regions = result_bundle["regions"]
        features = result_bundle["features"]
        candidates = sorted(result_bundle["candidates"], key=lambda c: c.score_rank_key)

        lines = []
        lines.append("## Step regions")
        for r in regions:
            lines.append(
                f"- region {r.index}: t=[{r.t0:.3f}, {r.t1:.3f}] s, Δu={r.du:.6f}, Δy={r.dy:.6f}, K_section={r.gain_section:.6f}"
            )

        lines.append("\n## Shape features")
        for k, v in features.items():
            lines.append(f"- {k}: `{v}`")

        lines.append("\n## Candidate ranking")
        for c in candidates:
            lines.append(
                f"- {c.name}: order={c.order}, RMSE={c.rmse:.6f}, BIC={c.bic:.3f}, converged={c.converged}"
            )

        mo.md("\n".join(lines))
    return


@app.cell
def _(np, plt, result_bundle):
    fig = None

    if result_bundle is not None:
        t = result_bundle["t"]
        u = result_bundle["u"]
        y = result_bundle["y"]
        step_idx = result_bundle["step_idx"]
        regions = result_bundle["regions"]
        selected = result_bundle["selected"]

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
                # predicted signal is incremental and normalized by Δu
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
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(asdict, json, output_json, result_bundle):
    if result_bundle is not None:
        selected = result_bundle["selected"]
        payload = {
            "dt": float(result_bundle["dt"]),
            "step_indices": [int(v) for v in result_bundle["step_idx"]],
            "shape_features": result_bundle["features"],
            "selection_info": result_bundle["selection_info"],
            "selected_model": None
            if selected is None
            else {
                "name": selected.name,
                "family": selected.family,
                "order": selected.order,
                "params": selected.params,
                "rmse": selected.rmse,
                "bic": selected.bic,
                "aic": selected.aic,
                "gs_num": selected.gs_num,
                "gs_den": selected.gs_den,
                "gz_num": selected.gz_num,
                "gz_den": selected.gz_den,
                "notes": selected.notes,
            },
            "candidate_models": [
                {
                    "name": c.name,
                    "family": c.family,
                    "order": c.order,
                    "params": c.params,
                    "rmse": c.rmse,
                    "sse": c.sse,
                    "bic": c.bic,
                    "aic": c.aic,
                    "converged": c.converged,
                    "message": c.message,
                    "gs_num": c.gs_num,
                    "gs_den": c.gs_den,
                    "gz_num": c.gz_num,
                    "gz_den": c.gz_den,
                    "notes": c.notes,
                }
                for c in sorted(result_bundle["candidates"], key=lambda c: c.score_rank_key)
            ],
            "regions": [asdict(r) for r in result_bundle["regions"]],
        }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return


if __name__ == "__main__":
    app.run()
