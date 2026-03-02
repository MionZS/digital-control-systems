"""scipy-based simulation backend."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from control_lab.models.lti import LTIModel
from control_lab.models.nonlinear import NonlinearModel
from control_lab.sim.common import SimulationResult


class ControlBackend:
    """Simulation backend using ``scipy.integrate.solve_ivp``.

    For LTI discrete-time models the state update is algebraic.
    For LTI continuous-time and nonlinear models, RK45 integration is used
    over each sample interval (zero-order hold on the control input).
    """

    def simulate(
        self,
        model: LTIModel | NonlinearModel,
        controller: object,
        x0: np.ndarray,
        r_func,
        t_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        t0, tf = t_span
        t_grid = np.arange(t0, tf + dt * 0.5, dt)
        n_steps = len(t_grid)

        n_states = model.n_states
        n_inputs = model.n_inputs
        n_outputs = model.n_outputs

        x_hist = np.zeros((n_steps, n_states))
        y_hist = np.zeros((n_steps, n_outputs))
        u_hist = np.zeros((n_steps, n_inputs))

        x = np.asarray(x0, dtype=float).copy()

        is_lti = isinstance(model, LTIModel)
        is_discrete = is_lti and model.dt is not None

        for i in range(n_steps):
            t = float(t_grid[i])
            x_hist[i] = x

            r = np.atleast_1d(r_func(t))
            u = np.atleast_1d(controller.compute(x, r, t))
            u_hist[i] = u

            if is_lti:
                y = model.C @ x + model.D @ u
            else:
                y = model.g_func(x, u, t)
            y_hist[i] = y

            if i < n_steps - 1:
                t_next = float(t_grid[i + 1])
                if is_discrete:
                    x = model.A @ x + model.B @ u
                elif is_lti:
                    _u = u  # capture for closure

                    def f_lti(t_: float, x_: np.ndarray, _u: np.ndarray = _u) -> np.ndarray:
                        return model.A @ x_ + model.B @ _u

                    sol = solve_ivp(
                        f_lti, [t, t_next], x, method="RK45", rtol=1e-6, atol=1e-8
                    )
                    x = sol.y[:, -1]
                else:
                    _u = u

                    def f_nl(t_: float, x_: np.ndarray, _u: np.ndarray = _u) -> np.ndarray:
                        return model.f_func(x_, _u, t_)

                    sol = solve_ivp(
                        f_nl, [t, t_next], x, method="RK45", rtol=1e-6, atol=1e-8
                    )
                    x = sol.y[:, -1]

        return SimulationResult(t=t_grid, x=x_hist, y=y_hist, u=u_hist)
