"""SINDy-based system identifier implementing IdentifierProtocol."""

from __future__ import annotations

from typing import Optional

import numpy as np


class SINDyIdentifier:
    """Sparse Identification of Nonlinear Dynamics (SINDy) wrapper.

    pysindy is imported lazily inside :meth:`fit` so this class can be
    instantiated even when pysindy is not installed.

    Parameters
    ----------
    feature_library : pysindy feature library (optional).
    optimizer       : pysindy optimizer (optional).
    dt              : sample time used when ``data['t']`` is not provided.
    """

    def __init__(
        self,
        feature_library: Optional[object] = None,
        optimizer: Optional[object] = None,
        dt: float = 0.01,
    ) -> None:
        self.feature_library = feature_library
        self.optimizer = optimizer
        self.dt = dt
        self._model: Optional[object] = None

    @property
    def model(self):
        """The fitted pysindy.SINDy model (None before fitting)."""
        return self._model

    def fit(self, data: dict) -> None:
        """Fit the SINDy model.

        Parameters
        ----------
        data : dict with keys ``'x'`` (state trajectory, shape (N, n)),
               optionally ``'u'`` (input, shape (N, m)) and ``'t'`` (time).
        """
        import pysindy as ps  # noqa: PLC0415

        x = np.asarray(data["x"], dtype=float)
        u = data.get("u")
        t = data.get("t")

        kwargs: dict = {}
        if self.feature_library is not None:
            kwargs["feature_library"] = self.feature_library
        if self.optimizer is not None:
            kwargs["optimizer"] = self.optimizer

        self._model = ps.SINDy(**kwargs)

        fit_kwargs: dict = {}
        if t is not None:
            fit_kwargs["t"] = np.asarray(t, dtype=float)
        else:
            fit_kwargs["t"] = self.dt

        if u is not None:
            self._model.fit(x, u=np.asarray(u, dtype=float), **fit_kwargs)
        else:
            self._model.fit(x, **fit_kwargs)

    def predict(
        self, x0: np.ndarray, u_seq: Optional[np.ndarray], t_grid: np.ndarray
    ) -> np.ndarray:
        """Simulate the identified model from *x0* over *t_grid*.

        Parameters
        ----------
        x0     : initial state, shape (n,)
        u_seq  : input sequence, shape (N, m), or None for autonomous systems.
        t_grid : time points, shape (N,)

        Returns
        -------
        x_pred : shape (N, n)
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x0 = np.asarray(x0, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float)
        if u_seq is not None:
            return self._model.simulate(x0, t_grid, u=np.asarray(u_seq, dtype=float))
        return self._model.simulate(x0, t_grid)

    def get_equations(self) -> list[str]:
        """Return human-readable discovered equations."""
        if self._model is None:
            return []
        return self._model.equations()
