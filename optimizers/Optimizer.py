from numbers import Integral, Real
from typing import Any

import jax.numpy as jnp
import numpy as np

class Optimizer:
    """Minimal base class for optimization algorithms.

    This class only contains generic history and metric-handling utilities so it
    can be shared across different optimizers without constraining their solver
    setup or step interfaces.
    """

    def __init__(self):
        self.history: dict[str, list[Any]] = {}

    @staticmethod
    def _coerce_metric_value(name: str, value: Any) -> bool | int | float | str:
        """Coerce a metric value to a scalar history-friendly Python object.

        Accepted values are:
        - Python bool / int / float / str
        - NumPy / JAX scalar arrays
        - size-1 NumPy / JAX arrays such as ``[1]`` or ``[[1]]``

        A ``ValueError`` is raised for values that cannot be unambiguously
        coerced to a scalar.
        """
        if value is None:
            return None

        if isinstance(value, (bool, str)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value)

        if hasattr(value, "shape"):
            arr = jnp.asarray(value)
            if arr.size != 1:
                raise ValueError(
                    f"Metric '{name}' must be scalar-like, but got shape {arr.shape}."
                )

            scalar = np.asarray(arr).reshape(()).item()
            if scalar is None:
                return None 
            if isinstance(scalar, (bool, np.bool_)):
                return bool(scalar)
            if isinstance(scalar, Integral):
                return int(scalar)
            if isinstance(scalar, Real):
                return float(scalar)
            if isinstance(scalar, str):
                return scalar

            raise ValueError(
                f"Metric '{name}' has unsupported scalar type {type(scalar)}."
            )

        raise ValueError(f"Metric '{name}' has unsupported type {type(value)}.")

    @classmethod
    def _sanitize_metrics(cls, metrics: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
        """Validate and prefix metrics before storing them in history."""
        if metrics is None:
            return {}

        out: dict[str, Any] = {}
        for key, value in dict(metrics).items():
            out[f"{prefix}{key}"] = cls._coerce_metric_value(key, value)
        return out

    def reset_history(self) -> None:
        self.history = {}

    def store_history(self, statistics, aux_metrics: dict[str, Any] | None, test_metrics: dict[str, Any] | None) -> None:
        row = self._sanitize_metrics(statistics._asdict())
        row.update(self._sanitize_metrics(aux_metrics, prefix="train_"))
        row.update(self._sanitize_metrics(test_metrics, prefix="test_"))

        if not self.history:
            self.history = {key: [] for key in row}

        for key, value in row.items():
            self.history.setdefault(key, []).append(value)
