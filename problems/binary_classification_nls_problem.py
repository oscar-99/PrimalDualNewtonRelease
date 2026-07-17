"""Binary classification problem builders using the shared Problem interface.

This module defines a sigmoid least-squares binary classification model with
mixed equality constraints on the weight vector,

    A w - beta = 0,
    ||w||^2 - radius_sq = 0,

where params = [w, b] contains the weights and scalar bias. The builders return
`Problem` objects from `build_problem.py`, so the resulting instances can be
passed directly to the runner/solver stack.
"""
from typing import Any

import jax
import jax.numpy as jnp

from problems.build_problem import Problem


Array = jax.Array


def unpack_linear_sigmoid_params(params: Array) -> tuple[Array, Array]:
    """Split a flat parameter vector into weights and scalar bias."""
    params = jnp.asarray(params)
    if params.ndim != 1:
        raise ValueError(f"`params` must be 1D, got shape {params.shape}.")
    if params.shape[0] < 2:
        raise ValueError("`params` must contain at least one weight and one bias.")
    return params[:-1], params[-1]


def pack_linear_sigmoid_params(w: Array, b: Array | float) -> Array:
    """Pack weights and scalar bias into a flat parameter vector."""
    w = jnp.asarray(w)
    b = jnp.asarray(b, dtype=w.dtype)
    return jnp.concatenate([w, b.reshape(1)])


def sigmoid_predict_logits(params: Array, X: Array) -> Array:
    """Return the linear logits Xw + b."""
    w, b = unpack_linear_sigmoid_params(params)
    return X @ w + b


def sigmoid_predict_proba(params: Array, X: Array) -> Array:
    """Return sigmoid probabilities."""
    return jax.nn.sigmoid(sigmoid_predict_logits(params, X))


def sigmoid_least_squares_objective(
    params: Array,
    data: dict[str, Array] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Binary classification least-squares objective with auxiliary metrics."""
    if data is None:
        raise ValueError("Objective requires a data dictionary with keys 'X' and 'y'.")

    X = jnp.asarray(data["X"])
    y = jnp.asarray(data["y"]).reshape(-1)

    p = sigmoid_predict_proba(params, X)
    residual = p - y
    mse = 0.5 * jnp.mean(residual**2)

    preds = p >= 0.5
    acc = jnp.mean(preds == (y >= 0.5))

    aux = {
        "mse": mse,
        "accuracy": acc,
        "mean_prob": jnp.mean(p),
        "param_norm": jnp.linalg.norm(params),
        "weight_norm": jnp.linalg.norm(params[:-1]),
    }
    return mse, aux


def mixed_linear_quadratic_constraint(
    params: Array,
    data: dict[str, Array] | None = None,
) -> Array:
    """Mixed linear/quadratic equality constraints acting on the weight vector."""
    if data is None:
        raise ValueError(
            "Constraint requires a data dictionary with keys 'A', 'beta', and 'radius_sq'."
        )

    w, _ = unpack_linear_sigmoid_params(params)
    A = jnp.asarray(data["A"])
    beta = jnp.asarray(data["beta"]).reshape(-1)
    radius_sq = jnp.asarray(data["radius_sq"])

    if A.ndim != 2:
        raise ValueError(f"`A` must be 2D, got shape {A.shape}.")
    if A.shape[1] != w.shape[0]:
        raise ValueError(
            f"Constraint matrix A has shape {A.shape}, incompatible with weight dimension {w.shape[0]}."
        )
    if beta.shape != (A.shape[0],):
        raise ValueError(f"`beta` must have shape ({A.shape[0]},), got {beta.shape}.")

    c_lin = A @ w - beta
    c_quad = jnp.vdot(w, w) - radius_sq
    return jnp.concatenate([c_lin, c_quad.reshape(1)])


def evaluate_binary_classifier(params: Array, data: dict[str, Array]) -> dict[str, Array]:
    """Evaluate basic classification metrics on a dataset."""
    X = jnp.asarray(data["X"])
    y = jnp.asarray(data["y"]).reshape(-1)

    logits = sigmoid_predict_logits(params, X)
    p = jax.nn.sigmoid(logits)
    residual = p - y
    preds = p >= 0.5

    return {
        "mse": 0.5 * jnp.mean(residual**2),
        "accuracy": jnp.mean(preds == (y >= 0.5))
    }


def make_test_function(test_data: dict[str, Array]):
    """Return a held-out evaluation function with the Problem signature."""

    def test(params: Array, data: Any = None) -> dict[str, Array]:
        del data
        return evaluate_binary_classifier(params, test_data)

    return test


def _validate_binary_dataset(X: Array, y: Array) -> tuple[Array, Array]:
    X = jnp.asarray(X)
    y = jnp.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"`X` must be 2D, got shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of samples in X and y must match, got {X.shape[0]} and {y.shape[0]}."
        )

    unique = jnp.unique(y)
    if not jnp.all((unique == 0) | (unique == 1)):
        raise ValueError("`y` must contain binary labels encoded as 0/1.")

    return X, y.astype(X.dtype)


def train_test_split_arrays(
    X: Array,
    y: Array,
    *,
    test_fraction: float = 0.25,
    shuffle: bool = True,
    key: Array | None = None,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Split arrays into train/test dictionaries."""
    X, y = _validate_binary_dataset(X, y)

    n = X.shape[0]
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("`test_fraction` must lie in (0, 1).")

    if shuffle:
        if key is None:
            raise ValueError("A PRNG key is required when `shuffle=True`.")
        perm = jax.random.permutation(key, n)
        X = X[perm]
        y = y[perm]

    n_test = max(1, int(test_fraction * n))
    n_train = n - n_test
    if n_train <= 0:
        raise ValueError("Train split is empty; reduce `test_fraction`.")

    train_data = {"X": X[:n_train], "y": y[:n_train]}
    test_data = {"X": X[n_train:], "y": y[n_train:]}
    return train_data, test_data


def standardize_from_train(
    train_data: dict[str, Array],
    test_data: dict[str, Array],
    *,
    eps: float = 1e-8,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
    """Standardize features using train-split statistics only."""
    X_train = jnp.asarray(train_data["X"])
    X_test = jnp.asarray(test_data["X"])

    mean = jnp.mean(X_train, axis=0)
    std = jnp.maximum(jnp.std(X_train, axis=0), eps)

    train_std = {"X": (X_train - mean) / std, "y": jnp.asarray(train_data["y"])}
    test_std = {"X": (X_test - mean) / std, "y": jnp.asarray(test_data["y"])}
    stats = {"feature_mean": mean, "feature_std": std}
    return train_std, test_std, stats


def make_constraint_data_from_reference(
    reference_params: Array,
    *,
    A: Array,
) -> dict[str, Array]:
    """Construct consistent constraint data so the reference point is feasible."""
    w_ref, _ = unpack_linear_sigmoid_params(reference_params)
    A = jnp.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"`A` must be 2D, got shape {A.shape}.")
    if A.shape[1] != w_ref.shape[0]:
        raise ValueError(
            f"Constraint matrix A has shape {A.shape}, incompatible with weight dimension {w_ref.shape[0]}."
        )

    return {
        "A": A,
        "beta": A @ w_ref,
        "radius_sq": jnp.vdot(w_ref, w_ref),
    }


def _default_initial_point(reference_params: Array, *, perturbation_scale: float) -> Array:
    """Deterministic initial point obtained by perturbing the reference parameters."""
    reference_params = jnp.asarray(reference_params)
    n = reference_params.shape[0]
    direction = jnp.linspace(-1.0, 1.0, n, dtype=reference_params.dtype)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0)
    return reference_params + perturbation_scale * direction


def build_binary_classification_problem(
    *,
    name: str,
    train_data: dict[str, Array],
    test_data: dict[str, Array],
    constraint_data: dict[str, Array],
    x0: Array,
) -> Problem:
    """Assemble and return a runner-facing Problem instance."""
    train_data = {"X": jnp.asarray(train_data["X"]), "y": jnp.asarray(train_data["y"]).reshape(-1)}
    test_data = {"X": jnp.asarray(test_data["X"]), "y": jnp.asarray(test_data["y"]).reshape(-1)}

    constraint_data = {
        "A": jnp.asarray(constraint_data["A"]),
        "beta": jnp.asarray(constraint_data["beta"]).reshape(-1),
        "radius_sq": jnp.asarray(constraint_data["radius_sq"]),
    }

    return Problem(
        name=name,
        x0=jnp.asarray(x0),
        objective=sigmoid_least_squares_objective,
        constraint=mixed_linear_quadratic_constraint,
        objective_data=train_data,
        constraint_data=constraint_data,
        test=make_test_function(test_data),
        obj_has_aux=True,
    )

def make_binary_classification_problem_from_arrays(
    *,
    X: Array,
    y: Array,
    A: Array,
    test_fraction: float = 0.25,
    split_key: Array = jax.random.PRNGKey(0),
    shuffle: bool = True,
    standardize: bool = True,
    x0: Array | None = None,
    reference_params: Array | None = None,
    x0_perturbation_scale: float = 0.25,
    name: str = "real_data_binary_classification",
) -> Problem:
    """Create a constrained classification problem from user-provided arrays."""
    train_data, test_data = train_test_split_arrays(
        X,
        y,
        test_fraction=test_fraction,
        shuffle=shuffle,
        key=split_key if shuffle else None,
    )

    if standardize:
        train_data, test_data, standardize_stats = standardize_from_train(train_data, test_data)
    else:
        standardize_stats = None

    n_features = train_data["X"].shape[1]
    if reference_params is None:
        reference_params = pack_linear_sigmoid_params(jnp.zeros((n_features,)), 0.0)
    else:
        reference_params = jnp.asarray(reference_params)

    constraint_data = make_constraint_data_from_reference(reference_params, A=jnp.asarray(A))

    if x0 is None:
        x0 = _default_initial_point(reference_params, perturbation_scale=x0_perturbation_scale)
    else:
        x0 = jnp.asarray(x0)

    return build_binary_classification_problem(
        name=name,
        train_data=train_data,
        test_data=test_data,
        constraint_data=constraint_data,
        x0=x0,
    )

