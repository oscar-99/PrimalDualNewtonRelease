from dataclasses import fields
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from problems.pinns.pinn_problems import PINN_UTRUE_REGISTRY
from runners.pinn_experiment_runner import PINNRunSpec
from runners.results_io import rebuild_predictor


def _filter_dataclass_kwargs(dataclass_type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data.items() if k in allowed}


def plot_pde_surface_contour_mixed(
    u: Callable,
    uhat: Callable | None = None,
    x_range: tuple[float, float] = (0.0, jnp.pi),
    y_range: tuple[float, float] = (0.0, jnp.pi),
    n_x: int = 60,
    n_y: int = 60,
    train_points: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    cmap: str = "viridis",
    title: str = "",
):
    """
    Plot true and predicted PDE solutions as 3D surfaces, with a 2D contour
    plot for the absolute error.

    Parameters
    ----------
    u
        Batched callable accepting an array of shape (N, 2) and returning
        values of shape (N,) or (N, 1).
    uhat
        Optional batched predictor with the same convention as `u`.
    x_range, y_range
        Plotting domain bounds.
    n_x, n_y
        Grid resolution.
    train_points
        Optional tuple (x_int, x_bndry) to overlay on the error contour.
    cmap
        Matplotlib colormap.

    Returns
    -------
    fig, axs[, metrics]
        If `uhat is None`, returns (fig, ax).
        Otherwise returns (fig, axs, metrics_dict).
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    pts = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=1))

    u_exact_val = jnp.asarray(u(pts)).reshape(-1)
    U_exact = np.asarray(u_exact_val).reshape(n_x, n_y)

    if uhat is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, U_exact, cmap=cmap, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.7)
        ax.set_title("u (true)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        return fig, ax

    u_hat_val = jnp.asarray(uhat(pts)).reshape(-1)
    U_hat = np.asarray(u_hat_val).reshape(n_x, n_y)
    U_error = np.abs(U_hat - U_exact)

    fig = plt.figure(figsize=(15, 5))
    ax_true = fig.add_subplot(131, projection="3d")
    ax_pred = fig.add_subplot(132, projection="3d")
    ax_err = fig.add_subplot(133)

    surf_true = ax_true.plot_surface(X, Y, U_exact, cmap=cmap, linewidth=0, antialiased=True)
    # fig.colorbar(surf_true, ax=ax_true, shrink=0.65)
    ax_true.set_title("u (true)")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    ax_true.set_zlabel("u")

    surf_pred = ax_pred.plot_surface(X, Y, U_hat, cmap=cmap, linewidth=0, antialiased=True)
    # fig.colorbar(surf_pred, ax=ax_pred, shrink=0.65)
    ax_pred.set_title("$\hat{u}$ (predicted)")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.set_zlabel("u")

    cf_err = ax_err.contourf(X, Y, U_error, levels=50, cmap=cmap)
    fig.colorbar(cf_err, ax=ax_err)
    ax_err.set_title("Absolute Error: |u - $\hat{u}$|")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("y")

    fig.suptitle("PDE Solution and Error: " + title)

    if train_points is not None:
        interior, boundary = train_points
        interior = np.asarray(interior)
        boundary = np.asarray(boundary)
        ax_err.scatter(interior[:, 0], interior[:, 1], s=5, label="Interior")
        ax_err.scatter(boundary[:, 0], boundary[:, 1], s=8, label="Boundary")
        ax_err.legend()

    rel_l2 = np.linalg.norm(U_hat - U_exact) / max(np.linalg.norm(U_exact), 1e-16)
    rel_linf = np.max(np.abs(U_hat - U_exact)) / max(np.max(np.abs(U_exact)), 1e-16)

    return fig, (ax_true, ax_pred, ax_err), {"rel_l2": rel_l2, "rel_linf": rel_linf}

def plot_pde_contours(
    u: Callable,
    uhat: Callable,
    x_range: tuple[float, float] = (0.0, jnp.pi),
    y_range: tuple[float, float] = (0.0, jnp.pi),
    n_x: int = 60,
    n_y: int = 60,
    cmap: str = "viridis",
    title: str = "",
):
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    pts = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=1))

    U_exact = np.asarray(jnp.asarray(u(pts)).reshape(n_x, n_y))
    U_hat = np.asarray(jnp.asarray(uhat(pts)).reshape(n_x, n_y))

    # Common colour scale
    vmin = min(U_exact.min(), U_hat.min())
    vmax = max(U_exact.max(), U_hat.max())
    levels = np.linspace(vmin, vmax, 50)

    fig, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.28, right=0.88)

    surf_true = ax_true.contourf(X, Y, U_exact, levels=levels, cmap=cmap)
    ax_true.set_title("u (true)")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    ax_true.set_aspect("equal")

    surf_pred = ax_pred.contourf(X, Y, U_hat, levels=levels, cmap=cmap)
    ax_pred.set_title(r"$\hat{u}$ (predicted)")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.set_aspect("equal")

    fig.colorbar(surf_pred, ax=[ax_true, ax_pred], shrink=0.9, pad=0.04)
    fig.suptitle("PDE and Learned Solution: " + title)
    return fig, (ax_true, ax_pred)


def plot_pinn_result_surface_contour_mixed(
    result,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    n_x: int = 200,
    n_y: int = 200,
    cmap: str = "viridis",
    show_train_points: bool = True,
    just_contours: bool = False,
):
    """
    Reconstruct the trained PINN predictor from a loaded RunResult and produce
    mixed plots: 3D true surface, 3D predicted surface, 2D error contour.

    This version uses saved experiment metadata, so it works for loaded results.
    """
    experiment_metadata = result.metadata.get("experiment_metadata", {})
    if not experiment_metadata:
        raise ValueError("No 'experiment_metadata' found in result metadata.")

    problem_spec_dict = experiment_metadata.get("PINNSpec", None)
    if problem_spec_dict is None:
        raise ValueError("No 'PINNSpec' found in experiment metadata.")

    problem_spec = PINNRunSpec(**_filter_dataclass_kwargs(PINNRunSpec, problem_spec_dict))

    pinn = PINN_UTRUE_REGISTRY.get(problem_spec.name, None)
    if pinn is None:
        raise ValueError(
            f"No registered PINN found for problem name '{problem_spec.name}'."
        )

    pde_param = problem_spec.pde_param
    u_true = lambda x : pinn["u_true"](x, pde_param)
    domain = pinn["domain"]
    predictor = rebuild_predictor(result)

    if x_range is None:
        x_range = domain.x_range
    if y_range is None:
        y_range = domain.y_range

    train_points = None
    if show_train_points:
        x_int = experiment_metadata.get("x_int", None)
        x_bndry = experiment_metadata.get("x_bndry", None)
        if x_int is not None and x_bndry is not None:
            train_points = (x_int, x_bndry)

    u_true_batched = jax.vmap(u_true)
    uhat_batched = jax.vmap(predictor)

    if result.solver_name == "alm_lbfgs_default":
        title= "ALM-LBFGS"
    if result.solver_name == "pdn_ls_dual":
        title= "Primal Dual Newton (least-squares dual)"
    if result.solver_name == "pdn_zero_dual":
        title=  "Primal Dual Newton (zero dual)"
    if result.solver_name == "sqp_default":
        title= "SQP"

    if just_contours:
        return plot_pde_contours(
            u=u_true_batched,
            uhat=uhat_batched,
            x_range=x_range,
            y_range=y_range,
            n_x=n_x,
            n_y=n_y,
            cmap=cmap,
            title=title,
        )
    else:
        return plot_pde_surface_contour_mixed(
            u=u_true_batched,
            uhat=uhat_batched,
            x_range=x_range,
            y_range=y_range,
            n_x=n_x,
            n_y=n_y,
            train_points=train_points,
            cmap=cmap,
            title=title,
        )

def plot_pde_1d_fit_and_error(
    u,
    uhat,
    *,
    x_range=(0.0, 1.0),
    n_x=400,
    train_points=None,
    title="",
):
    """
    Plot a 1D PINN result with two panels:
      1. true solution and fitted solution
      2. absolute error over the interval

    Parameters
    ----------
    u : callable
        Batched callable taking array of shape (N, 1) and returning (N,) or (N, 1).
    uhat : callable
        Batched callable taking array of shape (N, 1) and returning (N,) or (N, 1).
    x_range : tuple[float, float]
        Interval bounds.
    n_x : int
        Number of evaluation points.
    train_points : tuple[jnp.ndarray, jnp.ndarray] | None
        Optional tuple (x_int, x_bndry). These are overlaid on the first panel.
    title : str
        Figure title.

    Returns
    -------
    fig, axs, metrics
    """
    x = jnp.linspace(x_range[0], x_range[1], num=n_x)[:, None]

    u_true_vals = jnp.asarray(u(x)).reshape(-1)
    u_hat_vals = jnp.asarray(uhat(x)).reshape(-1)
    err_vals = jnp.abs(u_hat_vals - u_true_vals)

    x_np = np.asarray(x[:, 0])
    u_true_np = np.asarray(u_true_vals)
    u_hat_np = np.asarray(u_hat_vals)
    err_np = np.asarray(err_vals)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax_fit, ax_err = axs

    ax_fit.plot(x_np, u_true_np, label="u (true)")
    ax_fit.plot(x_np, u_hat_np, "--", label="u_hat (predicted)")
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("u(x)")
    ax_fit.set_title("True and Predicted Solution")
    ax_fit.legend()

    if train_points is not None:
        x_int, x_bndry = train_points
        x_int = np.asarray(x_int).reshape(-1)
        x_bndry = np.asarray(x_bndry).reshape(-1)

        # Put training points along the x-axis for visual reference
        ax_fit.scatter(
            x_int,
            np.interp(x_int, x_np, u_true_np),
            s=10,
            alpha=0.5,
            label="Interior points",
        )
        ax_fit.scatter(
            x_bndry,
            np.interp(x_bndry, x_np, u_true_np),
            s=30,
            marker="x",
            label="Boundary points",
        )
        ax_fit.legend()

    ax_err.plot(x_np, err_np)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("|u - u_hat|")
    ax_err.set_title("Absolute Error")

    fig.suptitle(title)

    rel_l2 = np.linalg.norm(u_hat_np - u_true_np) / max(np.linalg.norm(u_true_np), 1e-16)
    rel_linf = np.max(np.abs(u_hat_np - u_true_np)) / max(np.max(np.abs(u_true_np)), 1e-16)

    return fig, axs, {"rel_l2": rel_l2, "rel_linf": rel_linf}


def plot_pinn_result_1d(
    result,
    *,
    x_range=None,
    n_x=400,
    show_train_points=True,
):
    """
    Reconstruct a trained 1D PINN from a loaded RunResult and plot:
      1. true vs predicted solution
      2. absolute error over the interval
    """
    experiment_metadata = result.metadata.get("experiment_metadata", {})
    if not experiment_metadata:
        raise ValueError("No 'experiment_metadata' found in result metadata.")

    problem_spec_dict = experiment_metadata.get("PINNSpec", None)
    if problem_spec_dict is None:
        raise ValueError("No 'PINNSpec' found in experiment metadata.")

    problem_spec = PINNRunSpec(**_filter_dataclass_kwargs(PINNRunSpec, problem_spec_dict))

    pinn = PINN_UTRUE_REGISTRY.get(problem_spec.name, None)
    if pinn is None:
        raise ValueError(
            f"No registered PINN found for problem name '{problem_spec.name}'."
        )

    domain = pinn["domain"]
    if domain.y_range is not None:
        raise ValueError(
            f"plot_pinn_result_1d only supports 1D problems, but '{problem_spec.name}' is 2D."
        )

    pde_param = problem_spec.pde_param
    u_true = lambda x: pinn["u_true"](x, pde_param)
    predictor = rebuild_predictor(result)

    if x_range is None:
        x_range = domain.x_range

    train_points = None
    if show_train_points:
        x_int = experiment_metadata.get("x_int", None)
        x_bndry = experiment_metadata.get("x_bndry", None)
        if x_int is not None and x_bndry is not None:
            train_points = (x_int, x_bndry)

    u_true_batched = jax.vmap(u_true)
    uhat_batched = jax.vmap(predictor)

    return plot_pde_1d_fit_and_error(
        u=u_true_batched,
        uhat=uhat_batched,
        x_range=x_range,
        n_x=n_x,
        train_points=train_points,
        title=result.solver_name,
    )

def plot_pinn_result(result, **kwargs):
    experiment_metadata = result.metadata.get("experiment_metadata", {})
    if not experiment_metadata:
        raise ValueError("No 'experiment_metadata' found in result metadata.")

    problem_spec_dict = experiment_metadata.get("PINNSpec", None)
    if problem_spec_dict is None:
        raise ValueError("No 'PINNSpec' found in experiment metadata.")

    problem_spec = PINNRunSpec(**_filter_dataclass_kwargs(PINNRunSpec, problem_spec_dict))

    if problem_spec.pinn_dim == 1:
        plot_pinn_result_1d(result, **kwargs)
    elif problem_spec.pinn_dim == 2:
        plot_pinn_result_surface_contour_mixed(result, **kwargs)
    else:
        raise ValueError("invalid pinn dimension.")
    
def _pinn_method_title(result) -> str:
    solver_name = getattr(result, "solver_name", "run")
    if solver_name == "alm_lbfgs_default":
        return "ALM-LBFGS"
    if solver_name == "pdn_ls_dual":
        return "PDN (least-squares dual)"
    if solver_name == "pdn_zero_dual":
        return "PDN (zero dual)"
    if solver_name == "sqp_default":
        return "SQP"
    return solver_name.replace("_", " ")

def _get_pinn_2d_error_data(
    result,
    *,
    n_x: int = 200,
    n_y: int = 200,
):
    """
    Reconstruct a 2D PINN predictor from a RunResult and return the grid and
    absolute error field over the problem domain.
    """
    experiment_metadata = result.metadata.get("experiment_metadata", {})
    if not experiment_metadata:
        raise ValueError("No 'experiment_metadata' found in result metadata.")

    problem_spec_dict = experiment_metadata.get("PINNSpec", None)
    if problem_spec_dict is None:
        raise ValueError("No 'PINNSpec' found in experiment metadata.")

    problem_spec = PINNRunSpec(**_filter_dataclass_kwargs(PINNRunSpec, problem_spec_dict))
    pinn = PINN_UTRUE_REGISTRY.get(problem_spec.name, None)
    if pinn is None:
        raise ValueError(f"No registered PINN found for problem name '{problem_spec.name}'.")

    domain = pinn["domain"]
    if domain.y_range is None:
        raise ValueError(
            f"plot_pinn_error_heatmaps only supports 2D PINN problems, but '{problem_spec.name}' is 1D."
        )

    pde_param = problem_spec.pde_param
    u_true = lambda x: pinn["u_true"](x, pde_param)
    predictor = rebuild_predictor(result)

    x_range = domain.x_range
    y_range = domain.y_range

    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=1))

    u_true_batched = jax.vmap(u_true)
    uhat_batched = jax.vmap(predictor)

    U_true = np.asarray(jnp.asarray(u_true_batched(pts)).reshape(n_x, n_y))
    U_hat = np.asarray(jnp.asarray(uhat_batched(pts)).reshape(n_x, n_y))
    U_err = np.abs(U_hat - U_true)

    return X, Y, U_err, _pinn_method_title(result)


def plot_pinn_error_heatmaps(
    results,
    *,
    n_x: int = 200,
    n_y: int = 200,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
):
    """
    Plot side-by-side absolute-error heatmaps for multiple 2D PINN runs using
    a common color scale and a single shared colorbar.
    """
    error_data = []
    vmax = 0.0

    for result in results:
        X, Y, U_err, title = _get_pinn_2d_error_data(result, n_x=n_x, n_y=n_y)
        error_data.append((X, Y, U_err, title))
        vmax = max(vmax, float(np.nanmax(U_err)))

    if not error_data:
        raise ValueError("No PINN results were provided.")

    n_plots = len(error_data)
    if figsize is None:
        figsize = (3.6 * n_plots + 1.2, 3.4)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    mappable = None
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    for ax, (X, Y, U_err, title) in zip(axes, error_data):
        mappable = ax.imshow(
            U_err.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.colorbar(
        mappable,
        ax=axes.tolist(),
        fraction=0.035,
        pad=0.08,
        shrink=0.95,
        label=r"$|u^\star-\hat{u}|$",
    )
    fig.subplots_adjust(wspace=0.22, right=0.86)
    return fig, axes