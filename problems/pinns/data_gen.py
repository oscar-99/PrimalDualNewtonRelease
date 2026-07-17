from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import math
import numpy as np
import jax.numpy as jnp
from scipy.stats import qmc


def get_box_with_boundary(
    x_range: tuple[float, float],
    y_range: tuple[float, float] | None,
    n_interior: int,
    n_boundary: int,
    sample_method: str = "sobol",
    seed: int = 0,
):
    """
    Generate interior and boundary points for a 1D interval or 2D box.

    Args:
        x_range: (x_min, x_max) bounds.
        y_range: (y_min, y_max) bounds, or None for 1D.
        n_interior: number of interior sample points.
        n_boundary: number of boundary sample points in 2D.
            In 1D, if n_boundary > 0 the two endpoints are returned.
        sample_method: "sobol" or "uniform".
        seed: random seed.

    Returns:
        interior: array of shape (n_interior, d)
        boundary: array of shape (m, d), where m = n_boundary in 2D and
            m = 2 when y_range is None and n_boundary > 0.
    """
    x_min, x_max = x_range

    if y_range is None:
        d = 1
        l_bounds = (x_min,)
        u_bounds = (x_max,)
    else:
        d = 2
        y_min, y_max = y_range
        l_bounds = (x_min, y_min)
        u_bounds = (x_max, y_max)

    def sample_unit_box(n: int, dim: int, local_seed: int) -> np.ndarray:
        if n == 0:
            return np.empty((0, dim))
        if sample_method == "sobol":
            sob = qmc.Sobol(d=dim, scramble=True, seed=local_seed)
            return sob.random(n)
        if sample_method == "uniform":
            rng = np.random.default_rng(local_seed)
            return rng.uniform(size=(n, dim))
        raise ValueError(f"Unknown sample_method: {sample_method}")

    # Interior points
    u = sample_unit_box(n_interior, d, seed)
    interior = qmc.scale(u, l_bounds, u_bounds)

    # Boundary points
    if n_boundary <= 0:
        boundary = np.empty((0, d))

    elif d == 1:
        boundary = np.array([[x_min], [x_max]])

    elif d == 2:
        width = x_max - x_min
        height = y_max - y_min

        n_horiz = math.floor(n_boundary * width / (width + height))
        n_vert = n_boundary - n_horiz

        n_bottom = n_horiz // 2
        n_top = n_horiz - n_bottom
        n_left = n_vert // 2
        n_right = n_vert - n_left

        rng = np.random.default_rng(seed + 2)

        xb = qmc.scale(sample_unit_box(n_bottom, 1, seed + 10), (x_min,), (x_max,)).reshape(-1)
        xt = qmc.scale(sample_unit_box(n_top, 1, seed + 11), (x_min,), (x_max,)).reshape(-1)
        yl = qmc.scale(sample_unit_box(n_left, 1, seed + 12), (y_min,), (y_max,)).reshape(-1)
        yr = qmc.scale(sample_unit_box(n_right, 1, seed + 13), (y_min,), (y_max,)).reshape(-1)

        bottom = np.column_stack([xb, np.full(n_bottom, y_min)])
        top = np.column_stack([xt, np.full(n_top, y_max)])
        left = np.column_stack([np.full(n_left, x_min), yl])
        right = np.column_stack([np.full(n_right, x_max), yr])

        boundary = np.vstack([bottom, top, left, right])
        rng.shuffle(boundary, axis=0)

    else:
        raise ValueError("Only 1D or 2D domains are supported.")

    return jnp.asarray(interior), jnp.asarray(boundary)

def plot_domain_points(x_range: tuple[float, float],
                       y_range: tuple[float, float] | None,
                       interior, boundary):
    """
    Plot interior + boundary Sobol points on top of a 1D interval or 2D rectangle.

    Args:
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max) or None for 1D
        interior: array-like, shape (n, d)
        boundary: array-like, shape (m, d)
    """

    # convert supplied ranges into lower/upper bound arrays
    if y_range is None:
        l_bounds = np.asarray((x_range[0],))
        u_bounds = np.asarray((x_range[1],))
    else:
        l_bounds = np.asarray((x_range[0], y_range[0]))
        u_bounds = np.asarray((x_range[1], y_range[1]))
    interior = np.asarray(interior)
    boundary = np.asarray(boundary)

    d = l_bounds.shape[0]
    assert d in (1, 2), "This plotting helper only supports 1D or 2D domains."

    if d == 1:
        # --- 1D Plot ---
        fig, ax = plt.subplots(figsize=(8, 2))

        # Draw interval as line
        ax.hlines(0, l_bounds[0], u_bounds[0], colors='black', linewidth=2)

        # Plot points
        if len(interior) > 0:
            ax.scatter(interior[:, 0], np.zeros(len(interior)), 
                       color='blue', label='Interior', s=25)
        if len(boundary) > 0:
            ax.scatter(boundary[:, 0], np.zeros(len(boundary)),
                       color='red', label='Boundary', s=40)

        ax.set_ylim(-0.1, 0.1)
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_title("1D Domain + Sobol Points")
        ax.legend()
        plt.show()

    else:
        # --- 2D Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw rectangle
        rect_x = [l_bounds[0], u_bounds[0], u_bounds[0], l_bounds[0], l_bounds[0]]
        rect_y = [l_bounds[1], l_bounds[1], u_bounds[1], u_bounds[1], l_bounds[1]]
        ax.plot(rect_x, rect_y, 'k-', linewidth=2)

        # Plot points
        if len(interior) > 0:
            ax.scatter(interior[:, 0], interior[:, 1],
                       color='blue', label='Interior', s=25)
        if len(boundary) > 0:
            ax.scatter(boundary[:, 0], boundary[:, 1],
                       color='red', label='Boundary', s=40)

        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("2D Domain + Sobol Points")
        ax.legend()
        ax.set_aspect("equal", "box")
        plt.show()