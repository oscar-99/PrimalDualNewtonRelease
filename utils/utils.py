from typing import Callable
import math

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import PyTreeDef


# Utilities to flatten and unflatten params.

def flatten(
    params: optax.Params,
) -> tuple[jax.Array, PyTreeDef, list[tuple[int, ...]]]:
    """Flatten params to a single vector.

    Returns the flattened vector together with the treedef and leaf shapes for
    unflattening.
    """
    leaves, treedef = jax.tree.flatten(params)
    flat_vecs = [jnp.ravel(x) for x in leaves]
    return jnp.concatenate(flat_vecs), treedef, [x.shape for x in leaves]


def unflatten(
    params_flat: jax.Array,
    treedef: PyTreeDef,
    leaf_shapes: list[tuple[int, ...]],
) -> optax.Params:
    """Unflatten ``params_flat`` into the structure defined by ``treedef``."""
    offsets: list[int] = []
    for leaf_shape in leaf_shapes:
        size = int(math.prod(leaf_shape))
        offsets.append(size if not offsets else size + offsets[-1])
    if offsets:
        offsets.pop()

    params_split = jnp.split(params_flat, offsets)
    reshaped = [
        jnp.reshape(param, leaf_shape)
        for param, leaf_shape in zip(params_split, leaf_shapes)
    ]
    return jax.tree.unflatten(treedef, reshaped)


def make_func_flat(
    fun: Callable | None,
    treedef: PyTreeDef,
    leaf_shapes: list[tuple[int, ...]],
) -> Callable | None:
    """Wrap a native-parameter callable so it accepts a flat parameter vector."""
    if fun is None:
        return None

    def f_flat(params_flat, *args, **kwargs):
        params = unflatten(params_flat, treedef, leaf_shapes)
        return fun(params, *args, **kwargs)

    return f_flat

def make_2d_grid(x_range: tuple[float, float], 
                 y_range: tuple[float, float], 
                 n_x: int, n_y: int) -> jnp.ndarray:
    """
    Generate a 2D grid of n_x x n_y points over specified ranges.
    
    Args:
        x_range: (x_min, x_max) range for x-axis
        y_range: (y_min, y_max) range for y-axis
        n_x: number of points along x-axis
        n_y: number of points along y-axis
        
    Returns:
        Array of shape (m, 2) where m = n_x * n_y, each row is [x, y]
    """
    x = jnp.linspace(x_range[0], x_range[1], n_x)
    y = jnp.linspace(y_range[0], y_range[1], n_y)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    # Stack and reshape to (m, 2)
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    return grid

import jax.numpy as jnp


def uniform_box_stats(*intervals):
    """
    Compute per-coordinate mean and standard deviation for a uniform
    distribution on a box domain.

    Parameters
    ----------
    *intervals
        One (lower, upper) tuple per coordinate, e.g.
            (-1.0, 1.0), (-1.0, 1.0)

    Returns
    -------
    mean : jax.Array, shape (d,)
    std  : jax.Array, shape (d,)
    """
    if len(intervals) == 0:
        raise ValueError("At least one interval must be provided.")

    lowers = []
    uppers = []

    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(
                "Each interval must be a length-2 tuple like (lower, upper)."
            )

        lower, upper = interval
        if upper <= lower:
            raise ValueError(
                f"Invalid interval {interval}: upper must be greater than lower."
            )

        lowers.append(lower)
        uppers.append(upper)

    lower = jnp.asarray(lowers, dtype=jnp.float32)
    upper = jnp.asarray(uppers, dtype=jnp.float32)

    mean = 0.5 * (lower + upper)
    std = (upper - lower) / jnp.sqrt(12.0)
    return mean, std