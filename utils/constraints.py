# Implement some common constraint functions.
import jax
import jax.numpy as jnp
import optax
import utils.utils as util

def get_ellipsoid_constraint(Q: jax.Array):
    """
    Returns a constraint which encodes ellipsoid defined by x^T Q x - 1 = 0. Q should be positive semidefinite."""
    @jax.jit
    def ellipsoid_constraint(params):
        params_flat, _, _ = util.flatten(params)
        return (jnp.dot(params_flat, jnp.dot(Q, params_flat)) - 1.0).reshape((1,))
    
    return ellipsoid_constraint

def get_ball_constraint(r: float):
    """
    Returns a constraint function that projects onto an L2 ball of given radius.

    Args:
        r (float): Radius of the L2 ball.
    Returns:
        function: l2 ball constraint with radius r.
    """
    @jax.jit
    def ball_constraint(params):
        return (optax.tree.norm(params, ord=2, squared=True) - r**2).reshape((1,))

    return ball_constraint

def get_linear_constraint(A: jax.Array, b: jax.Array):
    """
    Returns a linear constraint function of the form Ax - b = 0.

    Args:
        A (jax.Array): Coefficient matrix.
        b (jax.Array): Right-hand side vector.
    Returns:
        function: linear constraint function.
    """

    @jax.jit
    def linear_constraint(params):
        params_flat, _, _ = util.flatten(params)
        return A@params_flat - b + 0.0*optax.tree.norm(params, ord=2, squared=True) # add zero quadratic term to force tracing this fixes a bug with second derivatives of linear constraints.

    return linear_constraint

def join_constraints(constraints):
    """
    Joins multiple constraint functions into a single constraint function.

    Args:
        constraints (list): List of constraint functions to be joined.
    Returns:
        function: A single constraint function that combines all input constraints.
    """

    @jax.jit
    def combined_constraint(params):
        return jnp.concatenate([constr(params).ravel() for constr in constraints])

    return combined_constraint