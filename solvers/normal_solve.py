from typing import NamedTuple

import jax
import jax.numpy as jnp

class QRNormalResult(NamedTuple):
    normal_step: jax.Array
    Omega: jax.Array
    R: jax.Array
    residual_norm: jax.Array
    rank_estimate: jax.Array

class SVDNormalResult(NamedTuple):
    normal_step: jax.Array
    U : jax.Array
    s : jax.Array
    Omega : jax.Array
    residual_norm: jax.Array
    rank_estimate: jax.Array
    sigma_max: jax.Array
    sigma_min: jax.Array

def get_QR_normal_solver(oracle, rank_tol: float = 1e-10):

    @jax.jit
    def QR_normal_solver(
        params,
        c_val: jax.Array,
        constraint_data=None,
    ):
        # Dense Jacobian from the oracle
        JT = oracle.cons(params, constraint_data=constraint_data, vals="J").T

        # Economy QR of J^T
        Q, R = jnp.linalg.qr(JT, mode="reduced")

        # Solve R^T y = -c
        y = jax.lax.linalg.triangular_solve(
            R,
            -c_val,
            left_side=True,
            lower=False,
            transpose_a=True,
        )

        normal_step = Q @ y

        residual = JT.T @ normal_step + c_val
        residual_norm = jnp.linalg.norm(residual)

        # Rank diagnostic (not returned as diag)
        diagR = jnp.abs(jnp.diag(R))
        scale = jnp.maximum(1.0, jnp.max(diagR))
        rank_estimate = jnp.sum(diagR > rank_tol * scale)

        return QRNormalResult(
            normal_step=normal_step,
            Omega=Q,
            R=R,
            residual_norm=residual_norm,
            rank_estimate=rank_estimate,
        )
    
    return QR_normal_solver

def get_SVD_normal_solver(oracle, rank_tol: float = 1e-10):

    @jax.jit
    def SVD_normal_solver(
        params,
        c_val: jax.Array,
        constraint_data=None,
    ):
        J = oracle.cons(params, constraint_data=constraint_data, vals="J")

        # Full-matrices=False gives:
        # J is c x d
        # J = U diag(s) Vt
        # U: (c, r), s: (r,), Omegat: (r, d), r = min(d, c)
        U, s, Omegat = jnp.linalg.svd(J, full_matrices=False)

        sigma_max = jnp.max(s)
        scale = jnp.maximum(1.0, sigma_max)
        keep = s >= rank_tol * scale
        rank_estimate = jnp.sum(keep)

        # Moore-Penrose pseudoinverse solve for min-norm step:
        # v = -V diag(1/s) U^T c
        safe_inv_s = jnp.where(keep, 1.0 / s, 0.0)
        normal_step = -(Omegat.T @ (safe_inv_s * (U.T @ c_val)))

        residual = J @ normal_step + c_val
        residual_norm = jnp.linalg.norm(residual)

        sigma_min = jnp.where(
            rank_estimate > 0,
            jnp.min(jnp.where(keep, s, jnp.inf)),
            0.0,
        )
        Omega = Omegat.T * keep[None, :]

        return SVDNormalResult(
            normal_step=normal_step,
            U = U,
            s = s,
            Omega = Omega,
            residual_norm=residual_norm,
            rank_estimate=rank_estimate,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    return SVD_normal_solver