
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional

import optax
from utils.differ import get_hvp

# =========================
# Solver state (fast PyTree) tracks state of CR across iterations. 
# =========================
class CRState(NamedTuple):
    FLAG: jnp.int32           # 0 running, 1 nc detected, 2 converged, 3 max iters
    it: jnp.int32
    # Storage vectors for CR iterations.
    x: jnp.ndarray
    r: jnp.ndarray
    p: jnp.ndarray
    Hp: jnp.ndarray
    Hr: jnp.ndarray
    # Store some projected vectors. Saves some redundant projections in body fun.
    Px: jnp.ndarray
    Pr: jnp.ndarray
    Pp: jnp.ndarray
    PHp: jnp.ndarray
    PHr: jnp.ndarray
    # Scalars for CR iterations.
    rho: jnp.ndarray
    rnorms: jnp.ndarray
    Hpnorms: jnp.ndarray
    Hrnorms: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray

class CRResult(NamedTuple):
    step: jax.Array
    flag: jax.Array
    num_iter: jax.Array
    projected_residual_norm: jax.Array
    curvature: jax.Array

def get_CR_hvp_solver(hvpx, *, tol, nc_tol, max_iter, regularizer):
    """
    Returns a conjugate residual solver based on a Hessian-vector product function.
    
    For jit puposes the CR solver must depend directly on the parameters.
    """
    @jax.jit
    def jit_CR(params : optax.Params, 
               b:jax.Array, 
               V:jax.Array|None,
               **kwargs):
        # Evaluate hvp oracle, we assume that params are the first argument.
        hvp = hvpx(params, **kwargs)
        if regularizer > 0.:
            hvp_reg = lambda v: hvp(v) + regularizer * v
        else:
            hvp_reg = hvp
            
        if V is None:
            proj = lambda v: v
        else:
            def proj(v):
                return v - jnp.dot(V, jnp.dot(V.T, v))  # projection onto V^\perp. If V is basis for R(J^T) this is projection onto Null(J)
        proj_jit = jax.jit(proj)    
        return CR_proj(hvp_reg, b, proj_jit, tol=tol, nc_tol=nc_tol, max_iter=max_iter)

    return jit_CR    

def CR_proj(
    Hv : Callable[[jax.Array], jax.Array], 
    b: jnp.ndarray,
    proj : Callable[[jax.Array], jax.Array], 
    *,        
    tol: float,
    nc_tol: float,
    max_iter:int):
    
    def cond_fun(s: CRState):
        # Continue if termination flag not identified and max iters not reached
        return jnp.logical_and(s.FLAG == 0, s.it < max_iter) 
    
    # helpers for body_fun
    def nc_detected(s):
        return s._replace(FLAG=jnp.array(1, dtype=jnp.int32))

    def nc_not_detected(s):
        alpha = s.rho / s.Hpnorms
        x = s.x + alpha * s.p
        r = s.r - alpha * s.Hp
        Px = s.Px + alpha * s.Pp
        Pr = s.Pr - alpha * s.PHp 
        rnorms = jnp.vdot(r, Pr)

        Hr = Hv(Pr)
        PHr = proj(Hr)
        rho_new = jnp.vdot(r, PHr) # <r_r, H_t r_t>
        beta = rho_new / s.rho

        p = r + beta * s.p
        Hp = Hr + beta * s.Hp
        Pp = Pr + beta * s.Pp
        PHp = PHr + beta * s.PHp

        Hpnorms = jnp.vdot(Hp, PHp)
        Hrnorms = jnp.vdot(Hr, PHr)
        Hxnorms = jnp.vdot(r - b, Pr - Pb)

        conv = (Hrnorms <= (tol**2) * Hxnorms)
        new_flag = jnp.where(conv, jnp.array(2, dtype=jnp.int32), s.FLAG)

        return CRState(
            FLAG=new_flag,
            it=s.it + 1,
            x=x, r=r, p=p, Hp=Hp, Hr=Hr,
            Px=Px, Pr=Pr, Pp=Pp, PHp=PHp, PHr=PHr,
            rho=rho_new, rnorms=rnorms,
            Hpnorms=Hpnorms, Hrnorms=Hrnorms,
            alpha=alpha, beta=beta
        )

    def body_fun(s: CRState):
        nc = (s.rho <= nc_tol * s.rnorms) # negative curvature detected.
        return jax.lax.cond(nc, nc_detected, nc_not_detected, s)
    
    # Initialization
    r = b; p = b
    Pb = proj(b); Pr = Pb; Pp = Pb # can store projections
    x = jnp.zeros_like(b)
    Px = x
    Hr = Hv(Pr)
    Hp = Hr
    PHr = proj(Hr); PHp = PHr

    rho = jnp.vdot(r, PHr)
    rnorms = jnp.vdot(r, Pr)
    Hpnorms = jnp.vdot(Hp, PHp)
    Hrnorms = Hpnorms
    alpha = rho / Hpnorms
    beta = jnp.array(0., dtype=b.dtype)

    state = CRState(
        FLAG=jnp.array(0, dtype=jnp.int32),
        it=jnp.array(0, dtype=jnp.int32),
        x=x, r=r, p=p, Hp=Hp, Hr=Hr,
        Px=Px, Pr=Pr, Pp=Pp, PHp=PHp, PHr=PHr,
        rho=rho, rnorms=rnorms, Hpnorms=Hpnorms, Hrnorms=Hrnorms,
        alpha=alpha, beta=beta
    )

    out = jax.lax.while_loop(cond_fun, body_fun, state)

    flag = jnp.where(
        (out.FLAG == 0) & (out.it >= max_iter),
        jnp.asarray(3, dtype=out.FLAG.dtype),
        out.FLAG
    )

    step = jax.lax.switch(
        flag,
        branches=(
            lambda: out.Px,
            lambda: out.Pr,
            lambda: out.Px,
            lambda: out.Px,
        ),
    )

    return CRResult(
        step=step,
        flag=flag,
        num_iter=out.it,
        projected_residual_norm=jnp.linalg.norm(out.Pr),
        curvature=out.rho)
