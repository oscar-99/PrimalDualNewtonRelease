from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

"""
Regularized SQP inner solver based on the primal-dual linear system in
Byrd, Curtis, and Nocedal (https://link.springer.com/article/10.1007/s10107-008-0248-3)


Note on sign convention
-----------------------
The paper uses residual = K z + rhs, while this implementation uses the CR
convention residual = b - K z with b = -rhs. Hence the CR residual is the
negative of the paper residual.

This is harmless for the Byrd/SQPReg termination tests, since they depend
only on residual norms. The sign difference only matters when reconstructing
vector identities from the residual itself, and those formulas in this file
follow the CR convention consistently.
"""


class SQPRegTerminationParams(NamedTuple):
    """Hyperparameters for the paper termination tests and Hessian modification rule."""

    sigma: float   # = tau * (1 - epsilon)
    kappa: float   # residual factor in Test I
    epsilon: float # paper parameter in (4.3a)
    beta: float    # paper parameter in (4.3b)
    tau: float     # penalty update factor in (4.5)
    theta: float   # curvature threshold
    psi: float     # tolerance for nu/Upsilon comparison in Test II

# PARAMETERS FROM THE PAPER (except theta, which requires Hessian norm)
SQP_DEFAULT_TERMINATION_PARAMS = SQPRegTerminationParams(
    sigma=2e-1 * (1.0 - 1e-2),  # tau(1-epsilon)
    kappa=1e-2,
    epsilon=1e-2,
    beta=1e1,
    tau=2e-1,
    theta=1e-3,
    psi=1e1,
)


class SQPRegInnerContext(NamedTuple):
    """Outer-step quantities frozen during one inner solve phase."""

    penalty_parameter: float
    stationarity_rhs_norm: float
    constraint_norm: float


class SQPRegInnerIterate(NamedTuple):
    """Diagnostics for the current trial primal-dual iterate used by the paper tests."""

    d: jax.Array
    delta: jax.Array
    rho: jax.Array
    r: jax.Array
    grad_fTd: float
    dWd_half: float
    step_norm_sq: float
    nu: float
    Upsilon: float
    rho_norm: float
    r_norm: float
    residual_norm: float


class SQPRegTerminationDecision(NamedTuple):
    """Compact decision from the paper termination checker.

    status codes:
      0 = continue iterating
      1 = accept via Test I
      2 = accept via Test II
      3 = modify Hessian / increase regularization
    """

    accept: bool
    status: int
    trial_penalty_parameter: float


class CRState(NamedTuple):
    """Algebraic state for one CR phase on a fixed shifted KKT system.

    This stores only the recurrence quantities needed by CR itself.
    Paper-specific diagnostics such as Wd are reconstructed in the controller.
    """

    x: jax.Array
    r: jax.Array
    p: jax.Array
    Ap: jax.Array
    rAr: float
    residual_norm: float
    phase_iter: int
    valid: bool


class SQPRegInnerLoopState(NamedTuple):
    """Controller state for one jitted SQPReg inner solve.

    This augments the current CR phase (`cr_state`) with the loop-level
    bookkeeping needed by the `lax.while_loop` controller: current
    regularization, total CR iterations across all phases, restart count,
    termination status, and whether any restart was triggered by CR breakdown
    or by the phase-iteration cap.
    """

    cr_state: CRState
    mu: float
    total_iter: int
    restart_count: int
    done: bool
    accepted: bool
    flag: int
    trial_penalty_parameter: float
    saw_CR_breakdown: bool
    saw_phase_limit_restart: bool


class SQPRegSolveResult(NamedTuple):
    """Terminal output of the SQPReg inner solve.

    Contains the final primal-dual trial step, the shifted KKT residual
    components at that step, the final regularization parameter, and
    controller metadata describing how the solve terminated.

    `flag` encodes the terminal outcome:
      1 = accepted via Test I
      2 = accepted via Test II
      3 = terminated after exceeding the restart budget

    `saw_CR_breakdown` is True only if additional regularization was required
    because a CR phase became algebraically invalid.

    `saw_phase_limit_restart` is True only if additional regularization was
    required because a CR phase ran to the phase-iteration cap without
    acceptance.
    """

    d: jax.Array
    delta: jax.Array
    r: jax.Array
    rho: jax.Array
    mu_final: float
    total_iter: int
    restart_count: int
    accepted: bool
    trial_penalty_parameter: float
    flag: int  # 1 = accepted Test I, 2 = accepted Test II, 3 = restart budget exceeded
    saw_CR_breakdown: bool
    saw_phase_limit_restart: bool


def check_sqpreg_inner_termination(
    iterate: SQPRegInnerIterate,
    context: SQPRegInnerContext,
    params: SQPRegTerminationParams,
) -> SQPRegTerminationDecision:
    """Check the Byrd/SQPReg termination tests and Hessian modification rule."""
    pi_prev = context.penalty_parameter
    c_norm = context.constraint_norm

    max_curv_term = jnp.maximum(iterate.dWd_half, params.theta * iterate.Upsilon)
    delta_m_pi_prev = -iterate.grad_fTd + pi_prev * (c_norm - iterate.r_norm)
    model_rhs = max_curv_term + params.sigma * pi_prev * jnp.maximum(c_norm, iterate.r_norm - c_norm)
    model_reduction_holds = delta_m_pi_prev >= model_rhs

    test1_residual_holds = iterate.residual_norm <= params.kappa * context.stationarity_rhs_norm
    test1_holds = model_reduction_holds & test1_residual_holds

    ck_positive = c_norm > 0.0
    test2_r_holds = iterate.r_norm <= params.epsilon * c_norm
    test2_rho_holds = iterate.rho_norm <= params.beta * c_norm
    test2_geometry_holds = (
        (iterate.dWd_half >= params.theta * iterate.Upsilon)
        | (params.psi * iterate.nu >= iterate.Upsilon)
    )
    test2_holds = ck_positive & test2_r_holds & test2_rho_holds & test2_geometry_holds

    denom = (1.0 - params.tau) * (c_norm - iterate.r_norm)
    safe_denom = jnp.where(denom > 0.0, denom, jnp.inf)
    pi_trial = (iterate.grad_fTd + max_curv_term) / safe_denom

    modify_hessian = (
        (~test1_holds)
        & (~test2_holds)
        & (delta_m_pi_prev < model_rhs)
        & (iterate.dWd_half < params.theta * iterate.Upsilon)
        & (params.psi * iterate.nu < iterate.Upsilon)
    )

    # Priority encoding of termination logic.
    status = jnp.where(test1_holds, 1, jnp.where(test2_holds, 2, jnp.where(modify_hessian, 3, 0)))

    accept = test1_holds | test2_holds
    trial_penalty_parameter = jnp.where(test2_holds, pi_trial, pi_prev)

    return SQPRegTerminationDecision(
        accept=accept,
        status=status,
        trial_penalty_parameter=trial_penalty_parameter,
    )


def pack_primal_dual(d: jax.Array, delta: jax.Array) -> jax.Array:
    return jnp.concatenate([d, delta])


def unpack_primal_dual(z: jax.Array, primal_dim: int) -> tuple[jax.Array, jax.Array]:
    return z[:primal_dim], z[primal_dim:]


def make_kkt_operator(
    Hv: Callable[[jax.Array], jax.Array],
    J: jax.Array,
    primal_dim: int,
) -> Callable[[jax.Array, float], jax.Array]:
    """Build the packed shifted KKT product [W, J^T; J, 0] with W = 10^{-mu} H + (1- 10^{-mu})I."""

    def K(z: jax.Array, mu: float) -> jax.Array:
        primal, dual = unpack_primal_dual(z, primal_dim)
        tau = 10.0 ** (-mu)
        W_primal = tau * Hv(primal) + (1.0 - tau) * primal
        JT_dual = J.T @ dual
        J_primal = J @ primal
        return pack_primal_dual(W_primal + JT_dual, J_primal)

    return K


def init_cr_state(
    K: Callable[[jax.Array, float], jax.Array],
    rhs: jax.Array,
    x0: jax.Array,
    mu: float,
    breakdown_tol: float = 1e-14,
) -> CRState:
    """Initialise one CR phase on the fixed shifted KKT system."""
    Kx0 = K(x0, mu)
    r0 = rhs - Kx0
    Ar0 = K(r0, mu)
    rAr0 = jnp.vdot(r0, Ar0)
    residual_norm0 = jnp.linalg.norm(r0)
    r_norm_sq0 = residual_norm0**2

    valid0 = (
        (jnp.abs(rAr0) > breakdown_tol * r_norm_sq0)
        & jnp.isfinite(rAr0)
        & jnp.isfinite(residual_norm0)
        & jnp.all(jnp.isfinite(x0))
        & jnp.all(jnp.isfinite(r0))
        & jnp.all(jnp.isfinite(Ar0))
    )

    return CRState(
        x=x0,
        r=r0,
        p=r0,
        Ap=Ar0,
        rAr=rAr0,
        residual_norm=residual_norm0,
        phase_iter=0,
        valid=valid0,
    )


def cr_step(
    state: CRState,
    K: Callable[[jax.Array, float], jax.Array],
    mu: float,
    breakdown_tol: float = 1e-16,
) -> CRState:
    """Take one CR step on the fixed shifted KKT system."""
    Ap_norm_sq = jnp.vdot(state.Ap, state.Ap)

    safe_Ap_norm_sq = jnp.where(jnp.abs(Ap_norm_sq) > breakdown_tol, Ap_norm_sq, 1.0)
    safe_rAr = jnp.where(jnp.abs(state.rAr) > breakdown_tol, state.rAr, 1.0)

    alpha = state.rAr / safe_Ap_norm_sq

    x_next = state.x + alpha * state.p
    r_next = state.r - alpha * state.Ap

    Ar_next = K(r_next, mu)
    rAr_next = jnp.vdot(r_next, Ar_next)

    beta = rAr_next / safe_rAr
    p_next = r_next + beta * state.p
    Ap_next = Ar_next + beta * state.Ap
    residual_norm_next = jnp.linalg.norm(r_next)

    valid_next = (
        state.valid
        & (jnp.abs(state.rAr) > breakdown_tol)
        & (jnp.abs(Ap_norm_sq) > breakdown_tol)
        & jnp.isfinite(alpha)
        & jnp.isfinite(beta)
        & jnp.isfinite(rAr_next)
        & jnp.isfinite(residual_norm_next)
        & jnp.all(jnp.isfinite(x_next))
        & jnp.all(jnp.isfinite(r_next))
        & jnp.all(jnp.isfinite(p_next))
        & jnp.all(jnp.isfinite(Ap_next))
    )

    return CRState(
        x=x_next,
        r=r_next,
        p=p_next,
        Ap=Ap_next,
        rAr=rAr_next,
        residual_norm=residual_norm_next,
        phase_iter=state.phase_iter + 1,
        valid=valid_next,
    )


def _jacobian_upper_norm_bound(J: jax.Array) -> float:
    """Upper bound for ||J||_2^2 used in the practical forms of (6.6) and (6.7)."""
    m, n = J.shape
    norm1 = jnp.linalg.norm(J, ord=1)
    norm_inf = jnp.linalg.norm(J, ord=jnp.inf)
    return jnp.minimum(n * norm1**2, m * norm_inf**2)


def _compute_nu_upsilon(Jd: jax.Array, d: jax.Array, J_norm_sq_upper: float) -> tuple[float, float]:
    """Practical forms of (6.6) and (6.7)."""
    safe_denom = jnp.where(J_norm_sq_upper > 0.0, J_norm_sq_upper, jnp.inf)
    nu = jnp.vdot(Jd, Jd) / safe_denom
    Upsilon = jnp.maximum(0.0, jnp.vdot(d, d) - nu)
    return nu, Upsilon


def _build_inner_iterate(
    cr_state: CRState,
    *,
    primal_dim: int,
    grad_f: jax.Array,
    rhs_primal: jax.Array,
    rhs_dual: jax.Array,
    J: jax.Array,
    J_norm_sq_upper: float,
) -> SQPRegInnerIterate:
    """Reconstruct the paper diagnostics for the current CR iterate."""
    d, delta = unpack_primal_dual(cr_state.x, primal_dim)

    # The packed CR residual is [rho; r] = rhs - K[x] for the shifted KKT system.
    # Hence
    #   Wd + J^T delta = rhs_primal - rho,
    #   Jd            = rhs_dual - r.
    rho, r = unpack_primal_dual(cr_state.r, primal_dim)
    JTdelta = J.T @ delta
    Wd = rhs_primal - rho - JTdelta
    Jd = rhs_dual - r

    nu, Upsilon = _compute_nu_upsilon(Jd=Jd, d=d, J_norm_sq_upper=J_norm_sq_upper)

    return SQPRegInnerIterate(
        d=d,
        delta=delta,
        rho=rho,
        r=r,
        grad_fTd=jnp.vdot(grad_f, d),
        dWd_half=0.5 * jnp.vdot(d, Wd),
        step_norm_sq=jnp.vdot(d, d),
        nu=nu,
        Upsilon=Upsilon,
        rho_norm=jnp.linalg.norm(rho),
        r_norm=jnp.linalg.norm(r),
        residual_norm=cr_state.residual_norm,
    )


########## Implementation Code ##########
def get_reg_sqp_inner_solver(
    lhvp,
    *,
    termination_params,
    max_phase_iters=None,
    max_restarts,
    mu_init,
    mu_increase_factor,
    breakdown_tol,
):
    """Return a jitted regularized SQP inner solver."""

    @jax.jit
    def inner_solver(params, duals, grad_f, c, J, lag_grad, penalty_parameter, objective_data, constraint_data):
        Lhvp = lhvp(
            params,
            duals=duals,
            objective_data=objective_data,
            constraint_data=constraint_data,
            vals="Lhvp",
        )

        return sqpreg_inner_solve(
            grad_f=grad_f,
            grad_lag=lag_grad,
            c=c,
            J=J,
            Hv=Lhvp,
            penalty_parameter=penalty_parameter,
            termination_params=termination_params,
            max_phase_iters=max_phase_iters,
            max_restarts=max_restarts,
            mu_init=mu_init,
            mu_increase_factor=mu_increase_factor,
            breakdown_tol=breakdown_tol,
        )

    return inner_solver


def sqpreg_inner_solve(
    *,
    grad_f: jax.Array,
    grad_lag: jax.Array,
    c: jax.Array,
    J: jax.Array,
    Hv: Callable[[jax.Array], jax.Array],
    penalty_parameter: float,
    termination_params: SQPRegTerminationParams,
    max_phase_iters: int | None,
    max_restarts: int,
    mu_init: float,
    mu_increase_factor: float,
    z0: jax.Array | None = None,
    breakdown_tol: float = 1e-14,
) -> SQPRegSolveResult:
    """Solve the regularized SQP primal-dual inner system with CR and warm restarts.

    Notes
    -----
    - The primal KKT right-hand side uses the Lagrangian gradient, while the
      model-reduction terms in the termination tests use the objective gradient.
    - The Jacobian J is supplied explicitly so that nu and Upsilon can be updated
      each iteration using the practical forms of (6.6) and (6.7).
    - On a Hessian modification, the CR recurrence is restarted from the current
      iterate with the updated mu.
    - If CR encounters an invalid algebraic state and the current iterate is not
      accepted, we treat this as a CR-breakdown restart and increase mu.
    - If one CR phase reaches the phase-iteration cap without acceptance, we
      treat the phase as numerically stale and restart with larger regularization.
    """
    primal_dim = grad_f.shape[0]
    dual_dim = c.shape[0]

    if max_phase_iters is None:
        max_phase_iters = primal_dim + dual_dim

    if z0 is None:
        z = pack_primal_dual(jnp.zeros_like(grad_f), jnp.zeros_like(c))
    else:
        z = z0

    rhs_primal = -grad_lag
    rhs_dual = -c
    rhs = pack_primal_dual(rhs_primal, rhs_dual)
    J_norm_sq_upper = _jacobian_upper_norm_bound(J)

    K = make_kkt_operator(Hv=Hv, J=J, primal_dim=primal_dim)
    context = SQPRegInnerContext(
        penalty_parameter=penalty_parameter,
        stationarity_rhs_norm=jnp.linalg.norm(rhs),
        constraint_norm=jnp.linalg.norm(c),
    )

    init_loop_state = SQPRegInnerLoopState(
        cr_state=init_cr_state(K=K, rhs=rhs, x0=z, mu=mu_init, breakdown_tol=breakdown_tol),
        mu=mu_init,
        total_iter=0,
        restart_count=0,
        done=False,
        accepted=False,
        flag=0,
        trial_penalty_parameter=penalty_parameter,
        saw_CR_breakdown=False,
        saw_phase_limit_restart=False,
    )

    def cond_fn(state: SQPRegInnerLoopState):
        return ~state.done

    # body_fn control flow
    #
    #   current CR state
    #         |
    #         v
    #   build iterate diagnostics
    #         |
    #         v
    #   evaluate termination tests
    #         |
    #         v
    #   accept?
    #    /   \
    #  yes    no
    #  |       |
    #  v       v
    # terminate   Hessian restart requested?
    #                /          \
    #              yes           no
    #              |              |
    #              v              v
    #        increase mu      CR breakdown restart needed?
    #        restart CR          /               \
    #                           yes               no
    #                           |                  |
    #                           v                  v
    #                  mark saw_CR_breakdown   phase limit restart needed?
    #                  increase mu restart        /               \
    #                                            yes               no
    #                                            |                  |
    #                                            v                  v
    #                               mark saw_phase_limit_restart   take CR step
    #                               increase mu restart
    def body_fn(state: SQPRegInnerLoopState) -> SQPRegInnerLoopState:
        iterate = _build_inner_iterate(
            state.cr_state,
            primal_dim=primal_dim,
            grad_f=grad_f,
            rhs_primal=rhs_primal,
            rhs_dual=rhs_dual,
            J=J,
            J_norm_sq_upper=J_norm_sq_upper,
        )
        decision = check_sqpreg_inner_termination(iterate, context, termination_params)

        def accept_branch(state):
            return SQPRegInnerLoopState(
                cr_state=state.cr_state,
                mu=state.mu,
                total_iter=state.total_iter,
                restart_count=state.restart_count,
                done=True,
                accepted=True,
                flag=decision.status,
                trial_penalty_parameter=decision.trial_penalty_parameter,
                saw_CR_breakdown=state.saw_CR_breakdown,
                saw_phase_limit_restart=state.saw_phase_limit_restart,
            )

        def nonaccept_branch(state):
            hessian_restart_requested = decision.status == 3
            cr_breakdown_restart_requested = ~state.cr_state.valid
            phase_limit_restart_requested = state.cr_state.phase_iter >= max_phase_iters

            def hessian_restart_branch(state):
                new_restart_count = state.restart_count + 1

                def restart_limit_branch(state):
                    return SQPRegInnerLoopState(
                        cr_state=state.cr_state,
                        mu=state.mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=True,
                        accepted=False,
                        flag=3,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=state.saw_CR_breakdown,
                        saw_phase_limit_restart=state.saw_phase_limit_restart,
                    )

                def do_restart_branch(state):
                    new_mu = mu_increase_factor + state.mu
                    new_cr_state = init_cr_state(
                        K=K,
                        rhs=rhs,
                        x0=state.cr_state.x,
                        mu=new_mu,
                        breakdown_tol=breakdown_tol,
                    )
                    return SQPRegInnerLoopState(
                        cr_state=new_cr_state,
                        mu=new_mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=False,
                        accepted=False,
                        flag=0,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=state.saw_CR_breakdown,
                        saw_phase_limit_restart=state.saw_phase_limit_restart,
                    )

                return jax.lax.cond(new_restart_count > max_restarts, restart_limit_branch, do_restart_branch, state)

            def cr_breakdown_restart_branch(state):
                new_restart_count = state.restart_count + 1

                def restart_limit_branch(state):
                    return SQPRegInnerLoopState(
                        cr_state=state.cr_state,
                        mu=state.mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=True,
                        accepted=False,
                        flag=3,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=True,
                        saw_phase_limit_restart=state.saw_phase_limit_restart,
                    )

                def do_restart_branch(state):
                    new_mu = mu_increase_factor + state.mu
                    new_cr_state = init_cr_state(
                        K=K,
                        rhs=rhs,
                        x0=state.cr_state.x,
                        mu=new_mu,
                        breakdown_tol=breakdown_tol,
                    )
                    return SQPRegInnerLoopState(
                        cr_state=new_cr_state,
                        mu=new_mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=False,
                        accepted=False,
                        flag=0,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=True,
                        saw_phase_limit_restart=state.saw_phase_limit_restart,
                    )

                return jax.lax.cond(new_restart_count > max_restarts, restart_limit_branch, do_restart_branch, state)

            def phase_limit_restart_branch(state):
                new_restart_count = state.restart_count + 1

                def restart_limit_branch(state):
                    return SQPRegInnerLoopState(
                        cr_state=state.cr_state,
                        mu=state.mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=True,
                        accepted=False,
                        flag=3,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=state.saw_CR_breakdown,
                        saw_phase_limit_restart=True,
                    )

                def do_restart_branch(state):
                    new_mu = state.mu + mu_increase_factor
                    new_cr_state = init_cr_state(
                        K=K,
                        rhs=rhs,
                        x0=state.cr_state.x,
                        mu=new_mu,
                        breakdown_tol=breakdown_tol,
                    )
                    return SQPRegInnerLoopState(
                        cr_state=new_cr_state,
                        mu=new_mu,
                        total_iter=state.total_iter,
                        restart_count=new_restart_count,
                        done=False,
                        accepted=False,
                        flag=0,
                        trial_penalty_parameter=decision.trial_penalty_parameter,
                        saw_CR_breakdown=state.saw_CR_breakdown,
                        saw_phase_limit_restart=True,
                    )

                return jax.lax.cond(new_restart_count > max_restarts, restart_limit_branch, do_restart_branch, state)

            def continue_branch(state):
                new_cr_state = cr_step(
                    state=state.cr_state,
                    K=K,
                    mu=state.mu,
                    breakdown_tol=breakdown_tol,
                )
                return SQPRegInnerLoopState(
                    cr_state=new_cr_state,
                    mu=state.mu,
                    total_iter=state.total_iter + 1,
                    restart_count=state.restart_count,
                    done=False,
                    accepted=False,
                    flag=0,
                    trial_penalty_parameter=decision.trial_penalty_parameter,
                    saw_CR_breakdown=state.saw_CR_breakdown,
                    saw_phase_limit_restart=state.saw_phase_limit_restart,
                )

            return jax.lax.cond(
                hessian_restart_requested,
                hessian_restart_branch,
                lambda state: jax.lax.cond(
                    cr_breakdown_restart_requested,
                    cr_breakdown_restart_branch,
                    lambda state: jax.lax.cond(
                        phase_limit_restart_requested,
                        phase_limit_restart_branch,
                        continue_branch,
                        state,
                    ),
                    state,
                ),
                state,
            )

        return jax.lax.cond(decision.accept, accept_branch, nonaccept_branch, state)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_loop_state)
    final_d, final_delta = unpack_primal_dual(final_state.cr_state.x, primal_dim)
    final_rho, final_r = unpack_primal_dual(final_state.cr_state.r, primal_dim)

    return SQPRegSolveResult(
        d=final_d,
        delta=final_delta,
        r=final_r,
        rho=final_rho,
        mu_final=final_state.mu,
        total_iter=final_state.total_iter,
        restart_count=final_state.restart_count,
        accepted=final_state.accepted,
        trial_penalty_parameter=final_state.trial_penalty_parameter,
        flag=final_state.flag,
        saw_CR_breakdown=final_state.saw_CR_breakdown,
        saw_phase_limit_restart=final_state.saw_phase_limit_restart,
    )
