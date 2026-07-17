from typing import NamedTuple

import jax
import jax.numpy as jnp


class LineSearchResult(NamedTuple):
    step_size: jax.Array
    phi_trial: jax.Array
    num_fun_evals: jax.Array
    accepted: jax.Array

class _ForwardTrackingState(NamedTuple):
    accepted_step_size: jax.Array
    accepted_phi_trial: jax.Array
    trial_step_size: jax.Array
    num_forward_steps: jax.Array
    continue_forward: jax.Array

def get_armijo_line_search(
    value_fun,
    *,
    max_backtracking_steps: jax.typing.ArrayLike,
    slope_rtol: jax.typing.ArrayLike = 1e-4,
    decrease_factor: jax.typing.ArrayLike = 0.5,
    max_learning_rate: jax.typing.ArrayLike = 1.0,
    atol: jax.typing.ArrayLike = 0.0,
    enable_forward_tracking: bool = False,
    max_forward_tracking_steps: jax.typing.ArrayLike | None = None,
):
    """Construct a generic jittable Armijo line search on value function ``phi``.

    Parameters
    ----------
    value_fun
        Scalar-valued callable with signature
        ``value_fun(params, **value_fun_kwargs)``.
    max_backtracking_steps, slope_rtol, decrease_factor, max_learning_rate, atol
        Standard Armijo line-search hyperparameters.
    max_learning_rate
        Initial trial step size/max step size for backtracking.
    atol
        Absolute tolerance for Armijo condition.
    enable_forward_tracking
        If ``True``, the returned line search may use the forward-tracking
        branch when the per-call runtime flag also enables it.
    max_forward_tracking_steps
        Maximum number of forward expansions attempted after accepting the
        initial trial step. Defaults to ``max_backtracking_steps``.

    Returns
    -------
    Callable
        Jitted line-search function with signature
        ``line_search(params, search_direction, phi_k, directional_derivative,
            forward_tracking_enabled, value_fun_kwargs)``.
        The ``value_fun_kwargs`` pytree must have a fixed structure across calls.
    """
    max_backtracking_steps = jnp.asarray(max_backtracking_steps)
    slope_rtol = jnp.asarray(slope_rtol)
    decrease_factor = jnp.asarray(decrease_factor)
    max_learning_rate = jnp.asarray(max_learning_rate)
    atol = jnp.asarray(atol)
    forward_tracking_available = jnp.asarray(enable_forward_tracking)

    if max_forward_tracking_steps is None:
        max_forward_tracking_steps = max_backtracking_steps
    max_forward_tracking_steps = jnp.asarray(max_forward_tracking_steps)

    if not (0.0 < float(decrease_factor) < 1.0):
        raise ValueError("decrease_factor must lie in (0, 1) for backtracking/forward tracking.")

    @jax.jit
    def line_search(
        params,
        search_direction,
        phi_k,
        directional_derivative,
        forward_tracking_enabled,
        value_fun_kwargs
    ):
        def trial_values(step_size):
            trial_params = jax.tree.map(
                lambda x, p: x + step_size * p,
                params,
                search_direction,
            )
            return value_fun(trial_params, **value_fun_kwargs)

        def armijo_rhs(step_size):
            return phi_k + step_size * slope_rtol * directional_derivative + atol

        alpha0 = max_learning_rate
        phi0 = trial_values(alpha0)
        accepted0 = phi0 <= armijo_rhs(alpha0)

        init_state = LineSearchResult(
            step_size=alpha0,
            phi_trial=phi0,
            num_fun_evals=jnp.asarray(1, dtype=jnp.int32),
            accepted=accepted0,
        )

        def backtracking_search(state: LineSearchResult):
            def cond_fun(backtrack_state: LineSearchResult):
                return jnp.logical_and(
                    ~backtrack_state.accepted,
                    backtrack_state.num_fun_evals < max_backtracking_steps,
                )

            def body_fun(backtrack_state: LineSearchResult):
                new_alpha = backtrack_state.step_size * decrease_factor
                phi_trial = trial_values(new_alpha)
                accepted = phi_trial <= armijo_rhs(new_alpha)

                return LineSearchResult(
                    step_size=new_alpha,
                    phi_trial=phi_trial,
                    num_fun_evals=backtrack_state.num_fun_evals + 1,
                    accepted=accepted,
                )

            return jax.lax.while_loop(cond_fun, body_fun, state)

        def forward_tracking_search(state: LineSearchResult):
            init_forward_state = _ForwardTrackingState(
                accepted_step_size=state.step_size,
                accepted_phi_trial=state.phi_trial,
                trial_step_size=state.step_size,
                num_forward_steps=jnp.asarray(0, dtype=jnp.int32),
                continue_forward=jnp.asarray(True),
            )

            def cond_fun(forward_state: _ForwardTrackingState):
                return jnp.logical_and(
                    forward_state.continue_forward,
                    forward_state.num_forward_steps < max_forward_tracking_steps,
                )

            def body_fun(forward_state: _ForwardTrackingState):
                new_alpha = forward_state.trial_step_size / decrease_factor
                phi_trial = trial_values(new_alpha)
                accepted = phi_trial <= armijo_rhs(new_alpha)

                accepted_step_size = jnp.where(
                    accepted,
                    new_alpha,
                    forward_state.accepted_step_size,
                )
                accepted_phi_trial = jnp.where(
                    accepted,
                    phi_trial,
                    forward_state.accepted_phi_trial,
                )

                return _ForwardTrackingState(
                    accepted_step_size=accepted_step_size,
                    accepted_phi_trial=accepted_phi_trial,
                    trial_step_size=new_alpha,
                    num_forward_steps=forward_state.num_forward_steps + 1,
                    continue_forward=accepted,
                )

            final_forward_state = jax.lax.while_loop(
                cond_fun,
                body_fun,
                init_forward_state,
            )

            return LineSearchResult(
                step_size=final_forward_state.accepted_step_size,
                phi_trial=final_forward_state.accepted_phi_trial,
                num_fun_evals=final_forward_state.num_forward_steps+1,
                accepted=state.accepted,
            )

        do_forward_tracking = jnp.logical_and(
            forward_tracking_available,
            jnp.logical_and(forward_tracking_enabled, accepted0),
        )
        return jax.lax.cond(
            do_forward_tracking,
            forward_tracking_search,
            backtracking_search,
            init_state,
        )

    return line_search
