from __future__ import annotations

from typing import Any, Callable, NamedTuple
import time

import jax
import jax.numpy as jnp
import optax

from optimizers.Optimizer import Optimizer
from utils.differ import AutodiffOracle, Oracle


class AugmentedLagrangianStepStatistics(NamedTuple):
    # iteration type: 0 = primal LBFGS step, 1 = primal step + gated outer update,
    # 2 = primal step + gated outer update + penalty increase
    step_type: int
    # line search statistics for primal iterations
    step_size: float | None
    ls_iters: int | None
    ls_flag: bool
    curvature_warning: bool
    decrease_warning: bool
    # iterate statistics (recorded at the start of the iteration)
    constraint_norm: float
    objective_val: float
    primal_gradient_norm: float
    dual_norm: float
    penalty: float
    al_val: float
    al_grad_norm: float
    subproblem_tol: float
    step_norm: float


class AugmentedLagrangianLBFGSParameters(NamedTuple):
    # termination parameters
    epsp: float
    epsc: float
    max_iters: int
    max_time: float
    
    # optax LBFGS parameters
    memory_size: int
    use_lbfgs_scaling: bool
    # optax zoom line search parameters
    max_linesearch_steps: int
    max_learning_rate: float
    slope_rtol: float
    curv_rtol: float
    increase_factor: float
    abs_tol: float
    verbose_linesearch: bool
    # penalty parameters
    initial_penalty: float
    max_penalty: float
    penalty_increase_factor: float
    feasibility_improvement_factor: float
    # subproblem tolerance for outer multiplier / penalty updates.
    subproblem_rel_tol: float 
    subproblem_tol_shrink_factor: float
    subproblem_tol_lower_bound: float | None = None


class _PreStepQuantities(NamedTuple):
    f_val: Any
    c_val: Any
    primal_gradient: Any
    al_val: Any
    al_grad: Any
    constraint_norm: Any
    primal_gradient_norm: Any
    al_grad_norm: Any
    dual_norm: Any
    aux: dict[str, Any]


class _LineSearchDiagnostics(NamedTuple):
    step_size: float | None
    ls_iters: int | None
    curvature_warning: bool
    decrease_warning : bool


class _StepComputationResult(NamedTuple):
    update: Any
    new_opt_state: Any
    step_norm: Any

class _State(NamedTuple):
    params: Any
    duals: Any
    penalty: Any
    last_outer_constraint_norm: Any
    subproblem_tol: Any
    opt_state: Any


def make_augmented_lagrangian_helpers(oracle: Oracle):
    @jax.jit
    def augmented_lagrangian_value(
        params,
        *,
        duals,
        penalty,
        objective_data,
        constraint_data,
    ):
        f_val = oracle.obj(params, objective_data=objective_data, vals="f")[0][0]
        c_val = oracle.cons(params, constraint_data=constraint_data, vals="c")
        return f_val + jnp.vdot(duals, c_val) + 0.5 * penalty * jnp.vdot(c_val, c_val)

    if oracle.obj_has_aux:

        @jax.jit
        def augmented_lagrangian_fg(
            params,
            *,
            duals,
            penalty,
            objective_data,
            constraint_data,
        ):
            (f_val, gradf), aux = oracle.obj(
                params, objective_data=objective_data, vals="fg"
            )
            c_val = oracle.cons(params, constraint_data=constraint_data, vals="c")
            JTv = oracle.cons(params, constraint_data=constraint_data, vals="JTv")

            shifted_duals = duals + penalty * c_val
            grad_al = gradf + JTv(shifted_duals)
            primal_gradient = gradf + JTv(duals)
            al_val = f_val + jnp.vdot(duals, c_val) + 0.5 * penalty * jnp.vdot(c_val, c_val)

            return f_val, c_val, primal_gradient, al_val, grad_al, aux

    else:

        @jax.jit
        def augmented_lagrangian_fg(
            params,
            *,
            duals,
            penalty,
            objective_data,
            constraint_data,
        ):
            f_val, gradf = oracle.obj(params, objective_data=objective_data, vals="fg")[0]
            c_val = oracle.cons(params, constraint_data=constraint_data, vals="c")
            JTv = oracle.cons(params, constraint_data=constraint_data, vals="JTv")

            shifted_duals = duals + penalty * c_val
            grad_al = gradf + JTv(shifted_duals)
            primal_gradient = gradf + JTv(duals)
            al_val = f_val + jnp.vdot(duals, c_val) + 0.5 * penalty * jnp.vdot(c_val, c_val)

            return f_val, c_val, primal_gradient, al_val, grad_al, {}

    return augmented_lagrangian_value, augmented_lagrangian_fg

def _extract_linesearch_info(opt_state, tol) -> _LineSearchDiagnostics:

    step_size = optax.tree_utils.tree_get(opt_state, "learning_rate")
    ls_iters = optax.tree_utils.tree_get(opt_state, "num_linesearch_steps")
    decrease_error = optax.tree_utils.tree_get(opt_state, "decrease_error")
    curvature_error = optax.tree_utils.tree_get(opt_state, "curvature_error")
    step_size = float(step_size)
    ls_iters = int(ls_iters)

    curvature_warning = curvature_error > tol
    decrease_warning = decrease_error > tol
  
    return _LineSearchDiagnostics(
        step_size=step_size,
        ls_iters=ls_iters,
        curvature_warning=curvature_warning,
        decrease_warning=decrease_warning,
    )


def _extract_zoom_final_state(opt_state):
    final_al_val = optax.tree_utils.tree_get(opt_state, "value")
    final_al_grad = optax.tree_utils.tree_get(opt_state, "grad")
    return final_al_val, final_al_grad


def make_optax_lbfgs_solver(hyperparameters: AugmentedLagrangianLBFGSParameters):
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=hyperparameters.max_linesearch_steps,
        max_learning_rate=hyperparameters.max_learning_rate,
        tol=hyperparameters.abs_tol,
        increase_factor=hyperparameters.increase_factor,
        slope_rtol=hyperparameters.slope_rtol,
        curv_rtol=hyperparameters.curv_rtol,
        verbose=hyperparameters.verbose_linesearch,
        initial_guess_strategy="one",
    )

    solver = optax.lbfgs(
        learning_rate=None,
        memory_size=hyperparameters.memory_size,
        scale_init_precond=hyperparameters.use_lbfgs_scaling,
        linesearch=linesearch,
    )
    return solver


class AugmentedLagrangianLBFGS(Optimizer):
    """One-loop augmented Lagrangian method with one Optax LBFGS step per iteration."""

    def __init__(self, oracle: Oracle, hyperparameters: AugmentedLagrangianLBFGSParameters):
        super().__init__()
        self.oracle = oracle
        self.hyperparameters = hyperparameters

        self.al_value, self.al_fg = make_augmented_lagrangian_helpers(oracle)
        self.opt = make_optax_lbfgs_solver(hyperparameters)
        self._inner_step = self._make_inner_lbfgs_step()
        self.subproblem_rel_tol = hyperparameters.subproblem_rel_tol
        self.subproblem_tol_shrink_factor = hyperparameters.subproblem_tol_shrink_factor
        if hyperparameters.subproblem_tol_lower_bound is None:
            self.subproblem_tol_lower_bound = hyperparameters.epsp
        else:
            self.subproblem_tol_lower_bound = hyperparameters.subproblem_tol_lower_bound

    def _make_inner_lbfgs_step(self):
        @jax.jit
        def inner_lbfgs_step(
            params,
            opt_state,
            *,
            al_val,
            al_grad,
            duals,
            penalty,
            objective_data,
            constraint_data,
        ):
            # Close over value function parameters to avoid recompilation during line search iterations.
            # value_fn = self._make_value_fn(duals, penalty, objective_data, constraint_data)
            value_fn = lambda p: self.al_value(
                p,
                duals=duals,
                penalty=penalty,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
            
            updates, new_opt_state = self.opt.update(
                al_grad,
                opt_state,
                params,
                value=al_val,
                grad=al_grad,
                value_fn=value_fn,
            )
            step_norm = optax.tree.norm(updates)
            return _StepComputationResult(
                update=updates,
                new_opt_state=new_opt_state,
                step_norm=step_norm,
            )

        return inner_lbfgs_step

    def _evaluate_current_quantities(
        self,
        params,
        duals,
        penalty,
        objective_data,
        constraint_data,
    ) -> _PreStepQuantities:
        f_val, c_val, primal_gradient, al_val, al_grad, aux = self.al_fg(
            params,
            duals=duals,
            penalty=penalty,
            objective_data=objective_data,
            constraint_data=constraint_data,
        )
        constraint_norm = jnp.linalg.norm(c_val)
        primal_gradient_norm = jnp.linalg.norm(primal_gradient)
        al_grad_norm = jnp.linalg.norm(al_grad)
        dual_norm = jnp.linalg.norm(duals)
        return _PreStepQuantities(
            f_val=f_val,
            c_val=c_val,
            primal_gradient=primal_gradient,
            al_val=al_val,
            al_grad=al_grad,
            constraint_norm=constraint_norm,
            primal_gradient_norm=primal_gradient_norm,
            al_grad_norm=al_grad_norm,
            dual_norm=dual_norm,
            aux=aux,
        )

    def step(
        self,
        state: _State,
        objective_data: tuple[jax.Array, ...] | None = None,
        constraint_data: tuple[jax.Array, ...] | None = (),
    ):
        pre = self._evaluate_current_quantities(
            state.params,
            state.duals,
            state.penalty,
            objective_data,
            constraint_data,
        )
        terminate = jnp.logical_and(
            pre.primal_gradient_norm <= self.hyperparameters.epsp,
            pre.constraint_norm <= self.hyperparameters.epsc,
        )

        if bool(terminate):
            stats = AugmentedLagrangianStepStatistics(
                step_type=0,
                step_size=None,
                ls_iters=None,
                ls_flag=True,
                curvature_warning=False,
                decrease_warning=False,
                constraint_norm=float(pre.constraint_norm),
                objective_val=float(pre.f_val),
                primal_gradient_norm=float(pre.primal_gradient_norm),
                dual_norm=float(pre.dual_norm),
                penalty=float(state.penalty),
                al_val=float(pre.al_val),
                al_grad_norm=float(pre.al_grad_norm),
                subproblem_tol=float(state.subproblem_tol),
                step_norm=0.0,
            )
            return state, True, stats, pre.aux

        
        step_result = self._inner_step(
            state.params,
            state.opt_state,
            al_val=pre.al_val,
            al_grad=pre.al_grad,
            duals=state.duals,
            penalty=state.penalty,
            objective_data=objective_data,
            constraint_data=constraint_data,
        )

        final_al_val, final_al_grad = _extract_zoom_final_state(step_result.new_opt_state)
        final_al_grad_norm = optax.tree.norm(final_al_grad)
        ls_info = _extract_linesearch_info(step_result.new_opt_state, self.hyperparameters.abs_tol)
        ls_flag = not (ls_info.decrease_warning or ls_info.curvature_warning)
        
        new_params = optax.apply_updates(state.params, step_result.update)

        new_opt_state = step_result.new_opt_state
        new_duals = state.duals
        new_penalty = state.penalty
        subproblem_tol = state.subproblem_tol
        new_last_outer_constraint_norm = state.last_outer_constraint_norm
        step_type = 0


        if final_al_grad_norm is not None and bool(final_al_grad_norm <= subproblem_tol):
            new_c_val = self.oracle.cons(new_params, constraint_data=constraint_data, vals="c")
            new_constraint_norm = jnp.linalg.norm(new_c_val)
            sufficient_progress = (
                new_constraint_norm
                <= self.hyperparameters.feasibility_improvement_factor * state.last_outer_constraint_norm
            )
            new_duals = state.duals + state.penalty * new_c_val
            new_last_outer_constraint_norm = new_constraint_norm

            # Shrink the subproblem tolerance for the next iteration, but not below the specified lower bound.
            subproblem_tol = jnp.maximum(state.subproblem_tol * self.subproblem_tol_shrink_factor, self.subproblem_tol_lower_bound)
            step_type = 1

            if not bool(sufficient_progress) and new_constraint_norm > self.hyperparameters.epsc:
                new_penalty = jnp.minimum(
                    self.hyperparameters.max_penalty,
                    self.hyperparameters.penalty_increase_factor * state.penalty,
                )
                step_type = 2
            
            new_opt_state = self.opt.init(new_params) # reset the LBFGS memory after an outer update to avoid using stale curvature information.

        new_state = _State(
            params=new_params,
            duals=new_duals,
            penalty=new_penalty,
            last_outer_constraint_norm=new_last_outer_constraint_norm,
            opt_state=new_opt_state,
            subproblem_tol=subproblem_tol,
        )

        stats = AugmentedLagrangianStepStatistics(
            step_type=step_type,
            step_size=ls_info.step_size,
            ls_iters=ls_info.ls_iters,
            ls_flag=ls_flag,
            decrease_warning=ls_info.decrease_warning,
            curvature_warning=ls_info.curvature_warning,
            constraint_norm=float(pre.constraint_norm),
            objective_val=float(pre.f_val),
            primal_gradient_norm=float(pre.primal_gradient_norm),
            dual_norm=float(pre.dual_norm),
            penalty=float(state.penalty),
            # Record al val and grad at end of iterations. This helps with subproblem diagnostics, since we can check these values against subproblem_tol.
            al_val=float(final_al_val),
            al_grad_norm=float(final_al_grad_norm),
            subproblem_tol=float(state.subproblem_tol),
            step_norm=float(step_result.step_norm),
        )
        return new_state, False, stats, pre.aux

    def optimize(
        self,
        x0: jax.Array,
        objective_data: tuple[jax.Array, ...] | None,
        constraint_data: tuple[jax.Array, ...] | None = (),
        dual0: jax.Array | None = None,
        test: Callable[[jax.Array], dict] | None = None,
        verbose: bool = False,
    ) -> tuple[jax.Array, jax.Array, str, dict]:
        self.reset_history()

        dtype = x0.dtype
        c0 = self.oracle.cons(x0, constraint_data=constraint_data, vals="c")
        dual0 = jnp.zeros_like(c0) if dual0 is None else dual0
        opt_state0 = self.opt.init(x0)
        penalty0 = jnp.asarray(self.hyperparameters.initial_penalty, dtype=dtype)
        al_grad0 = self.al_fg(x0, duals=dual0, penalty=penalty0, objective_data=objective_data, constraint_data=constraint_data)[4]
        subproblem_tol0 = jnp.maximum(self.hyperparameters.subproblem_rel_tol * jnp.linalg.norm(al_grad0), self.subproblem_tol_lower_bound)

        state = _State(
            params=x0,
            duals=dual0,
            penalty=penalty0,
            last_outer_constraint_norm=jnp.linalg.norm(c0),
            opt_state=opt_state0,
            subproblem_tol=subproblem_tol0,
        )

        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # warmup: not recorded, washes out compile time for timing.
        warmup_out = self.step(
            state,
            objective_data=objective_data,
            constraint_data=constraint_data,
        )
        jax.block_until_ready(warmup_out)
        total_time = 0.0

        for i in range(self.hyperparameters.max_iters):
            if i % max(1, self.hyperparameters.max_iters // 10) == 0:
                vprint(f"iteration {i}/{self.hyperparameters.max_iters}")

            test_metrics = test(state.params) if test is not None else {}

            t0 = time.perf_counter()
            step_out = self.step(
                state,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
            jax.block_until_ready(step_out)
            t1 = time.perf_counter()
            step_time = t1 - t0

            new_state, terminate, statistics, aux_metrics = step_out
            aux_metrics = dict(aux_metrics)
            aux_metrics["time"] = total_time
            total_time += step_time
            self.store_history(statistics, aux_metrics, test_metrics)
            self.history["final_total_time"] = total_time

            if terminate:
                vprint(f"Converged in {i} iterations")
                return state.params, state.duals, "converged", self.history

            if statistics.ls_flag is False:
                vprint(f"Warning: Line search warning at iteration {i} with {statistics.ls_iters} iterations and decrease_warning={statistics.decrease_warning} curvature_warning={statistics.curvature_warning}.")
                # return state.params, state.duals, "ls_failed", self.history

            state = new_state

            if total_time >= self.hyperparameters.max_time:
                vprint(f"Maximum time of {self.hyperparameters.max_time} seconds reached.")
                return state.params, state.duals, "max_time", self.history

        vprint("Maximum iterations reached without convergence.")
        return state.params, state.duals, "max_iters", self.history
