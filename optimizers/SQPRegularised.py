from typing import Callable, NamedTuple
import time

import jax
import jax.numpy as jnp
import optax

from optimizers.Optimizer import Optimizer
from utils.differ import Oracle
from utils.line_search import get_armijo_line_search
from solvers.RegSQPInnerSolver import SQPRegTerminationParams, get_reg_sqp_inner_solver


class SQPRegularizedStepStatistics(NamedTuple):
    inner_total_iters: int | None
    inner_restart_count: int | None
    inner_accepted: bool | None
    inner_flag: int | None
    inner_mu_final: float | None
    saw_cr_breakdown: bool | None
    saw_phase_limit_restart: bool | None

    step_size: float | None
    ls_iters: int | None
    ls_flag: bool | None

    objective_val: float
    constraint_norm: float
    primal_gradient_norm: float
    dual_norm: float

    primal_step_norm: float | None
    dual_step_norm: float | None

    merit_func_val: float
    merit_parameter: float
    directional_derivative: float | None
    inner_r_norm: float
    inner_rho_norm: float


class SQPRegularizedParameters(NamedTuple):
    epsp: float
    epsc: float
    max_iters: int
    max_time: float

    slope_rtol: float
    max_backtracking_steps: int
    shrink_factor: float

    initial_penalty_parameter: float
    inner_termination_params: SQPRegTerminationParams
    inner_max_phase_iters: int
    inner_max_restarts: int
    inner_mu_init: float
    inner_mu_increase_factor: float
    inner_breakdown_tol: float = 1e-14


class SQPRegularized(Optimizer):
    """Non-jitted outer regularized SQP method using the Byrd inner solver.

    The outer loop mirrors the structure of the existing primal-dual Newton code:
    each iteration evaluates derivatives, calls the inner solver to obtain a
    primal-dual step, updates the merit parameter, performs an Armijo line
    search on the primal variables, and then updates both primal and dual
    variables using the accepted step size.
    """

    def __init__(
        self,
        oracle: Oracle,
        hyperparameters: SQPRegularizedParameters,
    ):
        super().__init__()
        self.oracle = oracle
        self.hyperparameters = hyperparameters

        self.inner_solver = get_reg_sqp_inner_solver(
            self.oracle.lag, 
            termination_params=hyperparameters.inner_termination_params,
            max_phase_iters=hyperparameters.inner_max_phase_iters,
            max_restarts=hyperparameters.inner_max_restarts,
            mu_init=hyperparameters.inner_mu_init,
            mu_increase_factor=hyperparameters.inner_mu_increase_factor,
            breakdown_tol=hyperparameters.inner_breakdown_tol,
        )

        @jax.jit
        def merit_function(
            params,
            objective_data=None,
            constraint_data=None,
            merit_parameter=None,
        ):
            fk = self.oracle.obj(params, objective_data=objective_data)[0][0]
            ck = self.oracle.cons(params, constraint_data=constraint_data, vals="c")
            return fk + merit_parameter * jnp.linalg.norm(ck)

        self.line_search = get_armijo_line_search(
            merit_function,
            max_backtracking_steps=self.hyperparameters.max_backtracking_steps,
            slope_rtol=self.hyperparameters.slope_rtol,
            decrease_factor=self.hyperparameters.shrink_factor,
            max_learning_rate=1.0,
            atol=0.0,
            enable_forward_tracking=False,
        )

    def step(
        self,
        params: jax.Array,
        dual: jax.Array,
        merit_param : jax.Array,
        objective_data=None,
        constraint_data=None,
    ) -> tuple[jax.Array, 
               jax.Array, 
               jax.Array,
               bool, 
               str | None, 
               SQPRegularizedStepStatistics, 
               dict]:
        
        if self.oracle.obj_has_aux:
            (fk, grad_f), aux = self.oracle.obj(
                params,
                objective_data=objective_data,
                vals="fg",
            )
        else:
            ((fk, grad_f),) = self.oracle.obj(
                params,
                objective_data=objective_data,
                vals="fg",
            )
            aux = {}

        ck = self.oracle.cons(params, constraint_data=constraint_data, vals="c").flatten()
        Jk = self.oracle.cons(params, constraint_data=constraint_data, vals="J")

        cknorm = jnp.linalg.norm(ck)
        grad_lag = (grad_f + Jk.T @ dual).flatten()
        lag_grad_norm = jnp.linalg.norm(grad_lag)
        dual_norm = jnp.linalg.norm(dual)

        terminate = (
            cknorm <= self.hyperparameters.epsc
            and lag_grad_norm <= self.hyperparameters.epsp
        )

        primal_update = jnp.zeros_like(params)
        dual_update = jnp.zeros_like(dual)
        alphak = None
        ls_iters = None
        ls_flag = None
        exit_status = None

        inner_solver_result = None
        directional_derivative = None
        merit_val = fk + merit_param * cknorm

        if not terminate:
            inner_solver_result = self.inner_solver(params, dual, grad_f, ck, Jk, grad_lag, merit_param, objective_data, constraint_data)
            if not inner_solver_result.accepted:
                statistics = SQPRegularizedStepStatistics(
                    inner_total_iters=int(inner_solver_result.total_iter),
                    inner_restart_count=int(inner_solver_result.restart_count),
                    inner_accepted=bool(inner_solver_result.accepted),
                    inner_flag=int(inner_solver_result.flag),
                    inner_mu_final=float(inner_solver_result.mu_final),
                    saw_cr_breakdown=bool(inner_solver_result.saw_CR_breakdown),
                    saw_phase_limit_restart=bool(inner_solver_result.saw_phase_limit_restart),
                    step_size=None,
                    ls_iters=None,
                    ls_flag=None,
                    objective_val=float(fk),
                    constraint_norm=float(cknorm),
                    primal_gradient_norm=float(lag_grad_norm),
                    dual_norm=float(dual_norm),
                    primal_step_norm=float(jnp.linalg.norm(inner_solver_result.d)),
                    dual_step_norm=float(jnp.linalg.norm(inner_solver_result.delta)),
                    merit_func_val=float(merit_val),
                    merit_parameter=float(merit_param),
                    directional_derivative=None,
                    inner_r_norm=float(jnp.linalg.norm(inner_solver_result.r)),
                    inner_rho_norm=float(jnp.linalg.norm(inner_solver_result.rho)),
                )
                return primal_update, dual_update, merit_param, False, "inner_failed", statistics, aux

            # update merit parameter for type 2 step
            if inner_solver_result.flag == 2:
                merit_param = jnp.maximum(
                    merit_param,
                    jnp.asarray(inner_solver_result.trial_penalty_parameter, dtype=params.dtype),
                )
            merit_val = fk + merit_param * cknorm

            primal_step = inner_solver_result.d
            dual_step = inner_solver_result.delta
            directional_derivative = jnp.vdot(grad_f, primal_step) + merit_param * (
                jnp.linalg.norm(inner_solver_result.r) - cknorm
            )

            value_fun_kwargs = {
                "objective_data": objective_data,
                "constraint_data": constraint_data,
                "merit_parameter": merit_param,
            }
            ls_result = self.line_search(
                params,
                primal_step,
                merit_val,
                directional_derivative,
                jnp.asarray(False),
                value_fun_kwargs,
            )

            alphak = ls_result.step_size
            ls_iters = ls_result.num_fun_evals
            ls_flag = bool(ls_result.accepted)

            if not ls_flag:
                statistics = SQPRegularizedStepStatistics(
                    inner_total_iters=int(inner_solver_result.total_iter),
                    inner_restart_count=int(inner_solver_result.restart_count),
                    inner_accepted=bool(inner_solver_result.accepted),
                    inner_flag=int(inner_solver_result.flag),
                    inner_mu_final=float(inner_solver_result.mu_final),
                    saw_cr_breakdown=bool(inner_solver_result.saw_CR_breakdown),
                    saw_phase_limit_restart=bool(inner_solver_result.saw_phase_limit_restart),
                    step_size=float(alphak),
                    ls_iters=int(ls_iters),
                    ls_flag=ls_flag,
                    objective_val=float(fk),
                    constraint_norm=float(cknorm),
                    primal_gradient_norm=float(lag_grad_norm),
                    dual_norm=float(dual_norm),
                    primal_step_norm=float(jnp.linalg.norm(primal_step)),
                    dual_step_norm=float(jnp.linalg.norm(dual_step)),
                    merit_func_val=float(merit_val),
                    merit_parameter=float(merit_param),
                    directional_derivative=float(directional_derivative),
                    inner_r_norm=float(jnp.linalg.norm(inner_solver_result.r)),
                    inner_rho_norm=float(jnp.linalg.norm(inner_solver_result.rho)),
                )
                return primal_update, dual_update, merit_param, False, "ls_failed", statistics, aux

            primal_update = primal_step * alphak
            dual_update = dual_step * alphak

        statistics = SQPRegularizedStepStatistics(
            inner_total_iters=None if inner_solver_result is None else int(inner_solver_result.total_iter),
            inner_restart_count=None if inner_solver_result is None else int(inner_solver_result.restart_count),
            inner_accepted=None if inner_solver_result is None else bool(inner_solver_result.accepted),
            inner_flag=None if inner_solver_result is None else int(inner_solver_result.flag),
            inner_mu_final=None if inner_solver_result is None else float(inner_solver_result.mu_final),
            saw_cr_breakdown=None if inner_solver_result is None else bool(inner_solver_result.saw_CR_breakdown),
            saw_phase_limit_restart=None if inner_solver_result is None else bool(inner_solver_result.saw_phase_limit_restart),
            step_size=None if alphak is None else float(alphak),
            ls_iters=ls_iters,
            ls_flag=ls_flag,
            objective_val=float(fk),
            constraint_norm=float(cknorm),
            primal_gradient_norm=float(lag_grad_norm),
            dual_norm=float(dual_norm),
            primal_step_norm=None if inner_solver_result is None else float(jnp.linalg.norm(inner_solver_result.d)),
            dual_step_norm=None if inner_solver_result is None else float(jnp.linalg.norm(inner_solver_result.delta)),
            merit_func_val=float(merit_val),
            merit_parameter=float(merit_param),
            directional_derivative=None if directional_derivative is None else float(directional_derivative),
            inner_r_norm=0.0 if inner_solver_result is None else float(jnp.linalg.norm(inner_solver_result.r)),
            inner_rho_norm=0.0 if inner_solver_result is None else float(jnp.linalg.norm(inner_solver_result.rho)),
        )
        return primal_update, dual_update, merit_param, terminate, exit_status, statistics, aux

    def optimize(
        self,
        x0: optax.Params,
        objective_data: tuple[jax.Array, ...] | None,
        constraint_data: tuple[jax.Array, ...] | None = (),
        test: Callable[[optax.Params], dict] | None = None,
        verbose: bool = False,
        duals0 = None
    ) -> tuple[optax.Params, jax.Array, str, dict]:
        self.reset_history()

        params = x0
        J0 = self.oracle.cons(params, constraint_data=constraint_data, vals="J")
        dual = duals0 if duals0 is not None else jnp.zeros(J0.shape[0])
        
        merit_param = jnp.asarray(
            self.hyperparameters.initial_penalty_parameter,
            dtype=params.dtype,
        )

        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        warmup_out = self.step(
            params,
            dual,
            merit_param,
            objective_data=objective_data,
            constraint_data=constraint_data,
        )
        jax.block_until_ready(warmup_out)
        total_time = 0.0

        for i in range(self.hyperparameters.max_iters):
            if i % max(1, self.hyperparameters.max_iters // 10) == 0:
                vprint(f"iteration {i}/{self.hyperparameters.max_iters}")
            
            test_metrics = test(params) if test is not None else {}

            t0 = time.perf_counter()
            step_out = self.step(
                params,
                dual,
                merit_param,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
            jax.block_until_ready(step_out)
            t1 = time.perf_counter()
            step_time = t1 - t0

            primal_update, dual_update, merit_param, terminate, exit_status, statistics, aux_metrics = step_out
            aux_metrics["time"] = total_time # Record the time at the start of the iteration for logging purposes
            total_time += step_time
            self.store_history(statistics, aux_metrics, test_metrics)
            self.history["final_total_time"] = total_time

            if terminate:
                vprint(f"Converged in {i} iterations")
                return params, dual, "converged", self.history

            if exit_status is not None:
                vprint(f"Terminated at iteration {i} with status '{exit_status}'.")
                return params, dual, exit_status, self.history

            params = params + primal_update
            dual = dual + dual_update

            if total_time >= self.hyperparameters.max_time:
                vprint(f"Maximum time of {self.hyperparameters.max_time} seconds reached.")
                return params, dual, "max_time", self.history


        vprint("Maximum iterations reached without convergence.")
        return params, dual, "max_iters", self.history
