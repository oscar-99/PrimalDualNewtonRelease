from typing import Callable, NamedTuple
import optax
import jax
import jax.numpy as jnp
import time
from solvers.CR import get_CR_hvp_solver
from solvers.normal_solve import get_SVD_normal_solver
from utils.differ import Oracle
from utils.line_search import get_armijo_line_search
from optimizers.Optimizer import Optimizer


class PrimalDualNewtonCRFactorisationStepStatistics(NamedTuple):
    # tangent solver statistics
    tan_inner_iters: int | None
    tan_inner_flag: int | None
    # line search statistics
    step_size: float | None
    ls_iters: int | None
    ls_flag: bool | None
    # iterate statistics
    constraint_norm: float
    objective_val: float
    proj_gradient_norm: float
    dual_norm: float | None
    normal_step_norm: float
    tangent_step_norm: float
    trial_step_norm: float
    # Jacobian diagnostics (svd)
    rank_estimate : int
    s_min : float
    s_max : float
    # Rii_min: float
    # directional derivative information
    gkvk: float
    gkpk: float | None
    # merit function information
    merit_func_val: float
    merit_parameter: float
    Dbar: float | None


class PrimalDualNewtonParameters(NamedTuple):
    # termination parameters
    epsp: float
    epsc: float
    max_iters: int
    max_time: float
    # tangent parameters
    tan_inner_tol: float
    nc_tol: float
    tan_max_inner_iters: int
    # line search parameters
    slope_rtol: float
    max_backtracking_steps: int
    shrink_factor: float
    fw_track : bool
    max_forward_tracking_steps: int
    # merit function parameters
    initial_merit_param: float
    merit_param_update_factor: float
    # Dual update
    dual_update: str  # ls, zero.
    tan_reg: float = 0.0 # tangent regularizer.
    # jacobian rank tolerance
    rank_tol: float = 1e-10
    merit_shrink: float = 1.


class PrimalDualNewtonCRFactorisation(Optimizer):
    """
    Primal-Dual Newton method for solving constrained optimization problems:
        min f(x)
        s.t. c(x) = 0

    There are two options for the dual update:
    - "ls": update the dual using a least-squares solve based on the QR factors.
    - "zero": set the dual to zero at each iteration.
    In either case, on the last iteration the least squares dual is returned for optimality verification.
    """

    def __init__(
        self,
        oracle: Oracle,
        hyperparameters: PrimalDualNewtonParameters,
    ):
        super().__init__()
        self.oracle = oracle
        self.hyperparameters = hyperparameters

        self.tangent_solver = get_CR_hvp_solver(
            self.oracle.lag,
            tol=self.hyperparameters.tan_inner_tol,
            nc_tol=self.hyperparameters.nc_tol,
            max_iter=self.hyperparameters.tan_max_inner_iters,
            regularizer=self.hyperparameters.tan_reg,
        )

        self.normal_solver = get_SVD_normal_solver(self.oracle, rank_tol=hyperparameters.rank_tol)

        @jax.jit
        def exact_projection(Omega, v):
            return v - Omega @ (Omega.T @ v)
        self.projection = exact_projection

        @jax.jit
        def merit_function(params, objective_data=None, constraint_data=None, merit_parameter=None):
            fk = self.oracle.obj(params, objective_data=objective_data)[0][0]
            ck = self.oracle.cons(params, constraint_data=constraint_data, vals="c")
            return fk + merit_parameter * jnp.linalg.norm(ck)

        self.ls_armijo = get_armijo_line_search(
            merit_function,
            max_backtracking_steps=self.hyperparameters.max_backtracking_steps,
            slope_rtol=self.hyperparameters.slope_rtol,
            decrease_factor=self.hyperparameters.shrink_factor,
            max_learning_rate=1.0,
            atol=0.,
            enable_forward_tracking=self.hyperparameters.fw_track,
            max_forward_tracking_steps=self.hyperparameters.max_forward_tracking_steps,
        )

        if self.hyperparameters.dual_update == "ls":
            self.dual_updater = jax.jit(self.least_squares_dual)
        elif self.hyperparameters.dual_update == "zero":
            self.dual_updater = jax.jit(lambda U, s, Omega, gradf: jnp.zeros(U.shape[0]))
        else:
            raise ValueError(f"Invalid dual update: {self.hyperparameters.dual_update}")

    def merit_parameter_trial(self, vkgk, constraint_norm):
        """Helper for updating the merit function."""
        return vkgk / (
            (1 - self.hyperparameters.merit_param_update_factor) * constraint_norm
        )

    def least_squares_dual(
        self,
        U: jax.Array,
        s: jax.Array,
        Omega: jax.Array,
        gradf: jax.Array,
    ):
        sigma_max = jnp.max(s)
        scale = jnp.maximum(1.0, sigma_max)
        keep = s > self.hyperparameters.rank_tol * scale

        safe_inv_s = jnp.where(keep, 1.0 / s, 0.0)

        return -(U @ (safe_inv_s * (Omega.T @ gradf)))

    def step(
        self,
        params: jax.Array,
        merit_param : jax.Array,
        objective_data: tuple[jax.Array, ...] | None,
        constraint_data: tuple[jax.Array, ...] | None,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        bool,
        PrimalDualNewtonCRFactorisationStepStatistics,
        dict,
    ]:
        """
        Step is a "pure" function that performs an iteration of primal-dual Newton. It returns the primal update, dual update, new merit parameter, termination flag, and statistics.
        """
        if self.oracle.obj_has_aux:
            (fk, gk), aux = self.oracle.obj(
                params,
                objective_data=objective_data,
                vals="fg",
            )
        else:
            ((fk, gk),) = self.oracle.obj(
                params,
                objective_data=objective_data,
                vals="fg",
            )
            aux = {}

        ck = self.oracle.cons(params, constraint_data=constraint_data, vals="c")
        cknorm = jnp.linalg.norm(ck)

        # Normal solve.
        normal_result = self.normal_solver(
            params,
            ck,
            constraint_data=constraint_data,
        )
        # Disable normal step and merit parameter update if constraint optimal.
        if cknorm > self.hyperparameters.epsc:
            vk = normal_result.normal_step
            gkvk = jnp.vdot(gk, vk)
            pi_trial = self.merit_parameter_trial(gkvk, cknorm)
        else:
            vk = jnp.zeros_like(params)
            gkvk = jnp.array(0.0)
            pi_trial = jnp.array(0.0)

        merit_param = jnp.maximum(pi_trial, merit_param)

        # Allow merit parameter to shrink if constraint optimal to allow more progress on objective. Don't let it drop below initial value to avoid numerical issues with very small merit parameters.
        if cknorm <= self.hyperparameters.epsc:
            merit_param = jnp.asarray(
                jnp.maximum(self.hyperparameters.merit_shrink*merit_param, self.hyperparameters.initial_merit_param), dtype=params.dtype
            )

        Omegak = normal_result.Omega # Projected Jacobian range basis truncated by the estimated rank.
        Projkgk = self.projection(Omegak, gk)
        Projkgk_norm = jnp.linalg.norm(Projkgk)

        wk = jnp.zeros_like(params)
        pk = jnp.zeros_like(params)
        tan_inner_iters = None
        tan_inner_flag = None
        ls_iters = None
        ls_flag = None
        alphak = None
        gkpk = None
        Dbar = None

        dual = self.dual_updater(normal_result.U, normal_result.s, Omegak, gk) # dual performs its own truncation.
        dual_norm = float(jnp.linalg.norm(dual))

        # Tangent component computation.
        if Projkgk_norm > self.hyperparameters.epsp:
            tangent_result = self.tangent_solver(
                params,
                -gk,
                Omegak,
                duals=dual,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
            wk = tangent_result.step
            tan_inner_iters = tangent_result.num_iter
            tan_inner_flag = tangent_result.flag

            if tangent_result.flag == 0 or tangent_result.flag == 3:
                raise ValueError(
                    "CR solver did not converge. "
                    f"Flag: {tangent_result.flag}, iters: {tangent_result.num_iter}."
                )

        terminate = (
            Projkgk_norm <= self.hyperparameters.epsp
            and cknorm <= self.hyperparameters.epsc
        )

        phik = fk + merit_param * cknorm

        if not terminate:
            pk = vk + wk
            gkpk = jnp.vdot(gk, pk)
            # if constraint optimal then v_k is zero. implies ||Jv +c|| = ||c|| so we can drop the second term in Dbar.
            if cknorm > self.hyperparameters.epsc:
                Dbar = gkpk + merit_param * (normal_result.residual_norm - cknorm)
            else:
                Dbar = gkpk

            value_fun_kwargs = {"objective_data": objective_data,
                    "constraint_data": constraint_data,
                    "merit_parameter": merit_param}
            fw_track = jnp.asarray(tan_inner_flag == 1) # if npc forward track.
            ls_results = self.ls_armijo(
                params,
                pk,
                phik,
                Dbar,
                fw_track,
                value_fun_kwargs)

            alphak = ls_results.step_size
            ls_iters = ls_results.num_fun_evals
            ls_flag = ls_results.accepted

        update = pk * alphak if alphak is not None else jnp.zeros_like(params)

        if terminate and self.hyperparameters.dual_update == "zero":
            # On last iteration return least squares dual for zero dual update. 
            dual = self.least_squares_dual(normal_result.U, normal_result.s, Omegak, gk)

        statistics = PrimalDualNewtonCRFactorisationStepStatistics(
            tan_inner_iters=tan_inner_iters,
            tan_inner_flag=tan_inner_flag,
            step_size=None if alphak is None else float(alphak),
            ls_iters=ls_iters,
            ls_flag=ls_flag,
            constraint_norm=float(cknorm),
            objective_val=float(fk),
            proj_gradient_norm=float(Projkgk_norm),
            dual_norm=float(dual_norm),
            normal_step_norm=float(jnp.linalg.norm(vk)),
            tangent_step_norm=float(jnp.linalg.norm(wk)),
            trial_step_norm=float(jnp.linalg.norm(pk)),
            s_max=float(normal_result.sigma_max),
            s_min=float(normal_result.sigma_min),
            rank_estimate=int(normal_result.rank_estimate),
            gkvk=float(gkvk),
            gkpk=None if gkpk is None else float(gkpk),
            merit_func_val=float(phik),
            merit_parameter=float(merit_param),
            Dbar=None if Dbar is None else float(Dbar),
        )
        return update, dual, merit_param, terminate, statistics, aux

    def optimize(
        self,
        x0: optax.Params,
        objective_data: tuple[jax.Array, ...] | None,
        constraint_data: tuple[jax.Array, ...] | None = (),
        test: Callable[[optax.Params], dict] | None = None,
        verbose: bool = False,
    ) -> tuple[optax.Params, jax.Array, str, dict]:
        self.reset_history()
        
        params = x0
        merit_param = jnp.asarray(self.hyperparameters.initial_merit_param, dtype=params.dtype)

        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        
        # warmup: not recorded, washes up compile time.
        warmup_out = self.step(params, merit_param, objective_data=objective_data, constraint_data=constraint_data)
        jax.block_until_ready(warmup_out)
        total_time = 0.

        for i in range(self.hyperparameters.max_iters):
            if i % (max(1, self.hyperparameters.max_iters // 10)) == 0:
                vprint("iteration {i}/{max_iters}".format(i=i, max_iters=self.hyperparameters.max_iters))

            test_metrics = test(params) if test is not None else {}

            t0 = time.perf_counter()
            step_out = self.step(
                params,
                merit_param,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
            jax.block_until_ready(step_out)
            t1 = time.perf_counter()
            step_time = t1 - t0
            
            update, dual, merit_param, terminate, statistics, aux_metrics = step_out
            aux_metrics["time"] = total_time # Record the time at the start of the iteration for logging purposes
            total_time += step_time
            self.store_history(statistics, aux_metrics, test_metrics)
            self.history["final_total_time"] = total_time

            if terminate:
                vprint(f"Converged in {i} iterations")
                return params, dual, "converged", self.history
            
            if not statistics.ls_flag:
                vprint(f"Warning: Line search failed at iteration {i}.")
                # return params, dual, "ls_failed", self.history

            # update parameters.
            params = params + update
            if total_time >= self.hyperparameters.max_time:
                vprint(f"Maximum time of {self.hyperparameters.max_time} seconds reached.")
                return params, dual, "max_time", self.history

        vprint("Maximum iterations reached without convergence.")
        return params, dual, "max_iters", self.history
