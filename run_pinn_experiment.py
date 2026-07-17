import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from optimizers.AugmentedLagrangianLBFGS import (
    AugmentedLagrangianLBFGS,
    AugmentedLagrangianLBFGSParameters,
)
from optimizers.PrimalDualNewtonCRFactorisation import (
    PrimalDualNewtonCRFactorisation,
    PrimalDualNewtonParameters,
)
from optimizers.SQPRegularised import SQPRegularizedParameters, SQPRegularized
from problems.build_model import ModelSpec, build_model
from problems.pinns.pinn_problems import PINN_UTRUE_REGISTRY
from runners.basic_runner import SolverConfig
from runners.cli_utils import add_common_run_args, resolve_solver_names, expand_solver_sweep
from runners.pinn_experiment_runner import PINNExperimentRunner, PINNRunSpec
from solvers.RegSQPInnerSolver import SQP_DEFAULT_TERMINATION_PARAMS, SQPRegTerminationParams


pinn_names = list(PINN_UTRUE_REGISTRY.keys())

MODEL_VARIANTS = {
    "fc": dict(
        family="fully_connected",
        out_dim=1,
        layers=(30, 30, 30),
        activation="tanh",
        use_skip=False,
        use_fourier_features=False,
        num_fourier_features=None,
        include_raw_input=False,
        normalize_input=True,
        input_std_eps=1e-8,
        use_xavier_init=True,
        xavier_gain=1.0,
    ),
    "fourier": dict(
        family="fully_connected",
        out_dim=1,
        layers=(30, 30, 30),
        activation="tanh",
        use_skip=True,
        use_fourier_features=True,
        num_fourier_features=14,
        include_raw_input=True,
        normalize_input=True,
        input_std_eps=1e-8,
        use_xavier_init=True,
        xavier_gain=1.0,
    )
}


RUNTIME_PROFILES = {
    "standard": {
        "epsp": 1e-5,
        "epsc": 1e-5,
        "n_bndry": 32,
        "n_int": 512,
        "eval_grid_n": 64,
        "merit_param": 1e-6,
        "merit_shrink" : 1.,
        "tan_inner_tol": 1e0,
        "subproblem_rel_tol" : 1e-6,
        "max_iters": 50000,
        "max_time": 120,
        "cons_penalty_fn": "identity",
    },
    "helmholtz": {
        "epsp": 1e-5,
        "epsc": 1e-5,
        "n_bndry": 128,
        "n_int": 512,
        "eval_grid_n": 64,
        "merit_param": 1e-6,
        "merit_shrink" : 0.9,
        "tan_inner_tol": 1e3,
        "subproblem_rel_tol" : 1e3,
        "max_iters": 500000,
        "max_time": 1800,
        "cons_penalty_fn": "identity", # "identity", "square", "abs"
    },
}

def make_solver_catalog(
    *,
    n_features: int,
    n_constraints: int,
    epsp: float,
    epsc: float,
    merit_param: float,
    merit_shrink: float,
    tan_inner_tol: float,
    subproblem_rel_tol: float,
    max_iters: int,
    max_time: float,
) -> dict[str, SolverConfig]:
    slope_rtol = 1e-4
    max_backtracking_steps = 25
    shrink_factor = 0.5

    pdn_common = dict(
        epsp=epsp,
        epsc=epsc,
        max_iters=max_iters,
        max_time=max_time,
        tan_inner_tol=tan_inner_tol,
        nc_tol=0.0,
        tan_max_inner_iters=n_features,
        slope_rtol=slope_rtol,
        max_backtracking_steps=max_backtracking_steps,
        shrink_factor=shrink_factor,
        initial_merit_param=merit_param,
        merit_param_update_factor=0.1,
        fw_track=True,
        max_forward_tracking_steps=25,
        merit_shrink=merit_shrink
    )

    pdn_ls = PrimalDualNewtonParameters(dual_update="ls", **pdn_common)
    pdn_zero = PrimalDualNewtonParameters(dual_update="zero", **pdn_common)

    alm_hparams = AugmentedLagrangianLBFGSParameters(
        epsp=epsp,
        epsc=epsc,
        max_iters=max_iters,
        max_time=max_time,
        memory_size=20,
        use_lbfgs_scaling=True,
        max_linesearch_steps=20,
        max_learning_rate=1.0,
        slope_rtol=1e-4,
        curv_rtol=0.9,
        increase_factor=2.0,
        abs_tol=0.0,
        verbose_linesearch=False,
        initial_penalty=1.0,
        max_penalty=1e4,
        penalty_increase_factor=2.0,
        feasibility_improvement_factor=0.25,
        subproblem_rel_tol=subproblem_rel_tol,
        subproblem_tol_shrink_factor=0.5,
        subproblem_tol_lower_bound=None,
    )

    sqp_params = SQPRegularizedParameters(
        epsp=epsp,
        epsc=epsc,
        max_iters=max_iters,
        max_time=max_time,
        slope_rtol=slope_rtol,
        max_backtracking_steps=max_backtracking_steps,
        shrink_factor=shrink_factor,
        initial_penalty_parameter=1e-4,
        inner_termination_params=SQPRegTerminationParams(
            sigma=2e-1 * (1.0 - 1e-1), # tau(1-epsilon)
            kappa=1e-1,
            epsilon=1e-1,
            beta=1e3,
            tau=2e-1,
            theta=1e-6,
            psi=1e4,
        ),
        inner_max_phase_iters=(n_features + n_constraints),
        inner_max_restarts=20,
        inner_mu_init=0.,
        inner_mu_increase_factor=1.,
        inner_breakdown_tol=1e-16,
    )

    return {
        "pdn_zero_dual": SolverConfig(
            name="pdn_zero_dual",
            solver_cls=PrimalDualNewtonCRFactorisation,
            hyperparameters=pdn_zero,
        ),
        "pdn_ls_dual": SolverConfig(
            name="pdn_ls_dual",
            solver_cls=PrimalDualNewtonCRFactorisation,
            hyperparameters=pdn_ls,
        ),
        "alm_lbfgs_default": SolverConfig(
            name="alm_lbfgs_default",
            solver_cls=AugmentedLagrangianLBFGS,
            hyperparameters=alm_hparams,
        ),
        "sqp_default": SolverConfig(
            name="sqp_default",
            solver_cls=SQPRegularized,
            hyperparameters=sqp_params,
        ),
    }

def get_pinn_dim(pinn_name: str) -> int:
    domain = PINN_UTRUE_REGISTRY[pinn_name]["domain"]
    return 1 if domain.y_range is None else 2

def resolve_runtime_profile(args: argparse.Namespace) -> dict[str, object]:
    runtime = dict(RUNTIME_PROFILES[args.runtime_profile])
    pinn_dim = get_pinn_dim(args.pinn_name)
    if pinn_dim == 1:
        runtime["n_bndry"] = 2
    return runtime

def make_model_spec(pinn_name: str, model_variant: str, model_seed: int) -> ModelSpec:
    try:
        pinn = PINN_UTRUE_REGISTRY[pinn_name]
    except KeyError as exc:
        available = ", ".join(sorted(PINN_UTRUE_REGISTRY))
        raise KeyError(
            f"Unknown PINN name '{pinn_name}'. Available names: {available}"
        ) from exc

    domain = pinn["domain"]
    in_dim = get_pinn_dim(pinn_name)
    variant_kwargs = MODEL_VARIANTS[model_variant]

    return ModelSpec(
        seed=model_seed,
        in_dim=in_dim,
        input_mean=domain.mean,
        input_std=domain.std,
        **variant_kwargs,
    )

def make_run_spec(args: argparse.Namespace, runtime: dict[str, object]) -> PINNRunSpec:
    model_spec = make_model_spec(args.pinn_name, args.model_variant, args.model_seed)
    pinn_dim = get_pinn_dim(args.pinn_name)

    n_int = int(runtime["n_int"])
    n_bndry = int(runtime["n_bndry"])

    return PINNRunSpec(
        name=args.pinn_name,
        pinn_dim=pinn_dim,
        pde_param=args.pde_param,
        model_spec=model_spec,
        n_int=n_int,
        n_bndry=n_bndry,
        eval_grid_shape=(int(runtime["eval_grid_n"]), int(runtime["eval_grid_n"])),
        problem_seed=args.problem_seed,
        obj_penalty_weight=1.0 / n_int,
        cons_penalty_weight=1.0 / jnp.sqrt(n_bndry),
        cons_penalty_fn=str(runtime["cons_penalty_fn"]),
    )


def count_features(model) -> int:
    return sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(model)
        if hasattr(leaf, "size")
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a registered PINN problem with stock model variants and runtime profiles.",)
    add_common_run_args(parser)
    parser.add_argument("pinn_name", nargs="?", default="poisson_single_mode", help=f"Registered PINN problem name {pinn_names}.",)
    parser.add_argument("--runtime_profile", choices=sorted(RUNTIME_PROFILES),default="standard", help="Named runtime profile. To experiment with parameters create a new test profile.",)
    parser.add_argument("--pde_param", default=None, type=float, help="PINN problem parameter, e.g., for helmholtz, parameter is k.",)
    parser.add_argument("--model_variant", choices=sorted(MODEL_VARIANTS), default="fc",help="Stock model variant to use.",)
    parser.add_argument("--problem_seed", type=int, default=0)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--max_time", type=float, default=None, help="Override max_time in runtime profile.")
    parser.add_argument("--max_iters", type=int, default=None, help="Override max_iters in runtime profile.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    runtime = resolve_runtime_profile(args)
    if args.max_time is not None:
        runtime["max_time"] = args.max_time
    if args.max_iters is not None:
        runtime["max_iters"] = args.max_iters
    spec = make_run_spec(args, runtime)

    model = build_model(spec.model_spec)
    n_features = count_features(model)
    device = jax.tree_util.tree_leaves(model)[0].device

    solver_catalog = make_solver_catalog(
        n_features=n_features,
        n_constraints=spec.n_bndry,
        epsp=float(runtime["epsp"]),
        epsc=float(runtime["epsc"]),
        merit_param=float(runtime["merit_param"]),
        merit_shrink=float(runtime["merit_shrink"]),
        tan_inner_tol=float(runtime["tan_inner_tol"]),
        subproblem_rel_tol=float(runtime["subproblem_rel_tol"]),
        max_iters=int(runtime["max_iters"]),
        max_time=float(runtime["max_time"]),
    )
    solver_names = resolve_solver_names(args, solver_catalog)
    configs = [solver_catalog[name] for name in solver_names]
    configs = expand_solver_sweep(
        configs,
        sweep_param=args.sweep_param,
        sweep_values=args.sweep_values,
    )
    if args.sweep_param is not None:
        print("Sweep parameter:", args.sweep_param)
        print("Sweep values:", list(args.sweep_values))

    save_name = args.save_name
    if save_name is None:
        save_name = spec.name if args.model_variant == "fc" else f"{spec.name}_{args.model_variant}"
        if args.sweep_param is not None:
            save_name = f"{save_name}__sweep_{args.sweep_param}"

    if not args.quiet:
        print("Model architecture:")
        print(model)
        print("running on device:", device)
        print("Running PINN:", spec.name)
        print("Model variant:", args.model_variant)
        print("Problem seed:", spec.problem_seed)
        print("Model seed:", args.model_seed)
        print("Number of primal parameters:", n_features)
        print(f"Number of interior points={spec.n_int}, boundary points={spec.n_bndry}")
        print("Selected solvers:", solver_names)
        print("Runtime profile:", args.runtime_profile)
        print("Resolved runtime:")
        for key in runtime.keys():
            print(f"  {key}: {runtime[key]}")
        print("Objective weight:", spec.obj_penalty_weight)
        print("Constraint weight:", spec.cons_penalty_weight)
        print("Constraint penalty fn:", spec.cons_penalty_fn)

    experiment_runner = PINNExperimentRunner()
    experiment_runner.run_problem(
        spec,
        configs,
        verbose=not args.quiet,
        save_dir=args.save_dir,
        save_name=save_name,
    )


if __name__ == "__main__":
    main()
