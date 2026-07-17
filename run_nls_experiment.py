import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml

from optimizers.SQPRegularised import SQPRegularizedParameters, SQPRegularized
from solvers.RegSQPInnerSolver import SQP_DEFAULT_TERMINATION_PARAMS
from runners.basic_runner import BasicExperimentRunner, SolverConfig
from runners.cli_utils import add_common_run_args, resolve_solver_names, expand_solver_sweep
from optimizers.PrimalDualNewtonCRFactorisation import (
    PrimalDualNewtonCRFactorisation,
    PrimalDualNewtonParameters,
)
from optimizers.AugmentedLagrangianLBFGS import (
    AugmentedLagrangianLBFGS,
    AugmentedLagrangianLBFGSParameters,
)
from problems.binary_classification_nls_problem import (
    make_binary_classification_problem_from_arrays,
)

RUNTIME_PROFILES = {
    "standard": {
        "epsp": 1e-5,
        "epsc": 1e-5,
        "merit_param": 1e-6,
        "merit_shrink" : 1.,
        "tan_inner_tol": 1e-1,
        "reg_value": 0.0,
        "subproblem_rel_tol" : 1e-7,
        "max_iters": 2000,
        "max_time": 60,
    },
    "large": {
        "epsp": 1e-5,
        "epsc": 1e-5,
        "merit_param": 1e-6,
        "merit_shrink" : 1.,
        "tan_inner_tol": 1e-2,
        "reg_value": 0.0,
        "subproblem_rel_tol" : 1e-5,
        "max_iters": 5000,
        "max_time": 60,
    },
}

def make_solver_catalog(
    *,
    n_features: int,
    n_constraints: int,
    epsp: float,
    epsc: float,
    tan_inner_tol: float,
    subproblem_rel_tol: float,
    merit_param: float,
    merit_shrink: float,
    reg_value: float,
    max_iters: int,
    max_time: float,
) -> dict[str, SolverConfig]:
    
    # common line search parameters
    slope_rtol = 1e-4
    max_backtracking_steps = 20
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
        merit_shrink=merit_shrink,
        merit_param_update_factor=0.1,
        fw_track=True,
        max_forward_tracking_steps=max_backtracking_steps,
        tan_reg=reg_value,
    )

    pdn_ls = PrimalDualNewtonParameters(dual_update="ls", **pdn_common)
    pdn_zero = PrimalDualNewtonParameters(dual_update="zero", **pdn_common)

    solver_configs = {
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
    }
    
    # These parameters reflect the choices in Byrd et al. Line search parameters are set to common values.
    sqp_params = SQPRegularizedParameters(
        epsp=epsp,
        epsc=epsc,
        max_iters=max_iters,
        max_time=max_time,

        slope_rtol=slope_rtol,
        max_backtracking_steps=max_backtracking_steps,
        shrink_factor=shrink_factor,

        initial_penalty_parameter=1e-1,
        inner_termination_params=SQP_DEFAULT_TERMINATION_PARAMS,
        inner_max_phase_iters=(n_features + n_constraints),
        inner_max_restarts=20,
        inner_mu_init=0.,
        inner_mu_increase_factor=1.,
        inner_breakdown_tol=1e-16,
    )
    solver_configs["sqp_default"] = SolverConfig("sqp_default", SQPRegularized, sqp_params)

    alm_hparams = AugmentedLagrangianLBFGSParameters(
        epsp=epsp,
        epsc=epsc,
        max_iters=max_iters,
        max_time=max_time,
        # LBFGS parameters
        memory_size=20,
        use_lbfgs_scaling=True,
        # Line search parameters
        max_linesearch_steps=20,
        max_learning_rate=1.0,
        slope_rtol=1e-4,
        curv_rtol=0.9,
        increase_factor=2.0,
        abs_tol=0.0,
        verbose_linesearch=False,
        # Penalty and subproblem parameters
        initial_penalty=1.0,
        max_penalty=1e4,
        penalty_increase_factor=2.0,
        feasibility_improvement_factor=0.25,
        subproblem_rel_tol=subproblem_rel_tol,
        subproblem_tol_shrink_factor=0.5,
        subproblem_tol_lower_bound=None,
    )

    solver_configs["alm_lbfgs_default"] = SolverConfig(
        name="alm_lbfgs_default",
        solver_cls=AugmentedLagrangianLBFGS,
        hyperparameters=alm_hparams,
        )

    return solver_configs

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run binary classification optimisation experiments."
    )
    add_common_run_args(parser)
    parser.add_argument("dataset", nargs="?", default="digits", choices=["digits", "mnist", "fashion"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_lin_constraints", dest="n_lin_constraints", type=int, default=20)
    parser.add_argument("--runtime_profile", choices=sorted(RUNTIME_PROFILES),default="standard", help="Named runtime profile. To experiment with parameters create a new test profile.",)
    parser.add_argument("--max_time", type=float, default=None, help="Override max_time in runtime profile.")
    parser.add_argument("--max_iters", type=int, default=None, help="Override max_iters in runtime profile.")
    return parser


def build_problem(args):
    if args.dataset == "digits":
        print("Loading Digits dataset.")
        X, y = load_digits(return_X_y=True)
        y = y % 2
    elif args.dataset == "mnist":
        print("Loading MNIST dataset.")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, data_home=os.path.join("datasets", "mnist"), parser="liac-arff") # don't need pandas for data loading.
        y = y.astype(int) % 2
    elif args.dataset == "fashion":
        print("Loading fashion MNIST dataset")
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, data_home=os.path.join("datasets", "fashion_mnist"), 
        parser="liac-arff") # don't need pandas for data loading.
        y = y.astype(int) % 2
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    _, n_features = X.shape
    print("Dataset shape:", X.shape)
    print("labels: ", jnp.unique(y))
    print("Seed: ", args.seed)
    key = jax.random.PRNGKey(args.seed)
    key_a, key_ref, key_prob, key_x0 = jax.random.split(key, 4)

    x_ref = jax.random.normal(key_ref, (n_features + 1,))/jnp.sqrt(n_features+1) # standardised reference solution.
    x0 = jax.random.normal(key_x0, (n_features + 1,))
    
    A = jax.random.normal(key_a, (args.n_lin_constraints, n_features))

    problem = make_binary_classification_problem_from_arrays(
        X=X,
        y=y,
        A=A,
        x0=x0,
        standardize=True,
        reference_params=x_ref,
        split_key=key_prob,
        test_fraction=args.test_size,
    )

    return problem, n_features

def main():
    parser = build_parser()
    args = parser.parse_args()

    problem, n_features = build_problem(args)
    print("Ball constraint radius squared: ", problem.constraint_data["radius_sq"])
    runtime = RUNTIME_PROFILES[args.runtime_profile]
    if args.max_time is not None:
        runtime["max_time"] = args.max_time
    if args.max_iters is not None:
        runtime["max_iters"] = args.max_iters

    solver_catalog = make_solver_catalog(
        n_features=n_features, 
        n_constraints=args.n_lin_constraints+1, 
        epsp=float(runtime["epsp"]),
        epsc=float(runtime["epsc"]),
        merit_param=float(runtime["merit_param"]),
        merit_shrink=float(runtime["merit_shrink"]),
        tan_inner_tol=float(runtime["tan_inner_tol"]),
        subproblem_rel_tol=float(runtime["subproblem_rel_tol"]),
        max_iters=int(runtime["max_iters"]),
        max_time=float(runtime["max_time"]),
        reg_value=float(runtime["reg_value"]),
    )
    solver_names = resolve_solver_names(args, solver_catalog)
    configs = [solver_catalog[name] for name in solver_names]
    configs = expand_solver_sweep(
        configs,
        sweep_param=args.sweep_param,
        sweep_values=args.sweep_values,
    )

    experiment_runner = BasicExperimentRunner()
    save_name = args.save_name if args.save_name is not None else f"nls_{args.dataset}"
    if args.n_lin_constraints != 20:
        save_name += f"_{args.n_lin_constraints}"
    if args.sweep_param is not None:
        save_name += f"__sweep_{args.sweep_param}"
    results = experiment_runner.run_problem(
        problem,
        configs,
        verbose=not args.quiet,
        save_dir=args.save_dir,
        save_name=save_name,
        experiment_metadata={"dataset": args.dataset, "seed": args.seed, "n_lin_constraints": args.n_lin_constraints},
    )


if __name__ == "__main__":
    main()