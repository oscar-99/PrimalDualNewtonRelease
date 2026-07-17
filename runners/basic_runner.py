from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
from jax.flatten_util import ravel_pytree

from problems.build_problem import Problem


@dataclass(frozen=True)
class SolverConfig:
    """One solver family paired with one hyperparameter choice."""

    name: str
    solver_cls: type
    hyperparameters: Any
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    optimize_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build(self, oracle: Any) -> Any:
        return self.solver_cls(
            oracle=oracle,
            hyperparameters=self.hyperparameters,
            **self.solver_kwargs,
        )


@dataclass
class RunResult:
    """Container for one completed solver run."""

    problem_name: str
    solver_name: str
    hyperparameters: Any
    status: str
    params_star_native: Any
    dual_star: jax.Array | None
    history: dict[str, list[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_iterations(self) -> int:
        if not self.history:
            return 0
        first_key = next(iter(self.history))
        return len(self.history[first_key])

    def params_star_flat(self) -> jax.Array:
        if self.params_star_native is None:
            raise ValueError("params_star_native is not available for this result.")
        flat, _ = ravel_pytree(self.params_star_native)
        return flat

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "problem": self.problem_name,
            "solver": self.solver_name,
            "status": self.status,
            "iterations": self.num_iterations,
            "total_time": self.history["final_total_time"]
            if "final_total_time" in self.history and self.history["final_total_time"]
            else None,
        }

        for key in (
            "objective_val",
            "constraint_norm",
            "proj_gradient_norm",
            "primal_gradient_norm",
            "dual_norm",
            "step_size",
            "penalty",
            "merit_parameter",
            "train_accuracy",
            "test_accuracy",
            "train_mse",
            "test_mse",
            "test_l2_error",
            "test_linf_error",
        ):
            if key in self.history and self.history[key]:
                out[key] = self.history[key][-1]

        return out


class Runner:
    """Thin orchestration layer for comparing solvers on one problem."""

    def run_solver(
        self,
        problem: Problem,
        config: SolverConfig,
        *,
        verbose: bool = False,
    ) -> RunResult:
        solver = config.build(problem.oracle)

        optimize_kwargs = dict(config.optimize_kwargs)
        optimize_kwargs.setdefault("x0", problem.x0)
        optimize_kwargs.setdefault("objective_data", problem.objective_data)
        optimize_kwargs.setdefault("constraint_data", problem.constraint_data)
        optimize_kwargs.setdefault("test", problem.test)
        optimize_kwargs.setdefault("verbose", verbose)

        params_star, dual_star, status, history = solver.optimize(**optimize_kwargs)
        params_star_native = problem.reconstruct(params_star)

        return RunResult(
            problem_name=problem.name,
            solver_name=config.name,
            hyperparameters=config.hyperparameters,
            status=status,
            params_star_native=params_star_native,
            dual_star=dual_star,
            history=history,
            metadata={
                "solver_metadata": dict(config.metadata),
            },
        )

    def run_many(
        self,
        problem: Problem,
        configs: Sequence[SolverConfig],
        *,
        verbose: bool = False,
        after_each: Callable[[RunResult], None] | None = None,
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for config in configs:
            result = self.run_solver(problem, config, verbose=verbose)
            results.append(result)
            if after_each is not None:
                after_each(result)
        return results


class BasicExperimentRunner:
    """Small helper for running and incrementally saving non-PINN experiments."""

    def __init__(self, runner: Runner | None = None):
        self.runner = Runner() if runner is None else runner

    def annotate_result(
        self,
        result: RunResult,
        *,
        experiment_metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        result.metadata.setdefault("experiment_metadata", {})
        if experiment_metadata:
            result.metadata["experiment_metadata"].update(experiment_metadata)
        return result

    def run_problem(
        self,
        problem: Problem,
        configs: Sequence[SolverConfig],
        *,
        verbose: bool = False,
        save_dir: str | Path | None = None,
        save_name: str | None = None,
        experiment_metadata: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        save_root: Path | None = None
        saved_solver_names: list[str] = []

        if save_dir is not None:
            resolved_save_name = save_name if save_name is not None else problem.name
            save_root = Path(save_dir) / resolved_save_name
            save_root.mkdir(parents=True, exist_ok=True)
            from runners.results_io import save_manifest

            save_manifest(save_root, save_name=resolved_save_name, solver_names=[])

        def after_each(result: RunResult) -> None:
            self.annotate_result(result, experiment_metadata=experiment_metadata)
            if save_root is None:
                return
            from runners.results_io import save_manifest, save_result

            save_result(result, save_root / result.solver_name)
            saved_solver_names.append(result.solver_name)
            save_manifest(save_root, save_name=save_root.name, solver_names=saved_solver_names)

        return self.runner.run_many(problem, configs, verbose=verbose, after_each=after_each)
