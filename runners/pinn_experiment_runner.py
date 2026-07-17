from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from problems.build_model import ModelSpec, build_model
from problems.build_problem import Problem
from problems.pinns.pinn_problems import build_pinn_problem
from runners.basic_runner import RunResult, Runner, SolverConfig
from runners.results_io import save_manifest, save_result


@dataclass(frozen=True)
class PINNRunSpec:
    """Fully resolved runtime spec for one PINN experiment."""

    name: str
    pinn_dim: int
    pde_param: float | None
    model_spec: ModelSpec
    n_int: int
    n_bndry: int
    eval_grid_shape: tuple[int, int]
    problem_seed: int
    obj_penalty_weight: float
    cons_penalty_weight: float
    cons_penalty_fn: str


class PINNExperimentRunner:
    """Thin wrapper for building, running, annotating, and saving PINN experiments."""

    def __init__(self, runner: Runner | None = None):
        self.runner = Runner() if runner is None else runner

    def build_problem(self, spec: PINNRunSpec) -> Problem:
        model = build_model(spec.model_spec)
        return build_pinn_problem(
            model,
            name=spec.name,
            pde_param=spec.pde_param,
            n_int=spec.n_int,
            n_bndry=spec.n_bndry,
            eval_grid_shape=spec.eval_grid_shape,
            seed=spec.problem_seed,
            obj_penalty_weight=spec.obj_penalty_weight,
            cons_penalty_weight=spec.cons_penalty_weight,
            cons_penalty_fn=spec.cons_penalty_fn,
        )

    def annotate_result(
        self,
        spec: PINNRunSpec,
        result: RunResult,
        problem: Problem,
    ) -> RunResult:
        result.metadata["experiment_metadata"] = {
            "PINNSpec": spec,
            "x_int": problem.objective_data[0],
            "x_bndry": problem.constraint_data[0],
        }
        return result

    def run_problem(
        self,
        spec: PINNRunSpec,
        configs: Sequence[SolverConfig],
        *,
        verbose: bool = False,
        save_dir: str | Path | None = None,
        save_name: str | None = None,
    ) -> list[RunResult]:
        problem = self.build_problem(spec)

        save_root: Path | None = None
        saved_solver_names: list[str] = []
        if save_dir is not None:
            resolved_save_name = save_name if save_name is not None else spec.name
            save_root = Path(save_dir) / resolved_save_name
            save_root.mkdir(parents=True, exist_ok=True)
            save_manifest(save_root, save_name=resolved_save_name, solver_names=[])

        def after_each(result: RunResult) -> None:
            self.annotate_result(spec, result, problem)
            if save_root is None:
                return
            save_result(result, save_root / result.solver_name)
            saved_solver_names.append(result.solver_name)
            save_manifest(save_root, save_name=save_root.name, solver_names=saved_solver_names)

        return self.runner.run_many(problem, configs, verbose=verbose, after_each=after_each)
