from __future__ import annotations

from typing import Any, Sequence
import argparse

from runners.basic_runner import SolverConfig

def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help="Explicit list of solver names to run. Overrides --solver_suite.",
    )
    parser.add_argument(
        "--solver_suite",
        nargs="+",
        default=["all"],
        help="Select all solver configs whose names contain any of these substrings.",
    )
    parser.add_argument(
        "--sweep_param",
        type=str,
        default=None,
        help="Single solver hyperparameter to sweep across all selected solvers.",
    )
    parser.add_argument(
        "--sweep_values",
        type=float,
        nargs="+",
        default=None,
        help="Values for the swept solver hyperparameter.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--save_name", type=str, default=None)


def resolve_solver_names(
    args: argparse.Namespace,
    solver_catalog: dict[str, SolverConfig],
) -> list[str]:
    if args.solvers:
        missing = [name for name in args.solvers if name not in solver_catalog]
        if missing:
            raise ValueError(f"Unknown solver names: {missing}")
        return list(args.solvers)

    suite_tokens = list(args.solver_suite or ["all"])
    if "all" in suite_tokens:
        return list(solver_catalog.keys())

    selected: list[str] = []
    for name, config in solver_catalog.items():
        family = config.metadata.get("family", "")
        if any(token in name or token == family for token in suite_tokens):
            selected.append(name)

    if not selected:
        raise ValueError(
            f"No solvers matched suite tokens {suite_tokens}. "
            f"Available solvers: {list(solver_catalog.keys())}"
        )

    return selected


def _set_namedtuple_field(hparams: Any, field_name: str, value: Any) -> Any:
    if hasattr(hparams, "_fields"):
        if field_name not in hparams._fields:
            raise ValueError(
                f"Hyperparameter '{field_name}' is not a field of "
                f"{type(hparams).__name__}. "
                f"Available fields: {list(hparams._fields)}"
            )
        return hparams._replace(**{field_name: value})

    raise TypeError(
        "Expected solver hyperparameters to be a namedtuple-like object "
        f"supporting '_replace'. Got type {type(hparams).__name__}."
    )


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def expand_solver_sweep(
    configs: Sequence[SolverConfig],
    *,
    sweep_param: str | None,
    sweep_values: Sequence[float] | None,
) -> list[SolverConfig]:
    if sweep_param is None:
        return list(configs)

    if not sweep_values:
        raise ValueError(
            "A sweep parameter was provided but no sweep values were given."
        )

    expanded: list[SolverConfig] = []
    for config in configs:
        for value in sweep_values:
            new_hparams = _set_namedtuple_field(
                config.hyperparameters,
                sweep_param,
                value,
            )
            value_str = _format_sweep_value(value)
            new_name = f"{config.name}__{sweep_param}_{value_str}"

            metadata = dict(config.metadata)
            metadata.update(
                {
                    "base_solver_name": config.name,
                    "sweep_param": sweep_param,
                    "sweep_value": value,
                }
            )

            expanded.append(
                SolverConfig(
                    name=new_name,
                    solver_cls=config.solver_cls,
                    hyperparameters=new_hparams,
                    solver_kwargs=dict(config.solver_kwargs),
                    optimize_kwargs=dict(config.optimize_kwargs),
                    metadata=metadata,
                )
            )

    return expanded