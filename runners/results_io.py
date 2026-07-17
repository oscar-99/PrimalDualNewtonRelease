from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from problems.build_model import ModelSpec, build_model
from runners.basic_runner import RunResult


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        # Handle namedtuple instances before generic tuple handling.
        if hasattr(obj, "_asdict"):
            return {str(k): _to_jsonable(v) for k, v in obj._asdict().items()}
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (jax.Array, np.ndarray)):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()

    if hasattr(obj, "__dict__"):
        return {
            str(k): _to_jsonable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return repr(obj)


def _history_to_numpy_dict(history: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    for key, values in history.items():
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            values = [values]

        cleaned = [np.nan if v is None else v for v in values]
        arr = np.asarray(cleaned)
        if arr.dtype == object:
            raise ValueError(
                f"History key '{key}' contains non-scalar values that cannot be converted to a numeric array."
            )
        out[key] = arr
    return out


def _numpy_dict_to_history(data: dict[str, np.ndarray]) -> dict[str, list[Any]]:
    return {k: v.tolist() for k, v in data.items()}


def _filter_dataclass_kwargs(dataclass_type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data.items() if k in allowed}


def save_manifest(out_dir: str | Path, *, save_name: str, solver_names: list[str]) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "save_name": save_name,
        "solvers": list(solver_names),
    }

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def save_result(result: RunResult, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history_np = _history_to_numpy_dict(result.history)
    np.savez(out_path / "history.npz", **history_np)

    if result.dual_star is not None:
        np.save(out_path / "dual_star.npy", np.asarray(result.dual_star))

    params_saved_as = None
    if result.params_star_native is not None:
        try:
            eqx.tree_serialise_leaves(out_path / "params_star.eqx", result.params_star_native)
            params_saved_as = "equinox"
        except Exception:
            try:
                np.save(out_path / "params_star_flat.npy", np.asarray(result.params_star_flat()))
                params_saved_as = "flat_numpy"
            except Exception:
                params_saved_as = None

    experiment_metadata = result.metadata.get("experiment_metadata", {})
    solver_metadata = result.metadata.get("solver_metadata", {})

    meta = {
        "problem_name": result.problem_name,
        "solver_name": result.solver_name,
        "status": result.status,
        "hyperparameters": _to_jsonable(result.hyperparameters),
        "experiment_metadata": _to_jsonable(experiment_metadata),
        "solver_metadata": _to_jsonable(solver_metadata),
        "summary": _to_jsonable(result.summary()),
        "artifacts": {
            "history": "history.npz",
            "dual_star": "dual_star.npy" if (out_path / "dual_star.npy").exists() else None,
            "params_star": "params_star.eqx" if (out_path / "params_star.eqx").exists() else None,
            "params_star_flat": "params_star_flat.npy" if (out_path / "params_star_flat.npy").exists() else None,
            "params_saved_as": params_saved_as,
        },
    }

    with open(out_path / "result_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def rebuild_predictor(result: RunResult):
    exp = result.metadata["experiment_metadata"]

    PINNSpec_dict = exp["PINNSpec"]
    model_spec_dict = PINNSpec_dict["model_spec"]

    model_spec = ModelSpec(**_filter_dataclass_kwargs(ModelSpec, model_spec_dict))
    model = build_model(model_spec)

    _, model_static = eqx.partition(model, eqx.is_array)
    return eqx.combine(result.params_star_native, model_static)


def load_result(out_dir: str | Path) -> RunResult:
    out_path = Path(out_dir)

    with open(out_path / "result_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    history_file = np.load(out_path / "history.npz", allow_pickle=False)
    history = _numpy_dict_to_history({k: history_file[k] for k in history_file.files})

    dual_path = out_path / "dual_star.npy"
    dual_star = jnp.asarray(np.load(dual_path, allow_pickle=False)) if dual_path.exists() else None

    experiment_metadata = meta.get("experiment_metadata", {})
    PINNSpec_dict = experiment_metadata.get("PINNSpec", None)
    model_spec_dict = PINNSpec_dict.get("model_spec", None) if PINNSpec_dict is not None else None

    params_star_native = None
    params_eqx_path = out_path / "params_star.eqx"
    if params_eqx_path.exists():
        if model_spec_dict is None:
            print(
                "No model spec provided; predictor/native parameters will not be reconstructed."
            )
        else:
            model_spec = ModelSpec(**_filter_dataclass_kwargs(ModelSpec, model_spec_dict))
            model = build_model(model_spec)
            params_template, _ = eqx.partition(model, eqx.is_array)

            params_star_native = eqx.tree_deserialise_leaves(
                params_eqx_path,
                params_template,
            )
    else:
        params_flat_path = out_path / "params_star_flat.npy"
        if params_flat_path.exists():
            params_star_native = jnp.asarray(np.load(params_flat_path, allow_pickle=False))

    return RunResult(
        problem_name=meta["problem_name"],
        solver_name=meta["solver_name"],
        hyperparameters=meta["hyperparameters"],
        status=meta["status"],
        params_star_native=params_star_native,
        dual_star=dual_star,
        history=history,
        metadata={
            "experiment_metadata": experiment_metadata,
            "solver_metadata": meta.get("solver_metadata", {}),
        },
    )


def load_experiment_results(
    experiment_dir: str | Path,
    *,
    solvers: list[str] | None = None,
) -> list[RunResult]:
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_path}")

    solver_dirs = sorted(
        p
        for p in exp_path.iterdir()
        if p.is_dir() and (p / "result_meta.json").exists()
    )

    if solvers is not None:
        allowed = set(solvers)
        solver_dirs = [p for p in solver_dirs if p.name in allowed]

    results = [load_result(solver_dir) for solver_dir in solver_dirs]

    if not results:
        raise ValueError(f"No saved solver results found under {exp_path}")

    return results
