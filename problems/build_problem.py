from typing import Any, Callable

import jax

from utils.utils import flatten, unflatten, make_func_flat
from utils.differ import AutodiffOracle


class Problem:
    """One fully specified optimization problem instance.

    The constructor accepts the native objective/constraint interface and then
    prepares the flattened solver-facing fields needed by the optimizers.
    """

    def __init__(
        self,
        *,
        name: str,
        x0: Any,
        objective: Callable,
        constraint: Callable,
        objective_data: Any = None,
        constraint_data: Any = None,
        test: Callable | None = None,
        obj_has_aux: bool = False,
    ) -> None:
        self.name = name
        self.objective_data = objective_data
        self.constraint_data = constraint_data

        x0_flat, treedef, leaf_shapes = flatten(x0)
        self.x0 = x0_flat
        self._treedef = treedef
        self._leaf_shapes = leaf_shapes

        self.objective = make_func_flat(objective, treedef, leaf_shapes)
        self.constraint = make_func_flat(constraint, treedef, leaf_shapes)
        self.test = make_func_flat(test, treedef, leaf_shapes)

        self.oracle = AutodiffOracle(
            self.objective,
            self.constraint,
            obj_has_aux=obj_has_aux,
        )

    def reconstruct(self, params_flat: jax.Array) -> Any:
        return unflatten(params_flat, self._treedef, self._leaf_shapes)