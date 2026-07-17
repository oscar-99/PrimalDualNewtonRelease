from typing import Any
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from problems.fully_connected import FullyConnectedNetwork


@dataclass(frozen=True)
class ModelSpec:
    """Raw data describing one stock PINN model architecture."""
    family: str
    in_dim: int
    out_dim: int
    layers: tuple[int, ...]
    activation: str
    seed: int
    use_skip: bool
    use_fourier_features: bool
    num_fourier_features: int
    include_raw_input: bool
    normalize_input: bool

    input_mean: jax.Array | None 
    input_std: jax.Array | None 
    input_std_eps: float 
    
    use_xavier_init: bool
    xavier_gain: float

def build_model(spec: ModelSpec) -> Any:
    # Can later be extended to more model families. For now just builds FullyConnectedNetworks.

    activation_map = {
        "silu": jax.nn.silu,
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
    }

    try:
        activation = activation_map[spec.activation]
    except KeyError as exc:
        available = ", ".join(sorted(activation_map))
        raise KeyError(
            f"Unknown activation '{spec.activation}'. Available activations: {available}"
        ) from exc

    if spec.family == "fully_connected":
        return FullyConnectedNetwork(
            spec.in_dim,
            spec.out_dim,
            list(spec.layers),
            key=jax.random.PRNGKey(spec.seed),
            activation=activation,
            use_skip=spec.use_skip,
            use_fourier_features=spec.use_fourier_features,
            num_fourier_features=spec.num_fourier_features,
            include_raw_input=spec.include_raw_input,
            normalize_input=spec.normalize_input,
            input_mean=spec.input_mean,
            input_std=spec.input_std,
            input_std_eps=spec.input_std_eps,
            use_xavier_init=spec.use_xavier_init,
            xavier_gain=spec.xavier_gain,
        )
    else:
        raise ValueError(f"Unknown model family '{spec.family}'")