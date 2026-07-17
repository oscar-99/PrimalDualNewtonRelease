from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


def _xavier_normal_weight(
    weight: jax.Array,
    *,
    key: jax.Array,
    gain: float = 1.0,
) -> jax.Array:
    out_size, in_size = weight.shape
    std = gain * jnp.sqrt(2.0 / (in_size + out_size))
    return std * jax.random.normal(key, weight.shape, dtype=weight.dtype)


def init_linear_xavier_normal_zero_bias(
    model,
    *,
    key: jax.Array,
    gain: float = 1.0,
):
    """
    Reinitialize every eqx.nn.Linear in a model with Xavier normal weights
    and zero bias.

    Parameters
    ----------
    model
        Any Equinox model / pytree.
    key
        PRNG key for the reinitialization.
    gain
        Optional Xavier gain multiplier.

    Returns
    -------
    new_model
        Model with all Linear weights/biases replaced.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)

    linear_layers = [
        x
        for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear)
        if is_linear(x)
    ]

    keys = jax.random.split(key, len(linear_layers))

    new_weights = []
    new_biases = []

    for layer, subkey in zip(linear_layers, keys):
        new_weights.append(
            _xavier_normal_weight(layer.weight, key=subkey, gain=gain)
        )

        if layer.bias is None:
            new_biases.append(None)
        else:
            new_biases.append(jnp.zeros_like(layer.bias))

    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]

    model = eqx.tree_at(get_weights, model, new_weights)
    model = eqx.tree_at(get_biases, model, new_biases)
    return model


class ResidualBlock(eqx.Module):
    linear: eqx.nn.Linear
    skip_proj: eqx.nn.Linear | None
    activation: Callable
    use_skip: bool

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        key: jax.Array,
        activation: Callable = jax.nn.silu,
        use_skip: bool = False,
    ):
        k1, k2 = jax.random.split(key, 2)
        self.linear = eqx.nn.Linear(in_size, out_size, use_bias=True, key=k1)
        self.activation = activation
        self.use_skip = use_skip

        if use_skip and in_size != out_size:
            self.skip_proj = eqx.nn.Linear(in_size, out_size, use_bias=False, key=k2)
        else:
            self.skip_proj = None

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.activation(self.linear(x))

        if self.use_skip:
            skip = x if self.skip_proj is None else self.skip_proj(x)
            y = y + skip

        return y


class FullyConnectedNetwork(eqx.Module):
    hidden_blocks: list[ResidualBlock]
    output_layer: eqx.nn.Linear
    squeeze_output: bool

    use_fourier_features: bool
    fourier_frequencies: jax.Array | None
    include_raw_input: bool

    normalize_input: bool
    input_mean: jax.Array | None
    input_std: jax.Array | None
    input_std_eps: float

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        *,
        key: jax.Array,
        activation: Callable,
        use_skip: bool,
        use_fourier_features: bool,
        num_fourier_features: int,
        include_raw_input: bool,

        normalize_input: bool,
        input_mean: jax.Array | float | None,
        input_std: jax.Array | float | None,
        input_std_eps: float,

        use_xavier_init: bool,
        xavier_gain: float,
    ):
        init_key, xavier_key, fourier_key = jax.random.split(key, 3)

        self.use_fourier_features = use_fourier_features
        self.include_raw_input = include_raw_input

        self.normalize_input = normalize_input
        self.input_std_eps = input_std_eps

        if normalize_input:
            if input_mean is None:
                input_mean = jnp.zeros((in_size,))
            if input_std is None:
                input_std = jnp.ones((in_size,))
            self.input_mean = jnp.broadcast_to(jnp.asarray(input_mean), (in_size,))
            self.input_std = jnp.broadcast_to(jnp.asarray(input_std), (in_size,))
        else:
            self.input_mean = None
            self.input_std = None

        if use_fourier_features:
            self.fourier_frequencies = jax.random.normal(fourier_key, (num_fourier_features, in_size))

            encoded_in_size = (
                (in_size if include_raw_input else 0)
                + 2 * num_fourier_features
            )
        else:
            self.fourier_frequencies = None
            encoded_in_size = in_size

        sizes = [encoded_in_size, *hidden_sizes]
        keys = jax.random.split(init_key, len(hidden_sizes) + 1)

        self.hidden_blocks = [
            ResidualBlock(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                activation=activation,
                use_skip=use_skip,
            )
            for i in range(len(hidden_sizes))
        ]

        last_hidden = hidden_sizes[-1] if hidden_sizes else encoded_in_size
        self.output_layer = eqx.nn.Linear(
            last_hidden, out_size, use_bias=True, key=keys[-1]
        )
        self.squeeze_output = (out_size == 1)

        if use_xavier_init:
            reinit_model = init_linear_xavier_normal_zero_bias(
                self,
                key=xavier_key,
                gain=xavier_gain,
            )
            self.hidden_blocks = reinit_model.hidden_blocks
            self.output_layer = reinit_model.output_layer

    def _normalize_input(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        if not self.normalize_input:
            return x
        return (x - self.input_mean) / (self.input_std + self.input_std_eps)

    def _encode_input(self, x: jax.Array) -> jax.Array:
        if not self.use_fourier_features:
            return x
        
        feats = []
        if self.include_raw_input:
            feats.append(x)
        proj = 2 * jnp.pi * (self.fourier_frequencies @ x)
        feats.append(jnp.sin(proj))
        feats.append(jnp.cos(proj))

        return jnp.concatenate(feats, axis=0)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self._normalize_input(x)
        x = self._encode_input(x)

        for block in self.hidden_blocks:
            x = block(x)

        x = self.output_layer(x)

        if self.squeeze_output:
            x = jnp.squeeze(x, axis=-1)

        return x