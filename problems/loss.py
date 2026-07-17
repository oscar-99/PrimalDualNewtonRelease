import jax.numpy as jnp
import jax
import equinox as eqx

def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

def mae_loss(predictions, targets):
    return jnp.mean(jnp.abs(predictions - targets))

def accuracy(predictions, targets):
    # predictions are the raw logits produced for each class push through softmax for probabilities. Targets are assumed to be class numbers
    probs = jax.nn.softmax(predictions, axis=-1)
    predicted_classes = jnp.argmax(probs, axis=1)
    return jnp.mean(predicted_classes == targets)

def cross_entropy_loss(predictions, targets):
    # predictions are the raw logits produced for each class
    log_probs = jax.nn.log_softmax(predictions, axis=1)
    targets_one_hot = jax.nn.one_hot(targets, num_classes=predictions.shape[1])
    return -jnp.mean(jnp.sum(targets_one_hot * log_probs, axis=1))

def get_regression_loss(model):
    # save the static elements of the model
    _, static = eqx.partition(model, eqx.is_array)
    
    @eqx.filter_jit
    def regression_loss(params, X, y):
        # Recombine new params with static to obtain full model
        model_at_params = eqx.combine(params, static)
        predictions = jax.vmap(model_at_params)(X)
        loss = mse_loss(predictions, y)
        return loss, {}
    
    return regression_loss

def get_classification_loss(model):
    # save the static elements of the model
    _, static = eqx.partition(model, eqx.is_array)
    
    @eqx.filter_jit
    def classification_loss(params, X, y):
        # Recombine new params with static to obtain full model
        model_at_params = eqx.combine(params, static)
        predictions = jax.vmap(model_at_params)(X)
        loss = cross_entropy_loss(predictions, y)
        acc = accuracy(predictions, y)
        return loss, {"acc": acc}
    
    return classification_loss