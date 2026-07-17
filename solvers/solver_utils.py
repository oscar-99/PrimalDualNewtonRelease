import jax
import jax.numpy as jnp
from utils.differ import get_hvp


def make_hvp_linear_solver(loss_fn, linear_solver, **solver_kwargs):
    """
    Returns a linear solver which utilizes the Hessian-vector product of the given loss function as the underlying matrix, i.e., solves H v = b where H is the Hessian of loss_fn at the current parameters. This allows for jitting of the linear solver over the parameters of the loss function, since jitting over the hessian vector product mapping itself would require recompilation at each new parameter value.
    """
    hvpx = get_hvp(loss_fn)
    
    # Include args for possible data terms.
    def hvp_linear_solver(params, b, *args):
        hvp = hvpx(params, *args)
        return linear_solver(hvp, b, **solver_kwargs)
        
    return jax.jit(hvp_linear_solver)
