import jax
import jax.numpy as jnp
import equinox as eqx

from typing import NamedTuple
from problems.build_problem import Problem
from problems.pinns.data_gen import get_box_with_boundary
from utils.utils import make_2d_grid
from utils.differ import get_laplacian

"""
This file handles building a PINN problem. It also contains some off the shelf
PINN problem constructors.
"""

jax.config.update("jax_enable_x64", True)

# Domain encode domain invariants. 
class Domain(NamedTuple):
    x_range: tuple[float, float] 
    y_range: tuple[float, float] | None
    mean: jax.Array 
    std: jax.Array

PiBoxDomain = Domain(
    x_range=(0.0, jnp.pi),
    y_range=(0.0, jnp.pi),
    mean=jnp.array([(0.0 + jnp.pi) / 2, (0.0 + jnp.pi) / 2]),
    std=jnp.array([(jnp.pi - 0.0) / jnp.sqrt(12), (jnp.pi - 0.0) / jnp.sqrt(12)]),
)

Box2Domain = Domain(
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    mean=jnp.array([0.0, 0.0]),
    std=jnp.array([2.0 / jnp.sqrt(12), 2.0 / jnp.sqrt(12)]),
)

UnitInterval = Domain(
    x_range=(0., 1.),
    y_range=None,
    mean = jnp.array([0.5]),
    std=jnp.array([1./jnp.sqrt(12)])
)


PINN_UTRUE_REGISTRY = {
    "poisson_single_mode" : 
    {
        "u_true" : lambda x, _: jnp.cos(x[0]) * jnp.sin(x[1]),
        "source" : lambda x, _: -2.0 * jnp.cos(x[0]) * jnp.sin(x[1]),
        "domain" : PiBoxDomain
    },
    "poisson_high_freq" : 
    {
        "u_true" : lambda x, _: jnp.cos(4*x[0]) * jnp.sin(x[1]),
        "source" : lambda x, _: -17.0 * jnp.cos(4*x[0]) * jnp.sin(x[1]),
        "domain" : PiBoxDomain
    },
    "helmholtz" :
    {
        "u_true" : lambda x, k: jnp.sin(jnp.pi * x[0]) * jnp.sin(4.0 * jnp.pi * x[1]),
        "source" : lambda x, k: (k**2 - 17.*jnp.pi**2)*jnp.sin(jnp.pi * x[0]) * jnp.sin(4.0 * jnp.pi * x[1]),
        "domain" : Box2Domain
    },
    "convection_diffusion" : 
    {
        "u_true" : lambda x, alpha: jnp.exp(-x[0]/alpha)/(1-jnp.exp(-1./alpha)) - 0.5,
        "source" : lambda x, alpha: 0.0,
        "domain" : UnitInterval
    }
}

def get_dirichlet_poisson(f, g):
    # returns the vmapped residual for the pde and boundary condition given the source term f and boundary function g.
    def pde(u, x):
        lapu = get_laplacian(lambda x: u(x))
        return lapu(x) - f(x)

    def boundary(u, x):
        return u(x) - g(x)
    
    # Returned batched functions 
    return jax.vmap(pde, in_axes=(None, 0)), jax.vmap(boundary, in_axes=(None, 0))

def get_dirichlet_helmholtz(f, g, k):
    # returns the vmapped residual for the pde and boundary condition given the source term f and boundary function g.
    def pde(u, x):
        lapu = get_laplacian(lambda x: u(x))
        return lapu(x) + k**2 * u(x) - f(x)
    
    def boundary(u, x):
        return u(x) - g(x)
    
    return jax.vmap(pde, in_axes=(None, 0)), jax.vmap(boundary, in_axes=(None, 0))

def get_convection_diffusion(f, g, alpha):
    # Returns vmapped residual for convection diffusion equation. f and g should already be closed over the pde parameters. 
    def pde(u, x):
        du = lambda x: jax.grad(u)(x)[0]
        dux, ddux =  jax.jvp(du, (x,), (jnp.ones_like(x),))
        return dux + alpha*ddux - f(x)

    def boundary(u, x):
        return u(x) - g(x)
    
    # Returned batched functions 
    return jax.vmap(pde, in_axes=(None, 0)), jax.vmap(boundary, in_axes=(None, 0)) 

def build_pinn_loss_constraints(model, 
                                pde, 
                                boundary,  
                                obj_penalty_weight=1.0,
                                cons_penalty_weight=1.0,
                                cons_penalty_fn="identity"):
    """
    Outputs the PDE loss:
        L(params) = obj_penalty_weight * ||pde(model(params), x_int)||^2
    and the boundary constraint violation:
        C(params) = cons_penalty_weight * cons_penalty_fn(model(params, x_bndry) - g(x_bndry))

    The cons_penalty_fn is a elementwise penalty function.
        "identity": cons_penalty_fn(x) = x
        "square": cons_penalty_fn(x) = x^2
        "abs": cons_penalty_fn(x) = |x|
    """
    _, static = eqx.partition(model, eqx.is_array) # save static part of the model.
    # Returns jitted loss function and boundary constraints function for PINN training in the constrained framework.
    if cons_penalty_fn == "identity":
        cons_penalty = lambda x: x
    elif cons_penalty_fn == "square":
        cons_penalty = lambda x: x**2
    elif cons_penalty_fn == "abs":
        cons_penalty = lambda x: jnp.abs(x)
    else:
        raise ValueError(f"Unknown cons_penalty_fn '{cons_penalty_fn}'. Supported options: identity, square, abs.")


    @jax.jit
    def loss(params, data=(None,))-> tuple[jax.Array, dict]:
        model = eqx.combine(params, static)
        x_int = data[0]
        res = pde(model, x_int)
        return (obj_penalty_weight) *jnp.sum(res**2), {}
    
    @jax.jit
    def boundary_constraints(params, data=(None,)):
        model = eqx.combine(params, static)
        x_bndry = data[0]
        res = boundary(model, x_bndry)
        return (cons_penalty_weight) * cons_penalty(res)
    
    return loss, boundary_constraints

def get_pinn(name, pde_param=None):
    if name not in PINN_UTRUE_REGISTRY:
        raise ValueError(
            f"Unknown pinn name '{name}'. "
            f"Available names: {', '.join(sorted(PINN_UTRUE_REGISTRY))}"
        )

    pinn = PINN_UTRUE_REGISTRY[name]
    u_true = pinn["u_true"]
    source = pinn["source"]
    # Get the PDE residual and boundary constraints based on the problem family.
    def f(x):
        return source(x, pde_param)
    def g(x):
        return u_true(x, pde_param)
    
    if "poisson" in name:
        pde, boundary = get_dirichlet_poisson(f, g)
    elif "helmholtz" in name:
        if pde_param is None:
            raise ValueError("Helmholtz equation requires 'k' parameter.")
        pde, boundary = get_dirichlet_helmholtz(f, g, pde_param)
    elif "convection_diffusion" in name:
        if pde_param is None:
            raise ValueError("Convection-diffusion equation requires 'alpha' parameter.")
        pde, boundary = get_convection_diffusion(f, g, pde_param)
    else:
        raise ValueError(f"unknown problem family {name}.")

    return pde, boundary

def build_pinn_problem(
    model,
    *,
    name: str,
    pde_param : None | float,
    n_int: int,
    n_bndry: int,
    eval_grid_shape: tuple[int, int],
    seed: int,
    obj_penalty_weight: float,
    cons_penalty_weight: float,
    cons_penalty_fn : str,
) -> Problem:
    """
    Build a pinn problem from the registry of true solutions indexed by name. Outputs a problem object with the relevant objective and constraints as well as a test function that compares to the true solution on a grid.
    """
    pinn = PINN_UTRUE_REGISTRY[name]
    pde, boundary = get_pinn(name, pde_param=pde_param)
    u_true = lambda x: pinn["u_true"](x, pde_param)
    domain = pinn["domain"]
    x_bounds = domain.x_range
    y_bounds = domain.y_range

    # construct the loss and constraint functions from the residuals and boundary conditions.
    pde_loss, boundary_constraints = build_pinn_loss_constraints(
        model,
        pde,
        boundary,
        obj_penalty_weight=obj_penalty_weight,
        cons_penalty_weight=cons_penalty_weight,
        cons_penalty_fn=cons_penalty_fn,
    )

    x_int, x_bndry = get_box_with_boundary(x_bounds, y_bounds, n_int, n_bndry, seed=seed)

    if y_bounds is not None:
        x_eval = make_2d_grid(
            x_bounds,
            y_bounds,
            n_x=eval_grid_shape[0],
            n_y=eval_grid_shape[1],
        )
    else:
        x_eval = jnp.linspace(x_bounds[0], x_bounds[1], num=eval_grid_shape[0])[:, None]
    
    # x0 from the initialised model.
    primals, static = eqx.partition(model, eqx.is_array)
    u_true_batched = jax.vmap(u_true)

    def test(params, data=None):
        model_eval = eqx.combine(params, static)
        pred = jax.vmap(model_eval)(x_eval)
        truth = u_true_batched(x_eval)

        pred = jnp.asarray(pred).reshape(-1)
        truth = jnp.asarray(truth).reshape(-1)
        err = pred - truth

        truth_norm = jnp.maximum(jnp.linalg.norm(truth), 1e-16)
        return {
            "l2_rel_error": jnp.linalg.norm(err) / truth_norm,
            "linf_error": jnp.max(jnp.abs(err)),
        }

    return Problem(
        name=name,
        x0=primals,
        objective=pde_loss,
        constraint=boundary_constraints,
        objective_data=(x_int,),
        constraint_data=(x_bndry,),
        test=test,
        obj_has_aux=True,
    )