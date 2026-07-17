from typing import Callable, Optional

import jax
import jax.numpy as jnp


def _value_only(fun: Callable, has_aux: bool):
    """Return a function that always produces only the scalar value."""
    if has_aux:
        return lambda params, data=None: fun(params, data=data)[0]
    else:
        return fun
    
def get_jacobian_operators(constraint):
    """
    Build reusable Jv and J^T w operators at a fixed point.

    Parameters
    ----------
    constraint : callable
        Function with signature constraint(params, data=None).

    Returns
    -------
    factory : callable
        factory(params, data=None) -> (Jv, JTv)
        where
            Jv(v)  = J(params) @ v
            JTv(w) = J(params).T @ w
    """
    def factory(params, data=None):
        # Reusable forward linear map at the current point.
        _, Jv_raw = jax.linearize(lambda p: constraint(p, data=data), params)

        # linear_transpose needs example primals with the same pytree structure.
        zeros = jax.tree.map(jnp.zeros_like, params)

        def JTv_raw(cotangent):
            return jax.linear_transpose(Jv_raw, zeros)(cotangent)[0]

        # Compile the operator applications once per shape/dtype specialization.
        Jv = jax.jit(Jv_raw)
        JTv = jax.jit(JTv_raw)

        return Jv, JTv
    
    return factory


def fval_grad_hvp(obj: Callable, has_aux: bool = False):
    """
    Returns an oracle producing value, gradient, and an HVP linear operator.

    Interface:
        obj(params, data=None) -> f
    or
        obj(params, data=None) -> (f, aux)

    Output:
        if has_aux is False:
            f_val, grad, hvp_fun

        if has_aux is True:
            (f_val, aux), grad, hvp_fun

    where hvp_fun(v) returns Hessian(params) @ v.
    """
    if has_aux:
        def grad_with_aux(params, data=None):
            (f_val, aux), grad = jax.value_and_grad(obj, has_aux=True)(params, data=data)
            return grad, (f_val, aux)

        def fgh_oracle(params, data=None):
            grad, hvp_fun, (f_val, aux) = jax.linearize(
                lambda p: grad_with_aux(p, data=data),
                params,
                has_aux=True,
            )
            return (f_val, aux), grad, hvp_fun

    else:
        def grad_with_aux(params, data=None):
            f_val, grad = jax.value_and_grad(obj)(params, data=data)
            return grad, f_val  # treat f_val as aux for linearize

        def fgh_oracle(params, data=None):
            grad, hvp_fun, f_val = jax.linearize(
                lambda p: grad_with_aux(p, data=data),
                params,
                has_aux=True,
            )
            return f_val, grad, hvp_fun

    return fgh_oracle


def get_hvp(obj: Callable, has_aux: bool = False):
    """
    Returns a function hvp(params, data=None) -> hvp_fun,
    where hvp_fun(v) = Hessian(params) @ v.

    This discards value/aux information, so it is not maximally efficient.
    """
    value_fun = _value_only(obj, has_aux=has_aux)

    def hvp(params, *args, **kwargs):
        grad_fun = jax.grad(value_fun)
        _, hvp_fun = jax.linearize(lambda p: grad_fun(p, *args, **kwargs), params)
        return hvp_fun

    return hvp

class Oracle:
    obj_has_aux: bool
    
    def obj(self, params, objective_data=None, vals="fgH"):
        raise NotImplementedError
    def cons(self, params, constraint_data=None, vals="c"):
        raise NotImplementedError
    def lag(self, params, duals, objective_data=None, constraint_data=None, vals="Lhvp"):
        raise NotImplementedError

class AutodiffOracle(Oracle):
    """
    Automatic differentiation oracle for nonlinear optimization problems.

    This class provides a unified interface for evaluating objective
    functions, constraints, and their derivatives using JAX. All core
    derivative computations are compiled with ``jax.jit`` and reuse
    efficient operator constructions such as ``jax.linearize`` and
    ``jax.linear_transpose`` for repeated Jacobian– and Hessian–vector
    products.

    Parameters
    ----------
    objective : callable
        Objective function with signature

            objective(params, data=None) -> f

        or

            objective(params, data=None) -> (f, aux)

        where

        - ``params`` is a pytree of JAX arrays representing the optimization
          variables,
        - ``data`` is optional auxiliary data,
        - ``f`` is a scalar objective value,
        - ``aux`` is optional auxiliary information returned alongside the
          objective value.

    constraint : callable, optional
        Vector-valued constraint function

            constraint(params, data=None) -> c(params)

        where ``c(params) ∈ R^m``. If provided, Jacobian and
        Jacobian-vector product operators are constructed.

    obj_has_aux : bool, default=False
        Indicates whether ``objective`` returns auxiliary data.

    Notes
    -----
    The oracle provides three main groups of functionality:

    Objective evaluation
        - value
        - gradient
        - Hessian-vector product operator

    Constraint evaluation
        - constraint value
        - full Jacobian
        - reusable operators for Jacobian-vector and transpose products

    Lagrangian evaluation
        - L(x, λ)
        - ∇L(x, λ)
        - Hessian-vector product of the Lagrangian

    All derivative operators are constructed using JAX automatic
    differentiation and compiled via ``jax.jit``.

    Methods
    -------
    obj(params, objective_data=None, vals="fgH")
        Evaluate the objective and derivatives.

        vals options:
        - ``"f"``   → objective value
        - ``"fg"``  → value and gradient
        - ``"fgH"`` → value, gradient, and Hessian-vector operator

        Returns
        -------
        If ``obj_has_aux=False``

            (derivative_information,)

        If ``obj_has_aux=True``

            (derivative_information, aux)

        where derivative_information is

            f
            (f, g)
            (f, g, Hv)

        with

            Hv(v) = ∇²f(x) v.

    cons(params, constraint_data=None, vals="c")
        Evaluate constraint functions and Jacobian information.

        vals options:
        - ``"c"``   → constraint value
        - ``"J"``   → full Jacobian matrix
        - ``"Jv"``  → reusable Jacobian-vector operator
        - ``"JTv"`` → reusable transpose Jacobian operator
        - ``"Jops"`` → return both operators ``(Jv, JTv)``

        The returned operators correspond to

            Jv(v)  = J(x) v
            JTv(w) = J(x)^T w

        and can be applied repeatedly without recomputing the Jacobian
        linearization.

    lag(params, duals, objective_data=None, constraint_data=None, vals="Lf")
        Evaluate the Lagrangian

            L(x, λ) = f(x) + λᵀ c(x)

        vals options:
        - ``"Lf"``  → Lagrangian value
        - ``"Lfg"`` → value and gradient
        - ``"Lhvp"`` → Hessian-vector operator of the Lagrangian

        The Hessian operator satisfies

            Hv(v) = ∇²L(x, λ) v.

    Design Notes
    ------------
    - Hessian-vector products are implemented using ``jax.linearize`` to
      allow efficient repeated application at a fixed point.
    - Jacobian transpose products are obtained using
      ``jax.linear_transpose`` applied to the linearized Jacobian operator.
    - All expensive operations are compiled via ``jax.jit`` to avoid
      repeated tracing.
    - The interface is intentionally operator-based to support Krylov
      methods (CG, MINRES, GMRES) used inside Newton and SQP algorithms.

    Examples
    --------
    Constructing an oracle:

    >>> oracle = AutodiffOracle(objective, constraint)

    Objective value and gradient:

    >>> (f, g), = oracle.obj(x, objective_data=data, vals="fg")

    Hessian-vector product:

    >>> (f, g, Hv), = oracle.obj(x, objective_data=data, vals="fgH")
    >>> Hv(v)

    Jacobian-vector product:

    >>> Jv = oracle.cons(x, constraint_data=data, vals="Jv")
    >>> Jv(v)

    Lagrangian gradient:

    >>> Lf, Lg = oracle.lag(x, lam, objective_data=data, constraint_data=data, vals="Lfg")
    """
    def __init__(self, objective, constraint=None, obj_has_aux=False):
        self.objective = objective
        self.constraint = constraint
        self.obj_has_aux = obj_has_aux

        self._objective_value = _value_only(objective, has_aux=obj_has_aux)

        # JIT compiled objective oracles
        self.f = jax.jit(objective)
        self.fg = jax.jit(jax.value_and_grad(objective, has_aux=obj_has_aux))
        self.fgh = jax.jit(fval_grad_hvp(objective, has_aux=obj_has_aux))

        if constraint is not None:
            self.c = jax.jit(constraint)
            self.jac = jax.jit(jax.jacrev(constraint))
            self._jacobian_ops = get_jacobian_operators(constraint)

            @jax.jit
            def lagrangian(params, duals, objective_data=None, constraint_data=None):
                f_val = self._objective_value(params, data=objective_data)
                c_val = constraint(params, data=constraint_data)
                return f_val + jnp.vdot(c_val, duals)

            self.lagrangian = lagrangian
            self.Lvalue_grad = jax.jit(jax.value_and_grad(self.lagrangian))
            self.Lhvp = jax.jit(get_hvp(self.lagrangian, has_aux=False))

    def _pack_obj_output(self, derivative_info):
        if self.obj_has_aux:
            derivs, aux = derivative_info
            return derivs, aux
        else:
            return (derivative_info,)

    def obj(self, params, objective_data=None, vals="fgH"):
        """
        Returns:
            if obj_has_aux is False:
                (derivative_information,)

            if obj_has_aux is True:
                (derivative_information, aux)

        where derivative_information is:
            vals == "f"   -> (f,)
            vals == "fg"  -> (f, g)
            vals == "fgH" -> (f, g, Hvp_fun)
        """
        if vals == "fgH":
            if self.obj_has_aux:
                (f_val, aux), grad, hvp_fun = self.fgh(params, data=objective_data)
                return (f_val, grad, hvp_fun), aux
            else:
                f_val, grad, hvp_fun = self.fgh(params, data=objective_data)
                return ((f_val, grad, hvp_fun),)

        elif vals == "fg":
            if self.obj_has_aux:
                (f_val, aux), grad = self.fg(params, data=objective_data)
                return (f_val, grad), aux
            else:
                f_val, grad = self.fg(params, data=objective_data)
                return ((f_val, grad),)

        elif vals == "f":
            if self.obj_has_aux:
                f_val, aux = self.f(params, data=objective_data)
                return (f_val,), aux
            else:
                f_val = self.f(params, data=objective_data)
                return ((f_val,),)

        else:
            raise ValueError(f"Invalid value for 'vals': {vals}")

    def cons(self, params, constraint_data=None, vals="c"):
        if self.constraint is None:
            raise ValueError("No constraint provided for oracle.")
        if vals == "c":
            return self.c(params, data=constraint_data)
        elif vals == "J":
            return self.jac(params, data=constraint_data)
        elif vals == "Jops":
            return self._jacobian_ops(params, data=constraint_data)
        elif vals == "Jv":
            Jv, _ = self._jacobian_ops(params, data=constraint_data)
            return Jv
        elif vals == "JTv":
            _, JTv = self._jacobian_ops(params, data=constraint_data)
            return JTv
        else:
            raise ValueError(f"Invalid value for 'vals': {vals}")

    def lag(self, params, duals, objective_data=None, constraint_data=None, vals="Lhvp"):
        if self.constraint is None:
            raise ValueError("No constraint provided for oracle.")

        if vals == "Lhvp":
            return self.Lhvp(
                params,
                duals,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
        elif vals == "Lfg":
            return self.Lvalue_grad(
                params,
                duals,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
        elif vals == "Lf":
            return self.lagrangian(
                params,
                duals,
                objective_data=objective_data,
                constraint_data=constraint_data,
            )
        else:
            raise ValueError(f"Invalid value for 'vals': {vals}")

def get_vector_value_JTv(f):
    """
    For a vector-valued function f, returns a function that computes the Jacobian-transpose-vector product with the dual variables (v), i.e., J^T v along with the function value.
    """
    # This function encodes the mapping from x->J^T v, constraint value is a byproduct.

    def JTv(params, duals, *args):
        f_params = lambda params: f(params, *args) # ensure function of params only for vjp.
        cval, vjp_map = jax.vjp(f_params, params) # vjp_map maps duals to J^T v
        return vjp_map(duals)[0], cval # duals should be just be an array, so we unpack the first element of the tuple.
    return JTv

def get_constraint_Hvp_map(JTv_func):
    """
    For a vector-valued function f (i.e., a constraint) input the Jacobian-transpose-vector product function (see above), and return a function that computes the constraint Hessian-vector product map along with the function value and Jacobian-transpose-vector product.
    """
    # We then differentiate this mapping to get the constraint Hessian vector product map
    def constraint_Hvp_map(params, duals, *args):
        JTv_params = lambda p: JTv_func(p, duals, *args) # function of params only, don't want to differentiate wrt duals.
        JTv, Hvp, cval = jax.linearize(JTv_params, params, has_aux=True) # linearize at the parameter value, auxiliary output is constraint value.
        return cval, JTv, Hvp
    
    return constraint_Hvp_map

def get_laplacian(u):
    hvpx = lambda x: get_hvp(u, has_aux=False)(x)
    
    @jax.jit
    def laplacian(x):
        """Compute the Laplacian of u."""
        hvp = hvpx(x)
        d = x.shape[0]
        eye = jnp.eye(d)
        hess_diag = lambda c, i: (None, hvp(eye[i])[i]) # format for scan
        tr = sum(jax.lax.scan(hess_diag, None, jnp.arange(d))[1])
        return tr
    return laplacian



