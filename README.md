# Primal--Dual Newton-MR

The experiments are implemented in Python using JAX. Optimisation problems are specified through callable objective and constraint functions, with the required derivatives generated using JAX automatic differentiation. The implementation uses just-in-time compilation and, where appropriate, operator-based Jacobian-, Jacobian-transpose-, and Hessian-vector products to support the iterative linear-algebra routines without explicitly forming the associated derivative matrices.

The repository includes implementations of the proposed primal--dual Newton method, its inner solvers, the comparison methods used in the paper, and the scripts required to run and analyse the NLS and PINN experiments.

## Reproducibility and runtime limits

Several experiments use a wall-clock time limit as a termination condition. Consequently, the number of iterations completed within a fixed time budget, and therefore the final numerical results, may vary across machines. Relevant factors include the available CPU or GPU hardware, the JAX backend, compilation and caching behaviour, and other system load. The supplied commands reproduce the experimental configurations used in the paper, but small machine-dependent differences should be expected.

## Dependencies

- JAX 0.8.0
- JAXlib 0.8.0
- NumPy 2.3.4
- SciPy 1.16.3
- Equinox 0.13.2
- Optax 0.2.6
- Matplotlib 3.10.7
- Scikit-learn 1.7.2
- JupyterLab 4.4.10

## Running the experiments

The scripts `run_nls_experiment.py` and `run_pinn_experiment.py` provide the main entry points for running the numerical experiments. The `runtime_profiles` defined near the top of each file provide preset solver configurations. An alternative profile can be selected using

```bash
--runtime_profile PROFILE
```

The objective, dataset, constraints, and other experiment settings can also be configured through the command-line interface. Run either script with `-h` to view the available options:

```bash
python run_nls_experiment.py -h
python run_pinn_experiment.py -h
```

### NLS experiments

Run the final MNIST experiment with

```bash
python run_nls_experiment.py mnist
```

Run the final Fashion-MNIST experiment with

```bash
python run_nls_experiment.py fashion \
    --runtime_profile large \
    --n_lin_constraints 200
```

### PINN experiments

Run the final single-mode Poisson experiment with

```bash
python run_pinn_experiment.py poisson_single_mode
```

Run the final Helmholtz experiment with

```bash
python run_pinn_experiment.py helmholtz \
    --pde_param 1.0 \
    --runtime_profile helmholtz
```

## Hyperparameter sweeps

The commands in this section reproduce the parameter sweeps used to select the solver configurations reported in the numerical experiments.

### Sweep arguments

- `--sweep_param PARAMETER` specifies the runtime parameter to vary.
- `--sweep_values VALUE [VALUE ...]` gives the values tested for that parameter.
- `--solver_suite SUITE` restricts the run to a compatible solver or solver family.
- `--runtime_profile PROFILE` selects the baseline runtime configuration from which the sweep is performed.

All parameters other than the selected sweep parameter retain the values specified by the chosen runtime profile and any additional command-line arguments.

### Swept parameters

- `tan_inner_tol` controls the termination tolerance for the tangent inner solve used by the primal--dual Newton method.
- `subproblem_rel_tol` controls the relative termination tolerance for the comparison solver's subproblem solve.
- `tan_reg` controls the regularisation applied to the tangent subproblem.

### MNIST sweeps

Sweep the tangent inner tolerance for the primal--dual Newton solver:

```bash
python run_nls_experiment.py mnist \
    --sweep_param tan_inner_tol \
    --sweep_values 1e-1 1e-2 1e-3 1e-4 \
    --solver_suite pdn_ls_dual
```

Sweep the relative subproblem tolerance for the augmented Lagrangian solver:

```bash
python run_nls_experiment.py mnist \
    --sweep_param subproblem_rel_tol \
    --sweep_values 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 \
    --solver_suite alm
```

### Fashion-MNIST sweeps

Sweep the tangent inner tolerance for the primal--dual Newton solver:

```bash
python run_nls_experiment.py fashion \
    --runtime_profile large \
    --n_lin_constraints 200 \
    --sweep_param tan_inner_tol \
    --sweep_values 1e-3 1e-2 1e-1 1e0 \
    --solver_suite pdn_ls_dual
```

Sweep the relative subproblem tolerance for the augmented Lagrangian solver:

```bash
python run_nls_experiment.py fashion \
    --runtime_profile large \
    --n_lin_constraints 200 \
    --sweep_param subproblem_rel_tol \
    --sweep_values 1e-5 1e-4 1e-3 1e-2 \
    --solver_suite alm
```

Sweep the tangent regularisation parameter for the Fashion-MNIST regularisation comparison:

```bash
python run_nls_experiment.py fashion \
    --sweep_param tan_reg \
    --sweep_values 1e-2 1e-1 1e0 1e1 0 \
    --solver_suite pdn_ls_dual \
    --n_lin_constraints 200
```

### Single-mode Poisson sweeps

Sweep the tangent inner tolerance for the primal--dual Newton solver:

```bash
python run_pinn_experiment.py poisson_single_mode \
    --sweep_param tan_inner_tol \
    --sweep_values 1e-2 1e-1 1e0 1e1 \
    --solver_suite pdn_ls_dual
```

Sweep the relative subproblem tolerance for the augmented Lagrangian solver:

```bash
python run_pinn_experiment.py poisson_single_mode \
    --sweep_param subproblem_rel_tol \
    --sweep_values 1e-7 1e-6 1e-5 1e-4 1e-3 \
    --solver_suite alm
```

### Helmholtz sweeps

Sweep the tangent inner tolerance for the primal--dual Newton solver:

```bash
python run_pinn_experiment.py helmholtz \
    --pde_param 1.0 \
    --runtime_profile helmholtz \
    --max_time 240 \
    --sweep_param tan_inner_tol \
    --sweep_values 1e3 1e4 1e5 \
    --solver_suite pdn_ls_dual
```

Sweep the relative subproblem tolerance for the augmented Lagrangian solver:

```bash
python run_pinn_experiment.py helmholtz \
    --pde_param 1.0 \
    --runtime_profile helmholtz \
    --max_time 240 \
    --sweep_param subproblem_rel_tol \
    --sweep_values 1e1 1e2 1e3 1e4 \
    --solver_suite alm
```

## Plotting

The plotting code used to generate the final figures is contained in `results_final.py`. Plots for an experiment can be generated by passing the corresponding experiment name from the `experiments` directory through the command-line interface.

More detailed comparisons can be generated using `pinn_results_analysis.py` for the PINN experiments and `results_analysis.py` for the NLS experiments. Run the relevant script with `-h` to view its available options.