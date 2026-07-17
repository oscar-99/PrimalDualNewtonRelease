# %%
import os
from runners.results_io import load_experiment_results, rebuild_predictor
from utils.plotting import plot_comparison_panel, plot_run_diagnostics
from utils.pinn_plotting import plot_pinn_result_surface_contour_mixed, plot_pinn_result
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import argparse
# Example to load

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PINN experiment results.")
    parser.add_argument(
        "experiment_name",
        type=str,
        default="poisson_single_mode",
        help="Name of the experiment to analyze (should match the directory name in 'experiments/')."
    )
    parser.add_argument(
        "--alg",
        type=str,
        default=[],
        nargs="+",
        help="Algorithms to display. Matches the substring."
    )
    return parser.parse_args()

def matches_selected_algorithms(solver_name: str, selected_algs: list[str]) -> bool:
    if not selected_algs:
        return True
    return any(alg in solver_name for alg in selected_algs)

def __main__():
    args = parse_args()
    experiment_name = args.experiment_name
    print(f"Loading results for experiment '{experiment_name}'...")
    result_dir = os.path.join("experiments", experiment_name)
    results = load_experiment_results(result_dir)

    results = [
        result for result in results
        if matches_selected_algorithms(result.solver_name, args.alg)
    ]


    plot_comparison_panel(
        results,
        time=True,
        metrics=(
            "objective",
            "constraint_norm",
            "stationarity",
            "test_l2_rel_error",
            "test_linf_error",
        ),
    )

    for result in results:
        print("")
        print(result.solver_name)
        print("convergence status:", result.summary()["status"])
        print("iterations:", result.summary()["iterations"], "total time:", result.summary()["total_time"])
        plot_run_diagnostics(result)
        print("Final test error: l2rel={:5e} linf={:5e}".format(result.history["test_l2_rel_error"][-1], result.history["test_linf_error"][-1]))

    for result in results:
        print(result.solver_name)
        plot_pinn_result(result, )

    plt.show()

if __name__ == "__main__":
    __main__()