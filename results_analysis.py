
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from runners.results_io import load_experiment_results
from utils.plotting import plot_comparison_panel, plot_run_diagnostics, plot_final_time_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PINN experiment results.")
    parser.add_argument(
        "experiment_name",
        type=str,
        default="poisson_single_mode",
        help="Name of the experiment to analyze (should match the directory name in 'experiments/')."
    )
    parser.add_argument("--final", action="store_true", help="Whether to only plot the final results (no diagnostics).")
    return parser.parse_args()


def __main__():
    args = parse_args()
    experiment_name = args.experiment_name
    print(f"Loading results for experiment '{experiment_name}'...")
    result_dir = os.path.join("experiments", experiment_name)
    results = load_experiment_results(result_dir)

    plot_comparison_panel(
        results,
        time=True,
        metrics=(
            "objective",
            "constraint_norm",
            "stationarity",
            "test_accuracy"
        ),
    )

    for result in results:
        print("")
        print(result.solver_name)
        print("convergence status:", result.summary()["status"])
        print("iterations:", result.summary()["iterations"], "total time:", result.summary()["total_time"])
        plot_run_diagnostics(result)
    
    plt.show()
        
if __name__ == "__main__":
    __main__()