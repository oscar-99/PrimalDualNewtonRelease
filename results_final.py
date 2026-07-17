import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from runners.results_io import load_experiment_results
from utils.plotting import plot_comparison_panel, plot_run_diagnostics, plot_final_time_comparison
from utils.pinn_plotting import plot_pinn_error_heatmaps

plot_config ={
    "helmholtz" : {
        "plot_config" : {"subsample_target_points": 1000, "logx" : False},
        "pinn" : True,
        "heatmap_exclude" : ["sqp_default"],
    },
    "poisson_single_mode" : {
        "plot_config" : {"subsample_target_points": 1000, "logx" : False},
        "pinn" : True,
        "heatmap_exclude" : ["sqp_default"],
    },
    "nls_mnist" : {
        "plot_config" : {"logx" : True},
        "pinn" : False,
        "logx" : False
    },
    "nls_fashion_200" : {
        "plot_config" : {"logx" : True},
        "pinn" : False,
    },
    "nls_fashion_200__sweep_tan_reg" : {
        "plot_config" : {"logx" : True},
        "pinn" : False,
    },
}

EXPERIMENT_NAMES = list(plot_config.keys()) # allows for quickly 


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PINN experiment results.")
    parser.add_argument(
        "experiment_name",
        type=str,
        default="poisson_single_mode",
        help="Name of the experiment to analyze (should match the directory name in 'experiments/'). Use 'all' to generate plots for all experiments. Available experiments: " + ", ".join(EXPERIMENT_NAMES)
    )
    return parser.parse_args()



def _filter_results(results, *, exclude_solver_names=None):
    if not exclude_solver_names:
        return results
    exclude = set(exclude_solver_names)
    print(exclude)
    return [result for result in results if result.solver_name not in exclude]

def generate_plots(experiment_name: str):
    
    print(f"Loading results for experiment '{experiment_name}'...")
    result_dir = os.path.join("experiments", experiment_name)
    results = load_experiment_results(result_dir)
    fig, axes = plot_final_time_comparison(results, stationarity_tol=1e-5, constraint_tol=1e-5, **plot_config.get(experiment_name, {}).get("plot_config", {}))
    plot_dir = os.path.join("plots", experiment_name+".pdf")
    fig.savefig(plot_dir, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved plot to {plot_dir}")

    if plot_config.get(experiment_name, {}).get("pinn", True):
        filtered_results = _filter_results(results, exclude_solver_names=plot_config.get(experiment_name, {}).get("heatmap_exclude", []))
        fig, axes = plot_pinn_error_heatmaps(filtered_results)
        plot_dir = os.path.join("plots", experiment_name+"_error_heatmap.pdf")
        fig.savefig(plot_dir, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved plot to {plot_dir}")

        for result in results:
            print("")
            print(result.solver_name)
            print("Final L2 relative error: {:.3e}".format(result.history["test_l2_rel_error"][-1]))
            print("Final Linfty error: {:.3e}".format(result.history["test_linf_error"][-1]))

    

if __name__ == "__main__":
    args = parse_args()
    experiment_name = args.experiment_name

    if experiment_name == "all":
        print("Generating plots for all experiments...")
        for exp_name in EXPERIMENT_NAMES:
            generate_plots(exp_name)
    else:
        generate_plots(experiment_name)
        plt.show()