#---AI Generated Start---

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsplib_parser import TSPInstance


# ---------------------------------------------------------------------------
# 1. Summary Table Generator
# ---------------------------------------------------------------------------
def generate_summary_table(
    csv_path: str = "BA/Part_A/results/results_summary.csv",
    output_path: str = "BA/Part_A/results/table_summary.csv",
) -> pd.DataFrame:
    """Reads raw results, computes statistical aggregations across the 10 seeds,

    and saves a clean summary table as required by the assignment.
    """
    df = pd.read_csv(csv_path)

    # City counts for each benchmark instance
    cities_map = {
        "eil51": 51,
        "berlin52": 52,
        "st70": 70,
        "eil76": 76,
        "kroA100": 100,
        "ch150": 150,
    }

    # Group by instance and operator, then compute statistics
    summary = (
        df.groupby(["instance", "operator"], sort=False)
        .agg(
            optimum=("optimal_length", "first"),
            best=("best_length", "min"),
            mean=("best_length", "mean"),
            std=("best_length", "std"),
            mean_gap=("gap_percent", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    # Add the 'cities' column and reorder to match the PDF
    summary["cities"] = summary["instance"].map(cities_map)
    columns_order = [
        "instance",
        "cities",
        "operator",
        "optimum",
        "best",
        "mean",
        "std",
        "mean_gap",
        "mean_runtime",
    ]
    summary = summary[columns_order]

    # Round floating-point values for neat reporting
    summary = summary.round(
        {"mean": 2, "std": 2, "mean_gap": 2, "mean_runtime": 2}
    )

    # Save to CSV
    summary.to_csv(output_path, index=False)
    print(f"[+] Summary table saved to: {output_path}")

    return summary


# ---------------------------------------------------------------------------
# 2. Convergence Curve Plotter
# ---------------------------------------------------------------------------
def plot_convergence(
    instance_name: str,
    detailed_data: list[dict],
    optimal_value: int,
    output_dir: str = "BA/Part_A/results",
):
    """Plots Evaluations (0 to 30,000) vs Mean Best-so-Far Tour Length across 10 runs

    for both Swap and Segment Reversal, marking the optimal reference line.
    """
    # Filter runs for this instance
    runs = [r for r in detailed_data if r["instance"] == instance_name]
    swap_runs = [r for r in runs if r["operator"] == "swap"]
    reversal_runs = [r for r in runs if r["operator"] == "reversal"]

    # Extract histories (shape: 10 runs x 30,000 evaluations)
    evaluations = np.arange(1, 30001)
    swap_matrix = np.array([[pt[1] for pt in r["history"]] for r in swap_runs])
    reversal_matrix = np.array(
        [[pt[1] for pt in r["history"]] for r in reversal_runs]
    )

    # Compute mean curve across the 10 seeds
    mean_swap = swap_matrix.mean(axis=0)
    mean_reversal = reversal_matrix.mean(axis=0)

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(
        evaluations, mean_swap, label="Swap Operator", color="tab:blue", lw=1.8
    )
    plt.plot(
        evaluations,
        mean_reversal,
        label="Segment Reversal",
        color="tab:orange",
        lw=1.8,
    )
    plt.axhline(
        y=optimal_value,
        color="tab:red",
        linestyle="--",
        lw=1.5,
        label=f"Optimum ({optimal_value})",
    )

    plt.title(
        f"Convergence Comparison: {instance_name} (Mean of 10 Runs)",
        fontsize=13,
        pad=10,
    )
    plt.xlabel("Evaluations", fontsize=11)
    plt.ylabel("Mean Best-so-Far Tour Length", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    out_file = f"{output_dir}/convergence_{instance_name}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] Convergence plot saved to: {out_file}")


# ---------------------------------------------------------------------------
# 3. Route Plotter (berlin52 Champion Tour)
# ---------------------------------------------------------------------------
def plot_best_tour(
    instance_name: str,
    tsp_file_path: str,
    detailed_data: list[dict],
    output_dir: str = "BA/Part_A/results",
):
    """Finds the single shortest tour found for berlin52 across all runs/operators,

    and draws the 2D path with starting depot, return leg, and labels.
    """
    # 1. Find the best run across all 20 runs
    b52_runs = [r for r in detailed_data if r["instance"] == instance_name]
    best_run = min(b52_runs, key=lambda r: r["best_length"])

    best_trip = best_run["best_trip"]
    operator_used = best_run["operator"]
    seed_used = best_run["seed"]
    best_length = best_run["best_length"]
    gap_percent = best_run["gap_percent"]

    # 2. Load city coordinates
    instance = TSPInstance()
    instance.read_file(tsp_file_path)
    coords = {city.id: (city.x, city.y) for city in instance.dataset}

    # 3. Build closed loop (append starting city at the end for the return leg)
    closed_trip = best_trip + [best_trip[0]]
    xs = [coords[cid][0] for cid in closed_trip]
    ys = [coords[cid][1] for cid in closed_trip]

    # 4. Plotting
    plt.figure(figsize=(9, 7))

    # Draw tour route lines (including return leg)
    plt.plot(
        xs,
        ys,
        color="tab:blue",
        linestyle="-",
        linewidth=1.5,
        alpha=0.85,
        label="Tour Route",
    )

    # Draw city scatter points
    all_xs = [c.x for c in instance.dataset]
    all_ys = [c.y for c in instance.dataset]
    plt.scatter(
        all_xs,
        all_ys,
        color="black",
        s=40,
        zorder=3,
        label="Cities (Identifiers)",
    )

    # Highlight City 1 / Depot (index 0 in 0-based code)
    depot_x, depot_y = coords[best_trip[0]]
    plt.scatter(
        [depot_x],
        [depot_y],
        color="tab:red",
        s=140,
        marker="*",
        zorder=4,
        label="Starting City (Depot, City 1)",
    )

    # Title & Metadata as mandated by the assignment
    title_text = (
        f"Shortest Valid Tour: {instance_name}\n"
        f"Neighborhood: {operator_used.capitalize()} | Seed: {seed_used} | "
        f"Length: {best_length} | Gap: {gap_percent:.2f}%"
    )
    plt.title(title_text, fontsize=12, pad=12)
    plt.xlabel("X Coordinate", fontsize=11)
    plt.ylabel("Y Coordinate", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out_file = f"{output_dir}/best_route_{instance_name}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] Route plot saved to: {out_file}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_file = "BA/Part_A/results/results_summary.csv"
    json_file = "BA/Part_A/results/results_detailed.json"
    results_dir = "BA/Part_A/results"

    print("Generating Part A visualizations and summary tables...\n")

    # 1. Summary Table
    summary_df = generate_summary_table(csv_file)
    print("\n--- Summary Table Preview ---")
    print(summary_df.to_string(index=False))

    # 2. Detailed Data Loading
    with open(json_file, "r", encoding="utf-8") as f:
        detailed_runs = json.load(f)

    # 3. Convergence Plots (berlin52 and ch150)
    plot_convergence(
        "berlin52", detailed_runs, optimal_value=7542, output_dir=results_dir
    )
    plot_convergence(
        "ch150", detailed_runs, optimal_value=6528, output_dir=results_dir
    )

    # 4. Route Plot for shortest berlin52 tour
    plot_best_tour(
        "berlin52",
        "BA/Part_A/data/berlin52.tsp",
        detailed_runs,
        output_dir=results_dir,
    )

    print("\nAll deliverables for Part A have been generated successfully!")

    #---AI Generated End---