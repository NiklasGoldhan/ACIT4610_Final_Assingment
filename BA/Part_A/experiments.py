import csv
import json
import random
import time

from tsplib_parser import TSPInstance
from operators import reverse_operator, swap_operator
from distance import TripEval, create_distance_matrix
from bees_algorithm import BeesAlgorithm
from visualization import (generate_summary_table,plot_convergence,plot_best_tour)


class ExperimentConfig:
    def __init__(
        self,
        seeds: list[int] = None,
        population_size: int = 20,
        selected_sites: int = 6,
        elite_sites: int = 2,
        recruited_elite: int = 20,
        recruited_selected: int = 8,
        max_evals: int = 30_000,
    ):
        self.seeds = seeds if seeds is not None else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        self.population_size = population_size
        self.selected_sites = selected_sites
        self.elite_sites = elite_sites
        self.recruited_elite = recruited_elite
        self.recruited_selected = recruited_selected
        self.max_evals = max_evals

        # Official TSPLIB optimal tour lengths from the assignment sheet
        self.optimal_lengths = {
            "eil51": 426,
            "berlin52": 7542,
            "st70": 675,
            "eil76": 538,
            "kroA100": 21282,
            "ch150": 6528,
        }

        # The 6 benchmark instances and their relative paths
        self.instances = [
            ("eil51", "BA/Part_A/data/eil51.tsp"),
            ("berlin52", "BA/Part_A/data/berlin52.tsp"),
            ("st70", "BA/Part_A/data/st70.tsp"),
            ("eil76", "BA/Part_A/data/eil76.tsp"),
            ("kroA100", "BA/Part_A/data/kroA100.tsp"),
            ("ch150", "BA/Part_A/data/ch150.tsp"),
        ]


def run_single_trial(
    instance_name: str,
    dataset: list,
    distance_matrix: list[list[int]],
    operator_name: str,
    operator_func,
    seed: int,
    config: ExperimentConfig,
) -> dict:
    random.seed(seed)

    start_time = time.perf_counter()

    evaluator = TripEval(distance_matrix)
    evaluator.max_evals = config.max_evals

    ba = BeesAlgorithm(
        num_cities=len(dataset),
        evaluator=evaluator,
        operator=operator_func,
        population_size=config.population_size,
        selected_sites=config.selected_sites,
        elite_sites=config.elite_sites,
        recruited_elite=config.recruited_elite,
        recruited_selected=config.recruited_selected,
    )

    best_trip, best_length, history = ba.run()

    # Assignment PDF Check: Validate that every returned tour contains every city once
    assert len(best_trip) == len(dataset), "Tour length mismatch!"
    assert len(set(best_trip)) == len(dataset), "Tour contains duplicate cities!"
    assert best_trip[0] == 0, "City 1 is not the starting depot!"

    runtime = time.perf_counter() - start_time

    optimum = config.optimal_lengths[instance_name]
    gap_percent = 100.0 * (best_length - optimum) / optimum

    return {
        "instance": instance_name,
        "operator": operator_name,
        "seed": seed,
        "best_length": best_length,
        "optimal_length": optimum,
        "gap_percent": gap_percent,
        "runtime_seconds": runtime,
        "evaluations": evaluator.eval_count,
        "best_trip": best_trip,
        "history": history,
    }

def run_all_experiments(config: ExperimentConfig) -> list[dict]:
    operators = {
        "swap": swap_operator,
        "reversal": reverse_operator,
    }
    all_results = []
    total_runs = len(config.instances) * len(operators) * len(config.seeds)
    run_counter = 0
    
    for instance_name, file_path in config.instances:
        print(f"\n--- Loading {instance_name} ({file_path}) ---")
        instance = TSPInstance()
        instance.read_file(file_path)

        if not instance.validate_dataset_metadata():
            raise ValueError(f"Instance validation failed for {file_path}")

        dist_matrix = create_distance_matrix(instance.dataset)

        for operator_name, operator_func in operators.items():
            for seed in config.seeds:
                run_counter += 1
                print(f"[{run_counter}/{total_runs}] {instance_name} | {operator_name} | seed={seed}...", end="", flush=True)

                result = run_single_trial(
                    instance_name,
                    instance.dataset,
                    dist_matrix,
                    operator_name,
                    operator_func,
                    seed,
                    config
                )

                all_results.append(result)

                print(f" Best: {result['best_length']} | Gap: {result['gap_percent']:.2f}% | Time: {result['runtime_seconds']:.2f}s")

    return all_results


def save_results_to_disk(
    results: list[dict],
    json_path: str = "BA/Part_A/results/results_detailed.json",
    csv_path: str = "BA/Part_A/results/results_summary.csv"
):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_fields = [
        "instance",
        "operator",
        "seed",
        "best_length",
        "optimal_length",
        "gap_percent",
        "runtime_seconds",
        "evaluations"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r[field] for field in summary_fields})

if __name__ == "__main__":
    config = ExperimentConfig()
    print(
        "Starting Part A experiments (120 runs: 6 instances x 2 operators x 10"
        " seeds)..."
    )

    results = run_all_experiments(config)

    save_results_to_disk(results)
    print("\nAll 120 TSP experiments completed and saved!")

    #---AI Generated Start---
    print("\nGenerating figures and summary tables...")

    csv_file = "BA/Part_A/results/results_summary.csv"
    json_file = "BA/Part_A/results/results_detailed.json"
    results_dir = "BA/Part_A/results"

    summary_df = generate_summary_table(csv_file)
    plot_convergence("berlin52", results, optimal_value=7542, output_dir=results_dir)
    plot_convergence("ch150", results, optimal_value=6528, output_dir=results_dir)
    plot_best_tour(
        "berlin52",
        "BA/Part_A/data/berlin52.tsp",
        results,
        output_dir=results_dir,
    )

    print("\nAll Part A experiments, tables, and plots are complete!")
    #---AI Generated End---