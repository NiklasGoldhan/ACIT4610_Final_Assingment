import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

from ES.one_plus_one import OnePlusOneES
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ES.base import PortfolioData, ESBase, BUDGET
from ES.population_es import MuPlusLambdaES, MuCommaLambdaES

"""
Runs every ES variant on all 15 test cases with 10 seeds each and writes
the raw results, the summary table and the convergence plot.

Run everything:      python -m ES.benchmark
Run one test case:   python -m ES.benchmark --case port1_alpha050
"""

ES_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ES_DIRECTORY / "data"
RESULTS_DIRECTORY = ES_DIRECTORY / "results"
PLOTS_DIRECTORY = ES_DIRECTORY / "plots"

DATASETS = ["port1", "port2", "port3", "port4", "port5"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CONVERGENCE_CASE = "port5_alpha050"

# (1+1)-ES is left out for now: its run() returns (weights, variance) instead of
# an ESRunResult, so it has no runtime or best-so-far history. Add it back later

VARIANTS = {
    "(1+1)-ES": OnePlusOneES,
    "(mu+lambda)-ES": MuPlusLambdaES,
    "(mu,lambda)-ES": MuCommaLambdaES,
}


def parser_validation_port1(problem, Req, targets):
    """Check the port1.txt parser against the values given in the assignment."""
    equal_weights = np.full(problem.mean_return.size, 1 / problem.mean_return.size)
    equal_variance = equal_weights @ problem.cov @ equal_weights

    assert np.isclose(Req, 0.003504064516129), "equal return is wrong"
    assert np.isclose(equal_variance, 0.001130937943724), "equal variance is wrong"
    assert np.isclose(targets[0.50], 0.007184532258065), "alpha 0.50 target is wrong"
    print("port1 validation passed")


def load_test_cases():
    """Load the five datasets and make the 15 test cases (3 return targets each)."""
    test_cases = []
    for dataset in DATASETS:
        problem = PortfolioData.load_data(DATA_DIRECTORY / f"{dataset}.txt")
        # target_calculations does not use Rmin, so None is fine here.
        Req, targets = ESBase(problem, None).target_calculations(problem.mean_return)

        if dataset == "port1":
            parser_validation_port1(problem, Req, targets)

        for alpha, Rmin in targets.items():
            case_name = f"{dataset}_alpha{round(alpha * 100):03d}"   # e.g. port1_alpha050
            test_cases.append((case_name, problem, Rmin))

    return test_cases


def run_case(case_name, problem, Rmin):
    """
    Run every ES variant on one test case, once per seed.
    Returns one row per run and the best-so-far history of every run.
    """
    rows = []
    histories = {}

    for name, variant in VARIANTS.items():
        histories[name] = []
        for seed in SEEDS:
            es = variant(problem, Rmin, seed=seed)
            result = es.run()
            valid = es.is_feasible(result.weights)

            rows.append({
                "case_name": case_name,
                "variant": name,
                "seed": seed,
                "assets": es.n,
                "Rmin": Rmin,
                "valid": valid,
                "variance": result.variance,
                "expected_return": result.expected_return,
                "evaluations": result.evaluations,
                "runtime": result.runtime,
                "weights": " ".join(f"{w:.10g}" for w in result.weights),
            })
            histories[name].append(result.history)

            print(f"{case_name:<16} {name:<16} seed {seed}  variance {result.variance:.6e}  "
                  f"valid {valid}  {result.runtime:.2f} s")

    return rows, histories


def summarize(rows):
    """
    Make one summary row per test case and variant.
    Variance statistics use valid runs only, runtime uses all runs.
    """
    summary = []
    for i in range(0, len(rows), len(SEEDS)):
        runs = rows[i:i + len(SEEDS)]   # rows come in blocks of one variant on one case
        variances = [r["variance"] for r in runs if r["valid"]]
        runtimes = [r["runtime"] for r in runs]

        summary.append({
            "case_name": runs[0]["case_name"],
            "variant": runs[0]["variant"],
            "assets": runs[0]["assets"],
            "Rmin": runs[0]["Rmin"],
            "best_variance": np.min(variances) if variances else np.nan,
            "mean_variance": np.mean(variances) if variances else np.nan,
            "std_variance": np.std(variances, ddof=1) if len(variances) > 1 else np.nan,
            "valid_runs": f"{len(variances)}/{len(runs)}",
            "mean_runtime": np.mean(runtimes),
        })

    return summary


def write_csv(rows, file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {file_path}")


def save_convergence(histories, case_name):
    """Save the best-so-far logs and plot the mean over the seeds for every variant."""
    log_path = RESULTS_DIRECTORY / f"convergence_{case_name}.npz"
    np.savez_compressed(log_path, **histories)
    print(f"Saved {log_path}")

    plt.figure(figsize=(8, 5))
    for name, runs in histories.items():
        mean_history = np.mean(runs, axis=0)
        evaluations = np.arange(1, len(mean_history) + 1)
        plt.plot(evaluations, mean_history, label=name)

    plt.yscale("log")
    plt.title(f"Convergence on {case_name} (mean of {len(SEEDS)} runs)")
    plt.xlabel("Objective evaluations")
    plt.ylabel("Mean best-so-far portfolio variance (log scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIRECTORY / f"convergence_{case_name}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the ES variants.")
    parser.add_argument("--case", help="run a single test case, e.g. port1_alpha050")
    args = parser.parse_args()

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    PLOTS_DIRECTORY.mkdir(exist_ok=True)

    test_cases = load_test_cases()
    if args.case:
        test_cases = [case for case in test_cases if case[0] == args.case]
        if not test_cases:
            raise ValueError(f"Unknown test case {args.case}")

    print(f"{len(test_cases)} test case(s), {len(SEEDS)} seeds, {BUDGET} evaluations per run\n")

    all_rows = []
    for case_name, problem, Rmin in test_cases:
        rows, histories = run_case(case_name, problem, Rmin)
        all_rows += rows
        if case_name == CONVERGENCE_CASE:
            save_convergence(histories, case_name)

    # A single case gets its own file names so it does not overwrite the full run.
    suffix = f"_{args.case}" if args.case else ""
    summary = summarize(all_rows)
    write_csv(all_rows, RESULTS_DIRECTORY / f"raw_runs{suffix}.csv")
    write_csv(summary, RESULTS_DIRECTORY / f"summary{suffix}.csv")

    print(f"\n{'case':<16} {'variant':<16} {'best':>12} {'mean':>12} {'std':>12} {'valid':>6} {'time':>7}")
    for row in summary:
        print(f"{row['case_name']:<16} {row['variant']:<16} "
              f"{row['best_variance']:>12.4e} {row['mean_variance']:>12.4e} "
              f"{row['std_variance']:>12.4e} {row['valid_runs']:>6} {row['mean_runtime']:>6.2f}s")


if __name__ == "__main__":
    main()
