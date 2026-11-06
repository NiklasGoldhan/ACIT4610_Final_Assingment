import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ES.base import PortfolioData, ESBase, BUDGET
from ES.population_es import MuPlusLambdaES, MuCommaLambdaES

"""
Runs every ES variant on all 15 test cases with 10 seeds each and writes
the raw results, summary table and convergence plot.

Run everything:      python -m ES.benchmark
Run one test case:   python -m ES.benchmark --case port1_alpha050
"""

ES_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ES_DIRECTORY / "data"
RESULTS_DIRECTORY = ES_DIRECTORY / "results"
PLOTS_DIRECTORY = ES_DIRECTORY / "plots"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CONVERGENCE_CASE = "port5_alpha050"

# Add OnePlusOneES here when one_plus_one.py is finished.
VARIANTS = {
    "(mu+lambda)-ES": MuPlusLambdaES,
    "(mu,lambda)-ES": MuCommaLambdaES,
}


# Move to benchmark.py later, just used for validation of port1.txt calculations
# as stated in the assignment.
def parser_validation_port1(mean_return, cov, return_targets):
        """
        Validate parser and calculations for port1.txt.
        """
        equal_weights = np.full(mean_return.size, 1 / mean_return.size)

        Req = mean_return @ equal_weights
        equal_variance = equal_weights @ cov @ equal_weights
        Rmin_050 = return_targets[0.50]

        expected = {
            "equal return": (Req, 0.003504064516129),
            "equal variance": (equal_variance, 0.001130937943724),
            "alpha 0.50 target": (Rmin_050, 0.007184532258065),
        }
        for name, (actual, expected_value) in expected.items():
            if not np.isclose(actual, expected_value):
                raise ValueError(
                    f"{name} validation failed: "
                    f"expected {expected_value}, got {actual}"
                )

        print("port1 validation passed")


def generate_test_cases():
    """
    Load the five datasets and make the 15 test cases (3 return targets each).
    Also runs the parser validation checkpoint on port1.txt.
    """
    datasets = ["port1.txt", "port2.txt", "port3.txt", "port4.txt", "port5.txt"]
    test_cases = []

    for data in datasets:
        problem = PortfolioData.load_data(DATA_DIRECTORY / data)
        # Rmin is not used here, target_calculations only needs the mean returns.
        _, return_targets = ESBase(problem, Rmin=0).target_calculations(problem.mean_return)

        if data == "port1.txt":
            parser_validation_port1(problem.mean_return, problem.cov, return_targets)

        for alpha, rmin in return_targets.items():
            test_cases.append({
                "case_name": f"{Path(data).stem}_alpha{round(alpha * 100):03d}",
                "dataset": data,
                "alpha": alpha,
                "Rmin": rmin,
                "problem": problem,
            })

    return test_cases


def run_case(case, seeds):
    """
    Run every ES variant on one test case, once per seed.
    Returns one row per run and the best-so-far history of every run.
    """
    rows = []
    histories = {name: [] for name in VARIANTS}

    for name, variant in VARIANTS.items():
        for seed in seeds:
            es = variant(case["problem"], case["Rmin"], seed=seed)
            result = es.run()
            valid = es.is_feasible(result.weights)

            rows.append({
                "case_name": case["case_name"],
                "variant": name,
                "seed": seed,
                "assets": es.n,
                "Rmin": case["Rmin"],
                "valid": valid,
                "variance": result.variance,
                "expected_return": result.expected_return,
                "evaluations": result.evaluations,
                "runtime": result.runtime,
                "weights": " ".join(f"{w:.10g}" for w in result.weights),
            })
            histories[name].append(result.history)

            print(
                f"{case['case_name']:<16} {name:<16} seed {seed}  "
                f"variance {result.variance:.6e}  valid {valid}  "
                f"{result.runtime:.2f} s"
            )

    return rows, histories


def summarize(rows):
    """
    Make one summary row per test case and variant.
    Variance statistics use valid runs only, runtime uses all runs.
    """
    summary = []
    keys = []
    for row in rows:
        if (row["case_name"], row["variant"]) not in keys:
            keys.append((row["case_name"], row["variant"]))

    for case_name, variant in keys:
        runs = [r for r in rows if r["case_name"] == case_name and r["variant"] == variant]
        variances = np.array([r["variance"] for r in runs if r["valid"]])

        summary.append({
            "case_name": case_name,
            "variant": variant,
            "assets": runs[0]["assets"],
            "Rmin": runs[0]["Rmin"],
            "best_variance": variances.min() if variances.size else np.nan,
            "mean_variance": variances.mean() if variances.size else np.nan,
            "std_variance": variances.std(ddof=1) if variances.size > 1 else np.nan,
            "valid_runs": f"{variances.size}/{len(runs)}",
            "mean_runtime": np.mean([r["runtime"] for r in runs]),
        })

    return summary


def write_csv(rows, file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {file_path}")


def save_convergence(histories, case_name):
    """
    Save the best-so-far histories and plot the mean across seeds
    for every variant in one figure.
    """
    log_path = RESULTS_DIRECTORY / f"convergence_{case_name}.npz"
    np.savez_compressed(
        log_path,
        **{name: np.array(runs) for name, runs in histories.items()},
    )
    print(f"Saved {log_path}")

    plt.figure(figsize=(8, 5))
    for name, runs in histories.items():
        mean_history = np.mean(runs, axis=0)
        evaluations = np.arange(1, mean_history.size + 1)
        plt.plot(evaluations, mean_history, label=name)

    plt.yscale("log")
    plt.title(f"Convergence on {case_name} (mean of {len(runs)} runs)")
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
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds to use")
    args = parser.parse_args()

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    PLOTS_DIRECTORY.mkdir(exist_ok=True)

    test_cases = generate_test_cases()
    if args.case:
        test_cases = [c for c in test_cases if c["case_name"] == args.case]
        if not test_cases:
            raise ValueError(f"Unknown test case {args.case}")
    seeds = SEEDS[:args.seeds]

    print(f"{len(test_cases)} test case(s), {len(seeds)} seed(s), {BUDGET} evaluations per run\n")

    all_rows = []
    for case in test_cases:
        rows, histories = run_case(case, seeds)
        all_rows.extend(rows)
        if case["case_name"] == CONVERGENCE_CASE:
            save_convergence(histories, case["case_name"])

    # A single case gets its own file names so it does not overwrite the full run.
    suffix = f"_{args.case}" if args.case else ""
    write_csv(all_rows, RESULTS_DIRECTORY / f"raw_runs{suffix}.csv")
    summary = summarize(all_rows)
    write_csv(summary, RESULTS_DIRECTORY / f"summary{suffix}.csv")

    print()
    print(f"{'case':<16} {'variant':<16} {'best':>12} {'mean':>12} {'std':>12} {'valid':>6} {'time':>7}")
    for row in summary:
        print(
            f"{row['case_name']:<16} {row['variant']:<16} "
            f"{row['best_variance']:>12.4e} {row['mean_variance']:>12.4e} "
            f"{row['std_variance']:>12.4e} {row['valid_runs']:>6} {row['mean_runtime']:>6.2f}s"
        )


if __name__ == "__main__":
    main()
