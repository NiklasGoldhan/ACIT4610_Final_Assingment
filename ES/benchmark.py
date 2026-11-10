"""Run all ES variants on the 15 test cases and save the results."""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from ES.base import PortfolioData, ESBase, BUDGET
from ES.one_plus_one import OnePlusOneES
from ES.population_es import MuPlusLambdaES, MuCommaLambdaES

ES_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ES_DIRECTORY / "data"
RESULTS_DIRECTORY = ES_DIRECTORY / "results"
PLOTS_DIRECTORY = ES_DIRECTORY / "plots"

DATASETS = ["port1", "port2", "port3", "port4", "port5"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CONVERGENCE_CASE = "port5_alpha050"
SIGNIFICANCE_LEVEL = 0.05

VARIANTS = {
    "(1+1)-ES": OnePlusOneES,
    "(mu+lambda)-ES": MuPlusLambdaES,
    "(mu,lambda)-ES": MuCommaLambdaES,
}

# Variant pairs for the Wilcoxon test.
COMPARISONS = [
    ("(1+1)-ES", "(mu+lambda)-ES"),
    ("(1+1)-ES", "(mu,lambda)-ES"),
    ("(mu+lambda)-ES", "(mu,lambda)-ES"),
]


# --- Test cases ---

def parser_validation_port1(problem: PortfolioData, Req: float, targets: dict) -> None:
    """Check port1 against the values in the assignment."""
    equal_weights = np.full(problem.mean_return.size, 1 / problem.mean_return.size)
    equal_variance = equal_weights @ problem.cov @ equal_weights

    assert np.isclose(Req, 0.003504064516129), "equal return is wrong"
    assert np.isclose(equal_variance, 0.001130937943724), "equal variance is wrong"
    assert np.isclose(targets[0.50], 0.007184532258065), "alpha 0.50 target is wrong"
    print("port1 validation passed")


def orlib_frontier_variance(dataset: str, Rmin: float) -> float:
    """Optimal variance at Rmin, interpolated from the OR-Library frontier (portef files)."""
    frontier = np.loadtxt(DATA_DIRECTORY / f"portef{dataset.removeprefix('port')}.txt")
    frontier = frontier[np.argsort(frontier[:, 0])]
    returns, variances = frontier[:, 0], frontier[:, 1]
    # Low targets: the minimum-variance portfolio already meets them.
    if Rmin <= returns[np.argmin(variances)]:
        return float(variances.min())
    return float(np.interp(Rmin, returns, variances))


def load_test_cases() -> list[tuple]:
    """Return the 15 test cases as (case_name, problem, Rmin, optimal variance)."""
    test_cases = []
    for dataset in DATASETS:
        problem = PortfolioData.load_data(DATA_DIRECTORY / f"{dataset}.txt")
        # Rmin is not needed for the targets, so None is fine.
        Req, targets = ESBase.target_calculations(problem.mean_return)
        if dataset == "port1":
            parser_validation_port1(problem, Req, targets)

        for alpha, Rmin in targets.items():
            case_name = f"{dataset}_alpha{round(alpha * 100):03d}"   # e.g. port1_alpha050
            ref_variance = orlib_frontier_variance(dataset, Rmin)
            test_cases.append((case_name, problem, Rmin, ref_variance))

    return test_cases


# --- Running the experiments ---

def run_single_trial(case_name: str, problem: PortfolioData, Rmin: float, ref_variance: float,
                     variant_name: str, seed: int) -> tuple[dict, np.ndarray]:
    """Run one variant with one seed. Returns the result row and best-so-far history."""
    es = VARIANTS[variant_name](problem, Rmin, seed=seed)
    result = es.run()

    row = {
        "case_name": case_name,
        "variant": variant_name,
        "seed": seed,
        "assets": es.n,
        "Rmin": Rmin,
        "valid": es.is_feasible(result.weights),
        "variance": result.variance,
        "expected_return": result.expected_return,
        "evaluations": result.evaluations,
        "runtime": result.runtime,
        "ref_variance": ref_variance,
        "gap_percent": 100 * (result.variance / ref_variance - 1),
        "weights": " ".join(f"{w:.10g}" for w in result.weights),
    }
    return row, result.history


def run_all_experiments(test_cases: list[tuple]) -> tuple[list[dict], dict]:
    """Run every variant, case and seed. Returns all rows and the convergence histories."""
    rows = []
    convergence_histories = {name: [] for name in VARIANTS}
    total_runs = len(test_cases) * len(VARIANTS) * len(SEEDS)
    run_counter = 0

    for case_name, problem, Rmin, ref_variance in test_cases:
        for variant_name in VARIANTS:
            for seed in SEEDS:
                run_counter += 1
                row, history = run_single_trial(case_name, problem, Rmin, ref_variance,
                                                variant_name, seed)
                rows.append(row)
                if case_name == CONVERGENCE_CASE:
                    convergence_histories[variant_name].append(history)

                print(f"[{run_counter}/{total_runs}] {case_name} | {variant_name} | seed {seed}"
                      f" | variance {row['variance']:.4e} | gap {row['gap_percent']:.2f}%"
                      f" | valid {row['valid']} | {row['runtime']:.2f} s")

    return rows, convergence_histories


# --- Summary and statistics ---

def summarize(rows: list[dict]) -> list[dict]:
    """One row per case and variant. Variance stats use valid runs only."""
    groups = {}
    for row in rows:
        groups.setdefault((row["case_name"], row["variant"]), []).append(row)

    summary = []
    for (case_name, variant_name), runs in groups.items():
        variances = [r["variance"] for r in runs if r["valid"]]
        ref_variance = runs[0]["ref_variance"]
        best_variance = np.min(variances) if variances else np.nan
        mean_variance = np.mean(variances) if variances else np.nan

        summary.append({
            "case_name": case_name,
            "variant": variant_name,
            "assets": runs[0]["assets"],
            "Rmin": runs[0]["Rmin"],
            "best_variance": best_variance,
            "mean_variance": mean_variance,
            "std_variance": np.std(variances, ddof=1) if len(variances) > 1 else np.nan,
            "valid_runs": f"{len(variances)}/{len(runs)}",
            "mean_runtime": np.mean([r["runtime"] for r in runs]),
            "ref_variance": ref_variance,
            "best_gap_percent": 100 * (best_variance / ref_variance - 1),
            "mean_gap_percent": 100 * (mean_variance / ref_variance - 1),
        })

    return summary


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, int]:
    """Paired Wilcoxon test. Returns (p-value, number of non-zero pairs)."""
    d = np.subtract(x, y)
    d = d[d != 0]
    if d.size == 0:   # all pairs equal, scipy would return NaN
        return 1.0, 0
    return float(wilcoxon(d, alternative="two-sided").pvalue), d.size


def compare_variants(rows: list[dict]) -> list[dict]:
    """Wilcoxon test for every pair in COMPARISONS on every case, paired by seed."""
    variance = {(r["case_name"], r["variant"], r["seed"]): r["variance"] for r in rows}
    case_names = list(dict.fromkeys(r["case_name"] for r in rows))
    tests = []

    for name_a, name_b in COMPARISONS:
        for case_name in case_names:
            x = [variance[case_name, name_a, s] for s in SEEDS]
            y = [variance[case_name, name_b, s] for s in SEEDS]
            p_value, pairs = wilcoxon_signed_rank(x, y)

            if p_value >= SIGNIFICANCE_LEVEL:
                better = "no significant difference"
            elif np.median(np.subtract(x, y)) < 0:
                better = name_a
            else:
                better = name_b

            tests.append({
                "case_name": case_name,
                "variant_a": name_a,
                "variant_b": name_b,
                "pairs": pairs,
                "p_value": p_value,
                "better_at_0.05": better,
            })

    return tests


# --- Output ---

def write_csv(rows: list[dict], file_path: Path) -> None:
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {file_path}")


def save_convergence(histories: dict, case_name: str) -> None:
    """Save the best-so-far logs and plot the mean per variant."""
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


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'case':<16} {'variant':<16} {'best':>12} {'mean':>12} {'std':>12} "
          f"{'gap':>8} {'valid':>6} {'time':>7}")
    for row in summary:
        print(f"{row['case_name']:<16} {row['variant']:<16} "
              f"{row['best_variance']:>12.4e} {row['mean_variance']:>12.4e} "
              f"{row['std_variance']:>12.4e} {row['mean_gap_percent']:>7.2f}% "
              f"{row['valid_runs']:>6} {row['mean_runtime']:>6.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the ES variants.")
    parser.add_argument("--case", help="run a single test case, e.g. port1_alpha050")
    args = parser.parse_args()

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    PLOTS_DIRECTORY.mkdir(exist_ok=True)

    test_cases = load_test_cases()
    if args.case:   # keep only the requested case (case[0] is its name)
        test_cases = [case for case in test_cases if case[0] == args.case]
    if not test_cases:
        raise ValueError(f"Unknown test case {args.case}")

    print(f"{len(test_cases)} test case(s), {len(VARIANTS)} variants, {len(SEEDS)} seeds, "
          f"{BUDGET} evaluations per run\n")

    rows, convergence_histories = run_all_experiments(test_cases)
    summary = summarize(rows)
    tests = compare_variants(rows)

    # Single-case runs get their own file names.
    suffix = f"_{args.case}" if args.case else ""
    write_csv(rows, RESULTS_DIRECTORY / f"raw_runs{suffix}.csv")
    write_csv(summary, RESULTS_DIRECTORY / f"summary{suffix}.csv")
    write_csv(tests, RESULTS_DIRECTORY / f"stat_tests{suffix}.csv")
    if any(convergence_histories.values()):
        save_convergence(convergence_histories, CONVERGENCE_CASE)

    print_summary(summary)


if __name__ == "__main__":
    main()
