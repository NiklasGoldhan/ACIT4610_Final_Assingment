import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ES.base import PortfolioData, ESBase, BUDGET
from ES.one_plus_one import OnePlusOneES
from ES.population_es import MuPlusLambdaES, MuCommaLambdaES

"""
Runs every ES variant on all 15 test cases with 10 seeds each and writes
the raw results, the summary table, the statistical tests and the
convergence and step-size plots.

The gap columns compare against the OR-Library efficient frontiers
(data/portef1-5.txt, https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/).

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

VARIANTS = {
    "(1+1)-ES": OnePlusOneES,
    "(mu+lambda)-ES": MuPlusLambdaES,
    "(mu,lambda)-ES": MuCommaLambdaES,
}

# Pairs of variants to compare with the Wilcoxon signed-rank test.
COMPARISONS = [
    ("(1+1)-ES", "(mu+lambda)-ES"),
    ("(1+1)-ES", "(mu,lambda)-ES"),
    ("(mu+lambda)-ES", "(mu,lambda)-ES"),
]


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
            ref_variance = orlib_frontier_variance(dataset, Rmin)
            test_cases.append((case_name, problem, Rmin, ref_variance))

    return test_cases


def orlib_frontier_variance(dataset, Rmin):
    """
    Optimal variance at return Rmin, read from the OR-Library efficient frontier.
    Each portef file holds 2000 (mean return, variance) points of the exact
    long-only frontier; we interpolate linearly between neighbouring points.
    """
    frontier = np.loadtxt(DATA_DIRECTORY / f"portef{dataset.removeprefix('port')}.txt")
    frontier = frontier[np.argsort(frontier[:, 0])]
    returns, variances = frontier[:, 0], frontier[:, 1]
    # Below the minimum-variance portfolio's return the return constraint is not active.
    if Rmin <= returns[np.argmin(variances)]:
        return float(variances.min())
    return float(np.interp(Rmin, returns, variances))


def last_improvement(history):
    """Evaluation number (1-based) at which the best-so-far value last improved."""
    return int(np.argmax(history <= history[-1])) + 1


def run_case(case_name, problem, Rmin, ref_variance):
    """
    Run every ES variant on one test case, once per seed.
    Returns one row per run, plus the best-so-far and median-sigma history of every run.
    """
    rows = []
    histories = {}
    sigma_histories = {}

    for name, variant in VARIANTS.items():
        histories[name] = []
        sigma_histories[name] = []
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
                "ref_variance": ref_variance,
                "gap_percent": 100 * (result.variance / ref_variance - 1),
                "last_improvement": last_improvement(result.history),
                "weights": " ".join(f"{w:.10g}" for w in result.weights),
            })
            histories[name].append(result.history)
            sigma_histories[name].append(result.sigma_history)

            print(f"{case_name:<16} {name:<16} seed {seed}  variance {result.variance:.6e}  "
                  f"valid {valid}  {result.runtime:.2f} s")

    return rows, histories, sigma_histories


def summarize(rows):
    """
    Make one summary row per test case and variant.
    Variance statistics use valid runs only, runtime uses all runs.
    """
    summary = []
    for i in range(0, len(rows), len(SEEDS)):
        runs = rows[i:i + len(SEEDS)]   # rows come in blocks of one variant on one case
        valid_runs = [r for r in runs if r["valid"]]
        variances = [r["variance"] for r in valid_runs]
        runtimes = [r["runtime"] for r in runs]
        case_name = runs[0]["case_name"]
        ref_variance = runs[0]["ref_variance"]
        mean_variance = np.mean(variances) if variances else np.nan

        summary.append({
            "case_name": case_name,
            "variant": runs[0]["variant"],
            "assets": runs[0]["assets"],
            "Rmin": runs[0]["Rmin"],
            "best_variance": np.min(variances) if variances else np.nan,
            "mean_variance": mean_variance,
            "std_variance": np.std(variances, ddof=1) if len(variances) > 1 else np.nan,
            "valid_runs": f"{len(variances)}/{len(runs)}",
            "mean_runtime": np.mean(runtimes),
            "ref_variance": ref_variance,
            "best_gap_percent": 100 * (np.min(variances) / ref_variance - 1) if variances else np.nan,
            "mean_gap_percent": 100 * (mean_variance / ref_variance - 1),
            "mean_last_improvement": np.mean([r["last_improvement"] for r in runs]),
        })

    return summary


def wilcoxon_signed_rank(x, y):
    """
    Exact two-sided Wilcoxon signed-rank test for paired samples (small n).
    Zero differences are dropped and tied absolute differences get average ranks.
    Returns (W+, p-value, number of non-zero pairs).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    m = d.size
    if m == 0:
        return 0.0, 1.0, 0

    # Average ranks of |d|.
    order = np.argsort(np.abs(d))
    sorted_abs = np.abs(d)[order]
    ranks = np.empty(m)
    start = 0
    while start < m:
        end = start
        while end + 1 < m and sorted_abs[end + 1] == sorted_abs[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1

    w_plus = ranks[d > 0].sum()
    centre = ranks.sum() / 2
    observed = abs(w_plus - centre)

    # Exact null distribution: every sign pattern is equally likely.
    extreme = 0
    for signs in itertools.product((0, 1), repeat=m):
        if abs(ranks[np.array(signs, dtype=bool)].sum() - centre) >= observed - 1e-12:
            extreme += 1
    return float(w_plus), extreme / 2 ** m, m


def compare_variants(rows, alpha=0.05):
    """
    Test every pair in COMPARISONS on every test case.
    Runs are paired by seed: with the same seed both variants start from the same
    initial parents and draw the same random numbers until they first differ.
    """
    tests = []
    for name_a, name_b in COMPARISONS:
        for case_name in dict.fromkeys(r["case_name"] for r in rows):
            a = {r["seed"]: r["variance"] for r in rows if r["case_name"] == case_name and r["variant"] == name_a}
            b = {r["seed"]: r["variance"] for r in rows if r["case_name"] == case_name and r["variant"] == name_b}
            seeds = sorted(a.keys() & b.keys())
            if not seeds:
                continue
            x = [a[s] for s in seeds]
            y = [b[s] for s in seeds]
            w_plus, p_value, pairs = wilcoxon_signed_rank(x, y)

            if p_value >= alpha:
                better = "no significant difference"
            elif np.median(np.subtract(x, y)) < 0:
                better = name_a
            else:
                better = name_b

            tests.append({
                "case_name": case_name,
                "variant_a": name_a,
                "variant_b": name_b,
                "a_mean_variance": np.mean(x),
                "b_mean_variance": np.mean(y),
                "pairs": pairs,
                "W_plus": w_plus,
                "p_value": p_value,
                "better_at_0.05": better,
            })
    return tests


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


def save_sigma(sigma_histories, case_name):
    """Save and plot the median parent sigma per generation, mean over the seeds."""
    # (1+1)-ES has no sigma_history (single scalar step size), so skip it here.
    sigma_histories = {name: runs for name, runs in sigma_histories.items()
                       if runs and runs[0] is not None}
    log_path = RESULTS_DIRECTORY / f"sigma_{case_name}.npz"
    np.savez_compressed(log_path, **sigma_histories)
    print(f"Saved {log_path}")

    plt.figure(figsize=(8, 5))
    for name, runs in sigma_histories.items():
        mean_sigma = np.mean(runs, axis=0)
        plt.plot(np.arange(len(mean_sigma)), mean_sigma, label=name)

    plt.yscale("log")
    plt.title(f"Step size on {case_name} (mean of {len(SEEDS)} runs)")
    plt.xlabel("Generation")
    plt.ylabel("Median parent sigma (log scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIRECTORY / f"sigma_{case_name}.png"
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
    for case_name, problem, Rmin, ref_variance in test_cases:
        rows, histories, sigma_histories = run_case(case_name, problem, Rmin, ref_variance)
        all_rows += rows
        if case_name == CONVERGENCE_CASE:
            save_convergence(histories, case_name)
            save_sigma(sigma_histories, case_name)

    # A single case gets its own file names so it does not overwrite the full run.
    suffix = f"_{args.case}" if args.case else ""
    summary = summarize(all_rows)
    tests = compare_variants(all_rows)
    write_csv(all_rows, RESULTS_DIRECTORY / f"raw_runs{suffix}.csv")
    write_csv(summary, RESULTS_DIRECTORY / f"summary{suffix}.csv")
    write_csv(tests, RESULTS_DIRECTORY / f"stat_tests{suffix}.csv")

    print(f"\n{'case':<16} {'variant':<16} {'best':>12} {'mean':>12} {'std':>12} "
          f"{'gap':>8} {'valid':>6} {'time':>7}")
    for row in summary:
        print(f"{row['case_name']:<16} {row['variant']:<16} "
              f"{row['best_variance']:>12.4e} {row['mean_variance']:>12.4e} "
              f"{row['std_variance']:>12.4e} {row['mean_gap_percent']:>7.2f}% "
              f"{row['valid_runs']:>6} {row['mean_runtime']:>6.2f}s")

    for name_a, name_b in COMPARISONS:
        print(f"\n{name_a} vs {name_b}")
        print(f"{'case':<16} {'p-value':>8}  better at 0.05")
        for test in tests:
            if test["variant_a"] == name_a and test["variant_b"] == name_b:
                print(f"{test['case_name']:<16} {test['p_value']:>8.4f}  {test['better_at_0.05']}")


if __name__ == "__main__":
    main()
