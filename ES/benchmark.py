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
CONVERGENCE_CASES = {f"{dataset}_alpha050" for dataset in DATASETS}
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
        "evaluations": result.num_evaluations,
        "runtime": result.runtime,
        "ref_variance": ref_variance,
        "gap_percent": 100 * (result.variance / ref_variance - 1),
        "weights": " ".join(f"{w:.10g}" for w in result.weights),
    }
    return row, result.history


def run_all_experiments(test_cases: list[tuple]) -> tuple[list[dict], dict]:
    """Run all trials and keep histories per plotted case, variant, and seed."""
    rows = []
    # Full experiment: one alpha050 plot per dataset. A single-case run plots that case.
    convergence_histories = {
        case_name: {name: [] for name in VARIANTS}
        for case_name, *_ in test_cases
        if case_name in CONVERGENCE_CASES or len(test_cases) == 1
    }
    total_runs = len(test_cases) * len(VARIANTS) * len(SEEDS)
    run_counter = 0

    for case_name, problem, Rmin, ref_variance in test_cases:
        for variant_name in VARIANTS:
            for seed in SEEDS:
                run_counter += 1
                row, history = run_single_trial(case_name, problem, Rmin, ref_variance,
                                                variant_name, seed)
                rows.append(row)
                if case_name in convergence_histories:
                    convergence_histories[case_name][variant_name].append(history)

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
        variances = [r["variance"] for r in runs
                     if r["valid"] and np.isfinite(r["variance"])]
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


def load_summary(file_path: Path) -> list[dict]:
    """Read a saved summary CSV back, so its table can be rebuilt without rerunning."""
    numeric = ("Rmin", "best_variance", "mean_variance", "std_variance", "mean_runtime",
               "ref_variance", "best_gap_percent", "mean_gap_percent")
    with open(file_path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["assets"] = int(row["assets"])
        for key in numeric:
            row[key] = float(row[key])
    return rows


# AI-generated (Claude Code): summary table required by the assignment.
def save_summary_table(summary: list[dict], suffix: str = "") -> None:
    """Save one table row per case and variant with the statistics the assignment asks for."""
    def number(value: float, style: str) -> str:
        return format(value, style) if np.isfinite(value) else "N/A"

    lookup = {(r["case_name"], r["variant"]): r for r in summary}
    case_names = sorted({r["case_name"] for r in summary})
    headers = ["Case", "Assets", "R_min", "Variant", "Best variance", "Mean variance",
               "Std", "Feasible", "Mean runtime (s)", "Mean gap (%)"]

    table_rows, lowest_mean, case_groups = [], [], []
    for case_index, case in enumerate(case_names):
        rows = [lookup[case, name] for name in VARIANTS if (case, name) in lookup]
        means = [r["mean_variance"] for r in rows if np.isfinite(r["mean_variance"])]
        best_mean = min(means) if means else np.nan
        for index, row in enumerate(rows):
            first = index == 0
            gap = number(row["mean_gap_percent"], ".2f").replace("-0.00", "0.00")
            table_rows.append([
                case if first else "",
                str(row["assets"]) if first else "",
                number(row["Rmin"], ".6f") if first else "",
                row["variant"],
                number(row["best_variance"], ".4e"),
                number(row["mean_variance"], ".4e"),
                number(row["std_variance"], ".2e"),
                row["valid_runs"],
                number(row["mean_runtime"], ".2f"),
                gap,
            ])
            lowest_mean.append(row["mean_variance"] == best_mean)
            case_groups.append(case_index)

    fig, ax = plt.subplots(figsize=(14, 0.3 * (len(table_rows) + 1) + 1.8))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center",
                     colWidths=[0.14, 0.06, 0.08, 0.15, 0.11, 0.11, 0.09, 0.07, 0.10, 0.09],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for column in range(len(headers)):
        table[0, column].set_facecolor("#e4e9ef")
        table[0, column].get_text().set_weight("bold")
    mean_column = headers.index("Mean variance")
    for row_index, (group, is_lowest) in enumerate(zip(case_groups, lowest_mean), start=1):
        if group % 2:   # shade every other case so its three rows read as one block
            for column in range(len(headers)):
                table[row_index, column].set_facecolor("#f4f6f8")
        if is_lowest:
            table[row_index, mean_column].set_facecolor("#d7efda")
            table[row_index, mean_column].get_text().set_weight("bold")
    ax.set_title(f"Summary of results per case and variant ({len(SEEDS)} runs each)", pad=12)
    fig.text(0.5, 0.015,
             "Returns and variances are weekly decimal values. Variance statistics use feasible runs only; "
             "Std is the sample standard deviation (n − 1).\n"
             "Mean gap = 100 × (mean variance / optimal variance − 1), with the optimum interpolated from "
             "the OR-Library efficient frontier.\n"
             "Green: lowest mean variance in the case, compared at full precision; displayed values are rounded. "
             "Mean runtime is averaged over all runs.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    path = PLOTS_DIRECTORY / f"summary_table{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def load_stat_tests(file_path: Path) -> list[dict]:
    """Read a saved Wilcoxon CSV back, so its table can be rebuilt without rerunning."""
    with open(file_path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["pairs"] = int(row["pairs"])
        row["p_value"] = float(row["p_value"])
    return rows


# AI-generated (Claude Code): Wilcoxon results table for the report.
def save_wilcoxon_table(tests: list[dict], suffix: str = "") -> None:
    """Save the paired Wilcoxon p-value and result for every case and variant pair."""
    lookup = {(t["case_name"], t["variant_a"], t["variant_b"]): t for t in tests}
    case_names = sorted({t["case_name"] for t in tests})
    headers = ["Case"]
    for name_a, name_b in COMPARISONS:
        headers.extend([f"{name_a} vs\n{name_b}\np-value", "Better"])

    table_rows, significant = [], []
    for case in case_names:
        row, marked = [case], []
        for index, (name_a, name_b) in enumerate(COMPARISONS):
            test = lookup.get((case, name_a, name_b))
            if test is None:
                row.extend(["N/A", "N/A"])
                continue
            better = test["better_at_0.05"]
            is_significant = better != "no significant difference"
            row.extend([f"{test['p_value']:.3f}", better if is_significant else "n.s."])
            if is_significant:
                marked.extend([1 + 2 * index, 2 + 2 * index])
        table_rows.append(row)
        significant.append(marked)

    fig, ax = plt.subplots(figsize=(14, 0.38 * len(case_names) + 3))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center",
                     colWidths=[0.16, 0.13, 0.14, 0.13, 0.14, 0.15, 0.15],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for column in range(len(headers)):
        table[0, column].set_facecolor("#e4e9ef")
        table[0, column].get_text().set_weight("bold")
        table[0, column].set_height(table[0, column].get_height() * 2.2)
    for row_index, columns in enumerate(significant, start=1):
        for column in columns:
            table[row_index, column].set_facecolor("#d7efda")
    ax.set_title("Wilcoxon signed-rank tests between the ES variants, paired by seed", pad=12)
    fig.text(0.5, 0.02,
             f"Two-sided test per case on the final variances of the {len(SEEDS)} paired runs; "
             "pairs with equal results are dropped.\n"
             f"Green: significant at p < {SIGNIFICANCE_LEVEL}. Better: the variant with the lower variance "
             "(median paired difference). n.s.: no significant difference.\n"
             f"With {len(SEEDS)} pairs, the smallest possible p-value is {2 / 2 ** len(SEEDS):.3f}.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98))
    path = PLOTS_DIRECTORY / f"wilcoxon_table{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


# AI-assisted illustrations: best candidate table and best/mean comparison heatmaps.
def save_best_results_table(rows: list[dict], suffix: str = "") -> None:
    """Save each variant's best feasible variance and seed for every available case."""
    case_names = sorted({r["case_name"] for r in rows})
    best = {}
    for row in rows:
        if not row["valid"] or not np.isfinite(row["variance"]):
            continue
        key = row["case_name"], row["variant"]
        # Equal variances within a variant select the smallest seed reproducibly.
        if key not in best or (row["variance"], row["seed"]) < (best[key]["variance"], best[key]["seed"]):
            best[key] = row

    table_rows, csv_rows, winners = [], [], []
    for case in case_names:
        candidates = [best[case, name]["variance"] for name in VARIANTS if (case, name) in best]
        lowest = min(candidates) if candidates else np.nan
        table_row, csv_row, winning_columns = [case], {"case_name": case}, []
        winning_names = []
        for index, name in enumerate(VARIANTS):
            candidate = best.get((case, name))
            value = candidate["variance"] if candidate else np.nan
            seed = candidate["seed"] if candidate else ""
            csv_row[f"{name}_best_variance"] = value
            csv_row[f"{name}_seed"] = seed
            table_row.extend([f"{value:.8e}" if candidate else "N/A", str(seed) if candidate else "N/A"])
            if value == lowest:
                winning_columns.extend([1 + 2 * index, 2 + 2 * index])
                winning_names.append(name)
        csv_row["best_variants"] = " / ".join(winning_names)
        csv_rows.append(csv_row)
        table_rows.append(table_row)
        winners.append(winning_columns)
    write_csv(csv_rows, RESULTS_DIRECTORY / f"best_candidates{suffix}.csv")

    headers = ["Case"]
    for name in VARIANTS:
        headers.extend([f"{name}\nBest variance", "Seed"])
    fig, ax = plt.subplots(figsize=(15, max(3, 0.38 * len(case_names) + 2)))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center",
                     colWidths=[0.19, 0.22, 0.05, 0.22, 0.05, 0.22, 0.05],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for column in range(len(headers)):
        table[0, column].set_facecolor("#e4e9ef")
        table[0, column].get_text().set_weight("bold")
        table[0, column].set_height(table[0, column].get_height() * 1.6)
    for row_index, columns in enumerate(winners, start=1):
        for column in columns:
            table[row_index, column].set_facecolor("#d7efda")
            table[row_index, column].get_text().set_weight("bold")
    ax.set_title("Best feasible candidates across seeds", pad=18)
    fig.text(0.5, 0.02, "Green cells: lowest variance per case (including exact ties). "
             "Seed identifies the portfolio in raw_runs.csv.\n"
             "Variance uses squared decimal returns. N/A: no valid, finite result. "
             "Within-variant ties select the smallest seed.\n"
             "Winners use full precision; displayed values are rounded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    path = PLOTS_DIRECTORY / f"best_candidates{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def save_comparison_heatmaps(summary: list[dict], suffix: str = "") -> None:
    """Plot best and mean variance differences from each case's best tested value."""
    case_names = sorted({r["case_name"] for r in summary})
    lookup = {(r["case_name"], r["variant"]): r for r in summary}
    differences = []
    for metric in ("best_variance", "mean_variance"):
        matrix = np.array([
            [lookup.get((case, name), {}).get(metric, np.nan) for name in VARIANTS]
            for case in case_names
        ], dtype=float)
        relative = np.full_like(matrix, np.nan)
        for index, values in enumerate(matrix):
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            lowest = np.min(values[finite])
            if lowest > 0:
                relative[index, finite] = 100 * (values[finite] / lowest - 1)
            elif lowest == 0:
                relative[index, finite & (values == 0)] = 0
        differences.append(relative)
    write_csv([
        {"case_name": case, "variant": name,
         "best_difference_percent": differences[0][i, j],
         "mean_difference_percent": differences[1][i, j]}
        for i, case in enumerate(case_names) for j, name in enumerate(VARIANTS)
    ], RESULTS_DIRECTORY / f"comparison_heatmaps{suffix}.csv")

    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in differences])
    maximum = max(1.0, float(np.max(finite_values))) if finite_values.size else 1.0
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("lightgray")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, max(4.5, 0.42 * len(case_names) + 2.5)),
                             layout="constrained")
    for ax, matrix, title in zip(axes, differences, ("Best variance across seeds", "Mean variance across seeds")):
        plot = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=maximum)
        ax.set_title(title)
        ax.set_xticks(range(len(VARIANTS)), list(VARIANTS), rotation=12)
        ax.set_yticks(range(len(case_names)), case_names)
        ax.set_xlabel("ES variant")
        for i in range(len(case_names)):
            for j in range(len(VARIANTS)):
                value = matrix[i, j]
                label = f"{value:.2f}%" if np.isfinite(value) else "N/A"
                if 0 < value < 0.005:
                    label = "<0.01%"
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                        color="white" if np.isfinite(value) and value > maximum / 2 else "black")
    axes[0].set_ylabel("Dataset and return target")
    fig.colorbar(plot, ax=axes, label="Difference from best tested value in that case (%)", shrink=0.85)
    fig.suptitle("Solution quality: strongest result and typical performance")
    fig.supxlabel("100 × (variant value / lowest case value − 1), calculated separately for each panel.\n"
                  "Valid, finite runs only. 0% is best tested; this is not a gap to the true optimum.\n"
                  "N/A: unavailable result or undefined percentage for a zero baseline.", fontsize=9)
    path = PLOTS_DIRECTORY / f"comparison_heatmaps{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


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
    parser.add_argument("--tables-only", action="store_true",
                        help="rebuild the summary and Wilcoxon tables from the saved CSVs without rerunning")
    args = parser.parse_args()

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    PLOTS_DIRECTORY.mkdir(exist_ok=True)

    # Single-case runs get their own file names.
    suffix = f"_{args.case}" if args.case else ""
    if args.tables_only:
        save_summary_table(load_summary(RESULTS_DIRECTORY / f"summary{suffix}.csv"), suffix)
        save_wilcoxon_table(load_stat_tests(RESULTS_DIRECTORY / f"stat_tests{suffix}.csv"), suffix)
        return

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

    write_csv(rows, RESULTS_DIRECTORY / f"raw_runs{suffix}.csv")
    write_csv(summary, RESULTS_DIRECTORY / f"summary{suffix}.csv")
    write_csv(tests, RESULTS_DIRECTORY / f"stat_tests{suffix}.csv")
    save_summary_table(summary, suffix)
    save_wilcoxon_table(tests, suffix)
    save_best_results_table(rows, suffix)
    save_comparison_heatmaps(summary, suffix)
    for case_name, histories in convergence_histories.items():
        save_convergence(histories, case_name)

    print_summary(summary)


if __name__ == "__main__":
    main()
