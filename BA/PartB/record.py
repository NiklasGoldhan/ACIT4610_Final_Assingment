import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import Results

# This file is mostly AI-generated, with some small edits.


def write_csv(path: Path, header: list[str], rows: Iterable) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# Summary

def summary_rows(results: Results) -> list[tuple]:
    rows = []
    for (name, dim), records in results.items():
        initial = np.array([r.initial_fx for r in records])
        fitness = np.array([r.fx for r in records])
        times = np.array([r.runtime for r in records])
        rows.append((
            name,
            dim,
            float(initial.mean()),
            float(fitness.min()),
            float(fitness.mean()),
            float(fitness.std(ddof=1)),
            float(times.mean()),
        ))
    return rows


def print_summary(rows: list[tuple]) -> None:
    print(
        f"\n{'Function':12s} {'Dim':>4s} {'Initial mean':>14s} "
        f"{'Best':>14s} {'Mean':>14s} "
        f"{'Std':>14s} {'Avg time':>12s}"
    )
    print("-" * 91)
    for name, dim, initial, best, mean, std, avg_time in rows:
        print(
            f"{name:12s} {dim:4d} {initial:14.6e} "
            f"{best:14.6e} {mean:14.6e} "
            f"{std:14.6e} {avg_time:11.3f}s"
        )


def write_summary(rows: list[tuple], out: Path) -> None:
    write_csv(out/"summary.csv",
        ["function", "dimension", "initial_mean", "best", "mean", "std",
         "mean_runtime_s"],
        rows,
    )


# Raw data

def write_runs(results: Results, out: Path) -> None:
    write_csv(out/"runs.csv",
        ["function", "dimension", "seed", "initial_fx", "final_fx", "best_x",
         "evaluations", "runtime_s"],
        (
            (name, dim, r.seed, r.initial_fx, r.fx, json.dumps(r.x.tolist()),
             r.evaluations, r.runtime)
            for (name, dim), records in results.items()
            for r in records
        ),
    )


def write_best_so_far(results: Results, out: Path) -> None:
    # Only write the evaluations where the best-so-far value was updated.
    write_csv(out/"best_so_far.csv",
        ["function", "dimension", "seed", "evaluation", "best_fx"],
        (
            (name, dim, r.seed, i, float(fx))
            for (name, dim), records in results.items()
            for r in records
            for i, fx in enumerate(r.history, start=1)
            if i == 1 or fx < r.history[i - 2]
        ),
    )


# Plots

def plot_convergence(results: Results, name: str, dim: int, path: Path) -> None:
    records = results[(name, dim)]
    mean_history = np.stack([r.history for r in records]).mean(axis=0)
    evaluations = np.arange(1, len(mean_history) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(evaluations, mean_history, linewidth=2)
    ax.set_title(f"{name.capitalize()} (D={dim}): Mean convergence")
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Mean best-so-far objective value")
    ax.grid(alpha=0.3)

    fig.text(
        0.5, 0.01,
        f"Mean best-so-far objective over {len(records)} independent runs. "
        "Both axes are linear.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(results: Results, names: list[str], dim: int, path: Path) -> None:
    # Two panels per row, as many rows as needed.
    cols = min(len(names), 2)
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)

    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        fitness = [r.fx for r in results[(name, dim)]]
        ax.boxplot([fitness], showmeans=True)
        ax.set_xticks([1], [f"D={dim}"])
        ax.set_title(f"({chr(97 + i)}) {name.capitalize()}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Final objective value")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Final objective values at D={dim}")

    n_runs = len(results[(names[0], dim)])
    fig.text(
        0.5, 0.01,
        f"Distributions over {n_runs} independent runs per function. "
        "Triangles indicate means; horizontal lines indicate medians.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
