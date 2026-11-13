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
        fitness = np.array([r.fx for r in records])
        times = np.array([r.runtime for r in records])
        rows.append((
            name,
            dim,
            float(fitness.min()),
            float(fitness.mean()),
            float(fitness.std(ddof=1)),
            float(times.mean()),
        ))
    return rows


def print_summary(rows: list[tuple]) -> None:
    print(
        f"\n{'Function':12s} {'Dim':>4s} "
        f"{'Best':>14s} {'Mean':>14s} "
        f"{'Std':>14s} {'Avg time':>12s}"
    )
    print("-" * 76)
    for name, dim, best, mean, std, avg_time in rows:
        print(
            f"{name:12s} {dim:4d} "
            f"{best:14.6e} {mean:14.6e} "
            f"{std:14.6e} {avg_time:11.3f}s"
        )


def write_summary(rows: list[tuple], out: Path) -> None:
    write_csv(out/"summary.csv",
        ["function", "dimension", "best", "mean", "std", "mean_runtime_s"],
        rows,
    )


# Raw data

def write_runs(results: Results, out: Path) -> None:
    write_csv(out/"runs.csv",
        ["function", "dimension", "seed", "final_fx", "best_x",
         "evaluations", "runtime_s"],
        (
            (name, dim, r.seed, r.fx, json.dumps(r.x.tolist()),
             r.evaluations, r.runtime)
            for (name, dim), records in results.items()
            for r in records
        ),
    )


def write_best_so_far(results: Results, out: Path) -> None:
    write_csv(out/"best_so_far.csv",
        ["function", "dimension", "seed", "evaluation", "best_fx"],
        (
            (name, dim, r.seed, i, float(fx))
            for (name, dim), records in results.items()
            for r in records
            for i, fx in enumerate(r.history, start=1)
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(results: Results, names: list[str], dim: int, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
