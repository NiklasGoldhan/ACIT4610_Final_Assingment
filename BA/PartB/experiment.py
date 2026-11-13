from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np

from algorithm import ba_function
from functions import BENCHMARKS, BenchmarkTarget, Objective


@dataclass(frozen=True)
class BAParams:
    ns: int
    ne: int
    nb: int
    nre: int
    nrb: int


@dataclass(frozen=True)
class Config:
    master_seed: int
    runs: int
    dimensions: tuple[int, ...]
    max_evals: int
    output_dir: Path
    ba: BAParams
    # Save to json
    def to_json(self) -> dict:
        return {
            "master_seed": self.master_seed,
            "runs": self.runs,
            "dimensions": list(self.dimensions),
            "max_evaluations": self.max_evals,
            **asdict(self.ba),
        }


class EvaluationTracker:
    def __init__(self, function: Objective):
        self.function = function
        self.evaluations = 0
        self.best_fx = float("inf")
        self.history: list[float] = []

    def __call__(self, x: np.ndarray) -> float:
        fx = float(self.function(x))
        self.evaluations += 1
        self.best_fx = min(self.best_fx, fx)
        self.history.append(self.best_fx)
        return fx


class RunResult(NamedTuple):
    seed: int
    x: np.ndarray
    fx: float
    runtime: float
    evaluations: int
    history: np.ndarray


Results = dict[tuple[str, int], list[RunResult]]

# Primarily AI-generated from an earlier implementation. VVVV

def run_one(bench: BenchmarkTarget, dim: int, seed: int, cfg: Config) -> RunResult:
    tracker = EvaluationTracker(bench.function)

    start = perf_counter()
    x, fx = ba_function(
        tracker,
        dim,
        bench.bounds,
        **asdict(cfg.ba),
        max_evals=cfg.max_evals,
        seed=seed,
    )
    elapsed = perf_counter() - start

    if tracker.evaluations != cfg.max_evals:
        raise RuntimeError(
            f"Expected {cfg.max_evals} evaluations, got {tracker.evaluations}"
        )
    if not np.isclose(fx, tracker.best_fx, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Tracked and returned best values differ")

    return RunResult(
        seed=seed,
        x=np.array(x, copy=True),
        fx=float(fx),
        runtime=elapsed,
        evaluations=tracker.evaluations,
        history=np.asarray(tracker.history),
    )


def run_all(cfg: Config, benchmarks: list[BenchmarkTarget] = BENCHMARKS) -> Results:
    # Reproducible, independent run seeds (same stream and order as before).
    n_runs = len(benchmarks) * len(cfg.dimensions) * cfg.runs
    seeds = np.random.default_rng(cfg.master_seed).integers(0, 2**32, size=n_runs)
    experiments = product(benchmarks, cfg.dimensions, range(cfg.runs))

    results: Results = {}
    for (bench, dim, run), seed in zip(experiments, seeds, strict=True):
        r = run_one(bench, dim, int(seed), cfg)
        results.setdefault((bench.name, dim), []).append(r)

        print(
            f"{bench.name:12s} | D={dim:2d} | "
            f"run={run + 1:2d}/{cfg.runs} | seed={r.seed:10d} | "
            f"fx={r.fx:.6e} | evals={r.evaluations} | "
            f"time={r.runtime:.3f}s"
        )

    return results
