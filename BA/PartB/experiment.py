from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np

from algorithm import ba_function
from functions import BENCHMARKS, BenchmarkTarget, Objective

# Scaffolding for the experiments, since I wanted to keep the algorithm itself pure.

@dataclass(frozen=True)
class BAParams:
    ns: int
    ne: int
    nb: int
    nre: int
    nrb: int
    ngh: float = 0.1
    stlim: int = 30
    shrink_factor: float = 0.95


@dataclass(frozen=True)
class Config:
    master_seed: int
    runs: int
    functions: tuple[str, ...]  # names from functions.BENCHMARKS
    dimensions: tuple[int, ...]
    max_evals: int
    output_dir: Path
    ba: BAParams
    # Save to json
    def to_json(self) -> dict:
        return {
            "master_seed": self.master_seed,
            "runs": self.runs,
            "functions": list(self.functions),
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
    initial_fx: float  # best value in the initial scout population
    fx: float
    runtime: float
    evaluations: int
    history: np.ndarray  # best-so-far fx after every evaluation


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
        # The first ns evaluations are the random initial population.
        initial_fx=float(tracker.history[cfg.ba.ns - 1]),
        fx=float(fx),
        runtime=elapsed,
        evaluations=tracker.evaluations,
        history=np.asarray(tracker.history),
    )


def run_all(cfg: Config) -> Results:
    by_name = {b.name: b for b in BENCHMARKS}
    benchmarks = [by_name[name] for name in cfg.functions]

    # The master seed drives a generator that gives each run its own seed.
    rng = np.random.default_rng(cfg.master_seed)

    results: Results = {}
    for bench, dim, run in product(benchmarks, cfg.dimensions, range(cfg.runs)):
        seed = int(rng.integers(0, 2**32))
        r = run_one(bench, dim, seed, cfg)
        results.setdefault((bench.name, dim), []).append(r)

        print(
            f"{bench.name:12s} | D={dim:2d} | "
            f"run={run + 1:2d}/{cfg.runs} | seed={r.seed:10d} | "
            f"fx={r.fx:.6e} | evals={r.evaluations} | "
            f"time={r.runtime:.3f}s"
        )

    return results
