from typing import NamedTuple, Callable, Tuple, List
from itertools import product
from algorithm import ba_function
import numpy as np


class OptimizationTask(NamedTuple):
    function: Callable[[np.ndarray], np.float64]
    bounds:   Tuple[float, float]

def sphere(x: np.ndarray) -> np.float64:
    return np.sum(x**2)

def rosenbrock(x: np.ndarray) -> np.float64:
    xi = x[..., :-1]
    xnext = x[..., 1:]
    return np.sum(100 * (xnext - xi**2)**2 + (xi - 1)**2)

def rastrigin(x: np.ndarray) -> np.float64:
    dim = x.shape[-1]
    return 10*dim + np.sum(x**2 - 10*np.cos(2 * np.pi * x))

def ackley(x: np.ndarray) -> np.float64:
    dim = x.shape[-1]
    t1 = -0.2 * np.sqrt(np.sum(x**2, axis=-1) / dim)
    t2 = np.sum(np.cos((2*np.pi)*x), axis=-1) / dim
    return -20 * np.exp(t1) - np.exp(t2) + 20 + np.e

tasks = [
    OptimizationTask(sphere, (-5.12, 5.12)),
    OptimizationTask(sphere, (-5, 10)),
    OptimizationTask(rastrigin, (-5.12, 5.12)),
    OptimizationTask(ackley, (-32.768, 32.768)),
]

dimensions = [10, 30]

def bench_ba(tasks, dims, runs, ns, ne, nb, nre, nrb):
    # ai helped flatten to this vvv
    for task, dim, run in product(tasks, dims, range(runs)):
        """TODO"""
