import time
import numpy as np
from dataclasses import dataclass

SEED = 42
BUDGET = 30_000
SIGMA_MIN = 1e-8
SIGMA_MAX = 1.0

"""
Blant annet brukt AI for riktig implementasjon av funksjoner fra oppgaveteksten med
numpy, til generell feilsøking og for å finne/fjerne bugs og redundant kode.
This is just a note to myself, I will mark and state which parts later.
"""

@dataclass(slots=True)
class PortfolioData:
    mean_return: np.ndarray
    std_values: np.ndarray
    corr: np.ndarray
    cov: np.ndarray

    @staticmethod
    def build_covariance_matrix(std_values, corr):
        """Build a covariance matrix from standard deviations and correlations."""
        return std_values[:, np.newaxis] * std_values[np.newaxis, :] * corr
    
    @classmethod
    def load_data(cls, file_path):
        """
        Load portfolio statistics and construct the covariance matrix.
        Returns:
            PortfolioData: Loaded asset statistics and covariance matrix.
        """
        with open(file_path, 'r') as file:
            lines = [line.split() for line in file if line.strip()]

        num_assets = int(lines[0][0])
        asset_lines = lines[1:num_assets + 1]
        
        mean_return = np.array(
            [float(line[0]) for line in asset_lines],
            dtype=float,
        )
        
        std_values = np.array(
            [float(line[1]) for line in asset_lines],
            dtype=float,
        )

        corr = np.zeros((num_assets, num_assets), dtype=float)
        for line in lines[num_assets + 1:]:
            i, j = int(line[0]) - 1, int(line[1]) - 1
            correlation = float(line[2])
            corr[i, j] = correlation
            corr[j, i] = correlation

        cov = cls.build_covariance_matrix(std_values, corr)

        return cls(mean_return, std_values, corr, cov)


@dataclass(slots=True)
class ESRunResult:
    weights: np.ndarray
    variance: float
    expected_return: float
    num_evaluations: int
    history: np.ndarray
    runtime: float


class ESBase:
    def __init__(self, problem, Rmin, seed=None):
        self.mean_return = problem.mean_return
        self.cov = problem.cov
        self.Rmin = Rmin
        self.n = problem.mean_return.size
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(SEED)

    
    def variance(self, weights):
        """Calculate the portfolio variance given weights."""
        return weights @ self.cov @ weights
    
    @staticmethod
    def target_calculations(mean_return):
        """Calculate the minimum required return for a dataset."""
        equal_weights = np.full(mean_return.size, 1 / mean_return.size)
        Req = mean_return @ equal_weights
        Rmax = np.max(mean_return)

        return_targets = {
            alpha: (1 - alpha) * Req + alpha * Rmax
            for alpha in (0.25, 0.50, 0.75)
        }
        return Req, return_targets

    
    def expected_return(self, weights):
        """Calculate the expected return for the given weights."""
        return self.mean_return @ weights

    
    def is_feasible(self, weights):
        """
        Check if the given weights are feasible.
        Is feasible if:
        1. All weights are non-negative.
        2. The sum of weights is equal to 1.
        3. The expected return is greater than or equal to the minimum required return.
        Using decimal precision of 1e-8 for floating-point comparisons.
        """
        return (
            np.all(weights >= -1e-8)
            and abs(np.sum(weights) - 1) <= 1e-8
            and self.expected_return(weights) >= self.Rmin - 1e-8
        )

    
    def repair(self, weights):
        """Project weights onto the simplex and repair the return constraint."""
        # Convert weights to a numpy array and ensure they have the correct shape.
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n,):
            raise ValueError(f"weights must have shape ({self.n},)")

        # Simplex projection and repair for negative weights and sum constraint.
        sorted_weights = np.sort(weights)[::-1] # Sort weights in descending order
        cumulative = np.cumsum(sorted_weights)
        indices = np.arange(1, self.n + 1)
        valid = sorted_weights - (cumulative - 1) / indices > 0
        if not np.any(valid):
            projected = np.full(self.n, 1 / self.n)
        else:
            rho = indices[valid][-1]
            theta = (cumulative[rho - 1] - 1) / rho
            projected = np.maximum(weights - theta, 0)

        # Repair if the expected return is below Rmin.
        current_return = self.expected_return(projected)

        if current_return < self.Rmin:
            maximum_return_index = np.argmax(self.mean_return)
            maximum_return = self.mean_return[maximum_return_index]
            if maximum_return < self.Rmin:
                raise ValueError("Rmin is infeasible for the available assets")

            beta = (self.Rmin - current_return) / (maximum_return - current_return)
            best_asset = np.zeros(self.n)
            best_asset[maximum_return_index] = 1
            projected = (1 - beta) * projected + beta * best_asset

        if not self.is_feasible(projected):
            raise ValueError(
                "Repair failed to produce a feasible solution. "
                "Check the Rmin value and the asset returns."
            )
        return projected

    
    def initialize_weights(self):
        """Initialize weights randomly and repair them to ensure feasibility."""
        weights = self.rng.uniform(0, 1, self.n)
        return self.repair(weights)


    def reset_archive(self):
        """Start a new run with an empty archive and no evaluations used."""
        self.best_weights = None
        self.best_variance = np.inf
        self.num_evaluations = 0
        self.history = []


    def evaluate(self, weights):
        """
        Calculate the variance of the weights and count it as one evaluation.
        The archive keeps the best weights found so far, and history stores
        the best variance after every evaluation (used for convergence plots).
        """
        variance = self.variance(weights)
        self.num_evaluations += 1
        if variance < self.best_variance:
            self.best_variance = variance
            self.best_weights = weights.copy()
        self.history.append(self.best_variance)
        return variance


    def make_result(self, start_time):
        """Collect the archive into a result for the benchmark."""
        return ESRunResult(
            weights=self.best_weights,
            variance=float(self.best_variance),
            expected_return=float(self.expected_return(self.best_weights)),
            num_evaluations=self.num_evaluations,
            history=np.array(self.history),
            runtime=time.perf_counter() - start_time,
        )

    def mutate(self, parent, sigma):
        """Mutate the parent solution to create a child solution."""
        noise = self.rng.normal(0, 1, self.n)
        child = parent + sigma * noise
        return self.repair(child)
