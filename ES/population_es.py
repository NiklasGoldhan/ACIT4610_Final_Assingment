import time
import numpy as np

from ES.base import ESBase, BUDGET, SIGMA_MIN, SIGMA_MAX

"""
(mu+lambda)-ES and (mu,lambda)-ES for the portfolio problem.
AI-assisted: the first version was generated with AI and later restructured
so both variants share one loop. Mark the exact share in the report.
"""

MU = 10
LAMBDA = 70


class PopulationES(ESBase):
    # True: parents and children compete for survival (mu+lambda).
    # False: only the children can survive (mu,lambda).
    plus = True

    def __init__(self, problem, Rmin, seed=None, mu=MU, lam=LAMBDA, budget=BUDGET):
        super().__init__(problem, Rmin, seed=seed)
        # Comma selection picks mu parents from the children only, so it needs lambda >= mu.
        if not self.plus and lam < mu:
            raise ValueError("(mu,lambda)-ES needs lambda >= mu")
        self.mu = mu
        self.lam = lam
        self.budget = budget


    def make_child(self, parent_weights, parent_sigmas, tau_prime, tau):
        """
        Create one child from two random parents.
        Recombine, self-adapt the step sizes, mutate and repair.
        """
        i, j = self.rng.choice(self.mu, size=2, replace=False)
        weights = (parent_weights[i] + parent_weights[j]) / 2
        sigma = np.sqrt(parent_sigmas[i] * parent_sigmas[j])

        # Z0 is shared by all assets, Zi is drawn for each asset.
        z0 = self.rng.normal()
        zi = self.rng.normal(size=self.n)
        sigma = sigma * np.exp(tau_prime * z0 + tau * zi)
        sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX)

        # New normal draw for the mutation itself, then repair to feasible weights.
        weights = self.repair(weights + sigma * self.rng.normal(size=self.n))
        return weights, sigma


    def run(self):
        """
        Run one search and return the best feasible allocation found.
        """
        start_time = time.perf_counter()
        self.reset_archive()
        n = self.n
        tau_prime = 1 / np.sqrt(2 * n)
        tau = 1 / np.sqrt(2 * np.sqrt(n))

        # Initial parents, these also count towards the budget.
        weights = np.array([self.initialize_weights() for _ in range(self.mu)])
        sigmas = np.full((self.mu, n), 0.1 / np.sqrt(n))
        variances = np.array([self.evaluate(w) for w in weights])

        while self.evaluations < self.budget:
            # The last batch can be smaller than lambda so we stop exactly at the budget.
            batch = min(self.lam, self.budget - self.evaluations)
            child_weights = np.empty((batch, n))
            child_sigmas = np.empty((batch, n))
            child_variances = np.empty(batch)

            for k in range(batch):
                child_weights[k], child_sigmas[k] = self.make_child(
                    weights, sigmas, tau_prime, tau
                )
                child_variances[k] = self.evaluate(child_weights[k])

            # Survivor selection, the only difference between the two variants.
            if self.plus:
                child_weights = np.vstack([weights, child_weights])
                child_sigmas = np.vstack([sigmas, child_sigmas])
                child_variances = np.concatenate([variances, child_variances])

            # Lowest variance is best, so keep the first mu after sorting.
            survivors = np.argsort(child_variances)[:self.mu]
            weights = child_weights[survivors]
            sigmas = child_sigmas[survivors]
            variances = child_variances[survivors]

        return self.make_result(start_time)


class MuPlusLambdaES(PopulationES):
    plus = True


class MuCommaLambdaES(PopulationES):
    plus = False
