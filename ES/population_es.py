import time
import numpy as np

from ES.base import ESBase, BUDGET, SIGMA_MIN, SIGMA_MAX

MU = 10
LAMBDA = 70


class PopulationES(ESBase):
    """Shared code for (mu+lambda)-ES and (mu,lambda)-ES."""

    # True: mu+lambda.
    # False: mu,lambda.
    plus = True

    def __init__(self, problem, Rmin, seed=None, mu=MU, lam=LAMBDA):
        super().__init__(problem, Rmin, seed)
        self.mu = mu
        self.lam = lam
        # Learning rates for self-adaptation, as given in the assignment.
        self.tau_prime = 1 / np.sqrt(2 * self.n)
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.n))

    def recombine(self, parents, parent_sigmas):
        """Pick two different parents at random and combine them."""
        i, j = self.rng.choice(self.mu, size=2, replace=False)
        weights = (parents[i] + parents[j]) / 2
        sigma = np.sqrt(parent_sigmas[i] * parent_sigmas[j])
        return weights, sigma

    def update_sigma(self, sigma):
        """
        Self-adapt the step sizes.
        Z0 is shared by all assets, Zi is drawn for each asset.
        """
        z0 = self.rng.normal()
        zi = self.rng.normal(0, 1, self.n)
        sigma = sigma * np.exp(self.tau_prime * z0 + self.tau * zi)
        return np.clip(sigma, SIGMA_MIN, SIGMA_MAX)

    def mutate(self, weights, sigma):
        """Mutate the weights with Gaussian noise, then repair them."""
        noise = self.rng.normal(0, 1, self.n)
        child = weights + sigma * noise
        return self.repair(child)

    def run(self, max_evaluations=BUDGET):
        """Run one search and return the best feasible allocation found."""
        start_time = time.perf_counter()
        self.reset_archive()

        # Initial parents. These also count towards the budget.
        parents = []
        parent_sigmas = []
        parent_variances = []
        for _ in range(self.mu):
            weights = self.initialize_weights()
            parents.append(weights)
            parent_sigmas.append(np.full(self.n, 0.1 / np.sqrt(self.n)))
            parent_variances.append(self.evaluate(weights))

        while self.evaluations < max_evaluations:
            # The last generation can be smaller than lambda so we stop exactly at the budget.
            num_children = min(self.lam, max_evaluations - self.evaluations)

            children = []
            child_sigmas = []
            child_variances = []
            for _ in range(num_children):
                weights, sigma = self.recombine(parents, parent_sigmas)
                sigma = self.update_sigma(sigma)        # step sizes first
                weights = self.mutate(weights, sigma)   # then the weights
                children.append(weights)
                child_sigmas.append(sigma)
                child_variances.append(self.evaluate(weights))

            # Survivor selection: keep the mu individuals with the lowest variance.
            if self.plus:
                pool = parents + children
                pool_sigmas = parent_sigmas + child_sigmas
                pool_variances = parent_variances + child_variances
            else:
                pool = children
                pool_sigmas = child_sigmas
                pool_variances = child_variances

            best = np.argsort(pool_variances)[:self.mu]
            parents = [pool[i] for i in best]
            parent_sigmas = [pool_sigmas[i] for i in best]
            parent_variances = [pool_variances[i] for i in best]

        return self.make_result(start_time)


class MuPlusLambdaES(PopulationES):
    plus = True


class MuCommaLambdaES(PopulationES):
    plus = False
