import time

from ES.base import ESBase, BUDGET, SIGMA_MIN, SIGMA_MAX
import numpy as np

class OnePlusOneES(ESBase):
    def __init__(self, problem, Rmin, seed=None):
        super().__init__(problem, Rmin, seed)


    def mutate(self, parent, sigma):
        """Mutate the parent solution to create a child solution."""
        noise = self.rng.normal(0, 1, self.n)
        child = parent + sigma * noise
        return self.repair(child)


    def update_sigma(self, sigma, success):
        """Update the mutation step size."""
        if success:
            sigma *= np.exp(1 / 3)
        else:
            sigma *= np.exp(-1 / 12)

        return np.clip(sigma, SIGMA_MIN, SIGMA_MAX)


    def run(self, max_evaluations=BUDGET):
        start_time = time.perf_counter()
        self.reset_archive()
        parent = self.initialize_weights()
        parent_variance = self.evaluate(parent)

        sigma = 0.1 / np.sqrt(self.n)
        evaluations = 1

        archive = []
        
        while evaluations < max_evaluations:
            archive.append((parent, parent_variance))
            
            child = self.mutate(parent, sigma)
            child_variance = self.variance(child)

            evaluations += 1

            success = child_variance < parent_variance

            if success:
                parent = child
                archive.append((parent, parent_variance))
                parent_variance = child_variance

            sigma = self.update_sigma(sigma, success)

        return self.make_result(start_time)

