import time

from ES.base import ESBase, BUDGET, SIGMA_MIN, SIGMA_MAX
import numpy as np

class OnePlusOneES(ESBase):
    
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
        
        while self.evaluations < max_evaluations:
            
            child = self.mutate(parent, sigma)
            child_variance = self.evaluate(child)

            success = child_variance < parent_variance

            if success:
                parent, parent_variance = child, child_variance

            sigma = self.update_sigma(sigma, success)

        return self.make_result(start_time)

