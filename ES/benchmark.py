from ES.base import PortfolioData
from pathlib import Path
import numpy as np


# Move to benchmark.py later, just used for validation of port1.txt calculations
# as stated in the assignment.
def parser_validation_port1(mean_return, cov, return_targets):
        """
        Validate parser and calculations for port1.txt.
        """
        equal_weights = np.full(mean_return.size, 1 / mean_return.size)

        Req = mean_return @ equal_weights
        equal_variance = equal_weights @ cov @ equal_weights
        Rmin_050 = return_targets[0.50]

        expected = {
            "equal return": (Req, 0.003504064516129),
            "equal variance": (equal_variance, 0.001130937943724),
            "alpha 0.50 target": (Rmin_050, 0.007184532258065),
        }
        for name, (actual, expected_value) in expected.items():
            if not np.isclose(actual, expected_value):
                raise ValueError(
                    f"{name} validation failed: "
                    f"expected {expected_value}, got {actual}"
                )

        print("port1 validation passed")


def generate_test_cases(self):
        datasets = [
            "port1.txt",
            "port2.txt",
            "port3.txt",
            "port4.txt",
            "port5.txt"
        ]

        test_cases = []
        data_directory = Path(__file__).resolve().parent / "data"

        for data in datasets:
            file_path = data_directory / data
            problem = PortfolioData.load_data(file_path)
            self.mean_return = problem.mean_return
            self.cov = problem.cov
            _, return_targets = self.target_calculations(self.mean_return)
            
            for alpha, rmin in return_targets.items():
                test_cases.append({
                    "case_name": f"{Path(data).stem}_alpha{int(alpha * 100):03d}",
                    "dataset": data,
                    "file_path": file_path,
                    "alpha": alpha,
                    "Rmin": rmin,
                    "mean_return": self.mean_return,
                    "cov": self.cov,
                })

        return test_cases