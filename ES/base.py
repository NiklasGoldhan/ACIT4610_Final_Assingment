import numpy as np


def build_covariance(std_values, corr):
    """Build a covariance matrix from standard deviations and correlations."""
    return std_values[:, np.newaxis] * std_values[np.newaxis, :] * corr


def load_data(file_path):
    """
    Load portfolio statistics and construct the covariance matrix.
    Returns:
        Tuple of four numpy arrays: Return vector, standard-deviation vector, 
        symmetric correlation matrix, and covariance matrix.
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

    cov = build_covariance(std_values, corr)

    return mean_return, std_values, corr, cov


def target_calculations(mean_return):
    """
    Calculate the minimum required return for a dataset.
    """
    alphas = [0.25, 0.50, 0.75]

    equal_weights = np.full(mean_return.size, 1 / mean_return.size)
    Req = mean_return @ equal_weights
    Rmax = np.max(mean_return)

    return_targets = {
        alpha: (1 - alpha) * Req + alpha * Rmax
        for alpha in alphas
    }

    return Req, return_targets