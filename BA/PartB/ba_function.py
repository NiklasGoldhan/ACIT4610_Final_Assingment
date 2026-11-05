# IMPORTS
import numpy as np

# CONSTANTS
MAX_OBJECTIVE_EVALUATIONS = 30_000

# we need to count evaluations including:
# initial population
# recruited neighbors
# fresh scouts, including rejected candidates

# MAIN ALGORITHM

def ba_function(f, dim, bounds, ns, ne, nb, nre, nrb, ngh, stlim, seed=None):
    """
    Minimizes a function f within the given bounds using the bees algorithm.

    f       ->  function to be minimized
    dim     ->  n dimensions
    bounds  ->  tuple with a form of (min, max)
    ns      ->  n scout bees
    ne      ->  n elite sites
    nb      ->  n best sites
    nre     ->  n recruited bees for elite sites
    nrb     ->  n recruited bees for remaining best sites
    ngh     ->  initial neighborhood size
    stlim   ->  n stagnant cycles before abandoning a site
    """
    rng = np.random.default_rng(seed)
    b_min, b_max = bounds
    evals = 0

    # initialize the scouts uniformally
    scouts = rng.uniform(b_min, b_max, size=(ns, dim))

    fitnesses = f(scouts)
    evals += ns


# FUNCTION DEFINITIONS

# will pass bounds directly to benchmark later ...

# (-5.12, 5.12)
def sphere(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2, axis=-1)

# (-5, 10)
def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    xi = x[..., :-1]
    xnext = x[..., 1:]
    return np.sum(100 * (xnext - xi**2)**2 + (xi - 1)**2, axis=-1)

# (-5.12, 5.12)
def rastrigin(x):
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    return 10*dim + np.sum(x**2 - 10*np.cos(2 * np.pi * x), axis = -1)

# (-32.768, 32.768)
def ackley(x):
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    t1 = -0.2 * np.sqrt(np.sum(x**2, axis=-1) / dim)
    t2 = np.sum(np.cos((2*np.pi)*x), axis=-1) / dim
    return -20 * np.exp(t1) - np.exp(t2) + 20 + np.e