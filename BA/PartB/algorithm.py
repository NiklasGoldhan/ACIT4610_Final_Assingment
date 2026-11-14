from collections.abc import Callable

import numpy as np

# MAIN ALGORITHM


def ba_function(
    f: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[float, float],
    ns: int,
    ne: int,
    nb: int,
    nre: int,
    nrb: int,
    ngh: float = 0.1,
    stlim: int = 30,
    shrink_factor: float = 0.95,
    max_evals: int = 30_000,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Minimizes a function f within the given bounds using the bees algorithm.
    Returns a (x, f(x)) value pair.

    f             ->  function to be minimized
    dim           ->  n dimensions
    bounds        ->  function bounds (min, max)
    ns            ->  n scout bees
    ne            ->  n elite sites
    nb            ->  n best sites
    nre           ->  n recruited bees for elite sites
    nrb           ->  n recruited bees for remaining best sites
    ngh           ->  initial neighborhood size (0.0 .. 1.0)
    stlim         ->  n stagnant cycles before abandoning a site
    shrink_factor ->  ngh shrinkage per iteration if stagnant (0.0 .. 1.0)
    seed          ->  custom seed for random
    """

    class MaxReached(Exception):
        """Raised when MAX_EVALS is reached."""

    rng = np.random.default_rng(seed)
    b_min, b_max = bounds

    ngh = ngh * (b_max - b_min)

    evals = 0
    best_x: np.ndarray | None = None
    best_fx: float = np.inf

    # Evaluate function, raising an exception when max evals is reached.
    def evaluate(x):
        nonlocal evals, best_x, best_fx
        if evals >= max_evals:
            raise MaxReached
        fx = f(x)
        evals += 1
        if fx < best_fx:
            best_fx, best_x = fx, np.array(x, copy=True)
        return fx

    # Initialization
    xs = rng.uniform(b_min, b_max, size=(ns, dim))
    fxs = np.array([evaluate(x) for x in xs])
    # Track neighborhood sizes and stagnant cycles.
    site_ngh = np.full(ns, ngh)
    st_count = np.zeros(ns, dtype=int)

    try:
        while True:
            # Ranking sites in ascending order
            ranked = np.argsort(fxs)

            # Local search
            for rank, i in enumerate(ranked[:nb]):
                n_bees = nre if rank < ne else nrb

                sites = rng.uniform(
                    xs[i] - site_ngh[i], xs[i] + site_ngh[i], size=(n_bees, dim)
                )
                np.clip(sites, b_min, b_max, out=sites)

                # Keep track of the local best value pair.
                local_best_x, local_best_fx = None, np.inf
                for x in sites:
                    fx = evaluate(x)
                    if fx < local_best_fx:
                        local_best_x, local_best_fx = x, fx

                if local_best_fx < fxs[i]:
                    xs[i] = local_best_x
                    fxs[i] = local_best_fx
                    st_count[i] = 0
                else:
                    site_ngh[i] *= shrink_factor
                    st_count[i] += 1

                    if st_count[i] >= stlim:
                        xs[i] = rng.uniform(b_min, b_max, size=dim)
                        fxs[i] = evaluate(xs[i])
                        site_ngh[i] = ngh
                        st_count[i] = 0

            # Global search
            for i in ranked[nb:]:
                xs[i] = rng.uniform(b_min, b_max, size=dim)
                fxs[i] = evaluate(xs[i])
                site_ngh[i] = ngh
                st_count[i] = 0

    # Exit the loop if MAX_EVALS is reached.
    except MaxReached:
        pass

    # Assert so type check stops complaining.
    assert best_x is not None
    return (best_x, best_fx)
