import numpy as np

# MAIN ALGORITHM

class MaxReached(Exception):
    """Raised when MAX_EVALS is reached."""

def ba_function(f, dim, bounds, ns, ne, nb, nre, nrb, ngh=0.1, stlim=10, shrink_factor=0.8, seed=None):
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
    MAX_EVALS = 30_000

    rng = np.random.default_rng(seed)
    b_min, b_max = bounds

    ngh = ngh * (b_max - b_min)

    evals = 0
    best_x, best_fx = None, np.inf

    def evaluate(x):
        nonlocal evals, best_x, best_fx
        if evals >= MAX_EVALS:
            raise MaxReached
        fx = float(f(x))
        evals += 1
        if fx < best_fx:
            best_fx, best_x = fx, np.array(x, copy=True)
        return fx

    # Initialization 
    xs = rng.uniform(b_min, b_max, size=(ns, dim))
    fxs = np.array([evaluate(x) for x in xs])
    site_ngh = np.full(ns, float(ngh))
    st_count = np.zeros(ns, dtype=int)

    try:
        while True:
            # Ranking
            ranked = np.argsort(fxs)

            # Local search
            for rank, i in enumerate(ranked[:nb]):
                n_bees = nre if rank < ne else nrb

                foragers = rng.uniform(xs[i] - site_ngh[i],
                                       xs[i] + site_ngh[i],
                                       size=(n_bees, dim))
                np.clip(foragers, b_min, b_max, out=foragers)

                local_best_x, local_best_fx = None, np.inf
                for x in foragers:
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

    return (best_x, best_fx)
