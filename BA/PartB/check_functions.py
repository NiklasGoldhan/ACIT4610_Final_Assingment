"""
Checks each benchmark function at its known minimizer and at one other point
against values derived by hand from the Simon Fraser University (SFU)
definitions (Surjanovic & Bingham).

Reference values for the second point (D = number of coordinates):
  Sphere     f(1)   = sum(1^2)                                   = D
  Rosenbrock f(0)   = sum_{i=1}^{D-1} [100(0 - 0)^2 + (1 - 0)^2] = D - 1
  Rastrigin  f(0.5) = 10D + sum[0.25 - 10cos(pi)] = 10D + 10.25D = 20.25D
  Ackley     f(1)   = -20exp(-0.2) - exp(cos(2pi)) + 20 + e
                    = 20(1 - exp(-0.2))                          ~ 3.62538
"""

# THIS FILE IS AI-GENERATED AND CHECKED.

import math

import numpy as np

from functions import ackley, rastrigin, rosenbrock, sphere

TOL = 1e-12


def cases(d: int):
    zeros, ones = np.zeros(d), np.ones(d)
    return [
        # (name, function, point description, x, expected f(x))
        ("sphere", sphere, "minimizer 0", zeros, 0.0),
        ("sphere", sphere, "point 1", ones, float(d)),
        ("rosenbrock", rosenbrock, "minimizer 1", ones, 0.0),
        ("rosenbrock", rosenbrock, "point 0", zeros, float(d - 1)),
        ("rastrigin", rastrigin, "minimizer 0", zeros, 0.0),
        ("rastrigin", rastrigin, "point 0.5", np.full(d, 0.5), 20.25 * d),
        ("ackley", ackley, "minimizer 0", zeros, 0.0),
        ("ackley", ackley, "point 1", ones, 20 * (1 - math.exp(-0.2))),
    ]


def main() -> None:
    failures = 0
    for d in (10, 30):
        for name, f, label, x, expected in cases(d):
            got = float(f(x))
            ok = math.isclose(got, expected, rel_tol=TOL, abs_tol=TOL)
            failures += not ok
            print(
                f"{'PASS' if ok else 'FAIL'}  {name:10s} D={d:2d}  {label:12s}"
                f"  expected={expected:.12g}  got={got:.12g}"
            )
    if failures:
        raise SystemExit(f"\n{failures} check(s) failed")
    print("\nAll function checks passed.")


if __name__ == "__main__":
    main()
