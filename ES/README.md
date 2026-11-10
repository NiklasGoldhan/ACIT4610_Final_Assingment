# Problem 2: Portfolio Optimization with Evolution Strategies

Problem 2 uses three Evolution Strategy variants (1+1-ES, μ+λ-ES, and μ,λ-ES) in order to allocate a budget
across *n* assets. Each one minimizes portfolio variance subject to a minimum expected return:

```
minimize    V(w) = wᵀ Σ w
subject to  w_i ≥ 0,   Σ w_i = 1,   rᵀ w ≥ R_min
```

The covariance matrix is Σ_ij = s_i · s_j · C_ij. It is built from the standard deviations *s*
and the correlation matrix *C* in each dataset.

## Test cases

There are 5 OR-Library datasets (port1-5), with 31, 85, 89, 98 and 225 assets and 3 return targets each:

```
R_min(α) = (1 − α) · R_eq + α · r_max,    α ∈ {0.25, 0.50, 0.75}
```

The target lies between the equal-weight return (R_eq) and the best single-asset return (r_max), giving 15 test cases (port1_alpha025 to port5_alpha075).

## Requirements

- Python ≥ 3.13
- numpy 2.5.4, matplotlib 3.11.2, scipy 1.18.1 (exact versions are pinned in `uv.lock`)
- [uv](https://docs.astral.sh/uv/)

Install from the repository root:

```bash
uv sync
```

## Running

Run these commands from the repository root, not from inside ES/.

```bash
# One test case
uv run python -m ES.benchmark --case port1_alpha050

# Full experiment: 15 cases × 3 variants × 10 seeds (ish 6–7 minutes)
uv run python -m ES.benchmark
```

## Files

- `base.py`: Data loading, covariance matrix, repair, feasibility check, best-so-far archive
- `one_plus_one.py`: (1+1)-ES
- `population_es.py`: (μ+λ)-ES and (μ,λ)-ES
- `benchmark.py`: Runs all experiments; writes tables, plots and statistical tests
- `data/port1–5.txt`: Asset statistics and correlations
- `data/portef1–5.txt`: OR-Library efficient frontiers, used only to compute the gap to the optimum

Data source: OR-Library, <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html>

## Settings

Each run has a budget of 30,000 objective evaluations. Every evaluated candidate counts, including the initial population and candidates that are rejected. Each variant is run 10 times on every test case, with seeds 0 to 9.

The (1+1)-ES starts with a step size of 0.1/sqrt(n) and adapts it with the 1/5th success rule. The step size is multiplied by exp(1/3) when the child is better than the parent and by exp(-1/12) otherwise.

The population variants use 10 parents and 70 offspring. Each child is made by recombining two random parents. It then mutates with its own self-adapted step size for each asset, which starts at 0.1/sqrt(n). The step sizes are kept between 1e-8 and 1 in all variants.

## Output

The `results/` and `plots/` folders are created when the benchmark runs.

- `results/raw_runs.csv`: One row per run: seed, variance, return, evaluations, runtime, gap, weights
- `results/summary.csv`: Per case and variant: best, mean and std of variance, valid runs, mean runtime, gap
- `results/stat_tests.csv`: Paired Wilcoxon signed-rank tests between the variants
- `results/convergence_port5_alpha050.npz`: Best-so-far log for every run
- `plots/convergence_port5_alpha050.png`: Mean best-so-far variance against evaluations

## Use of AI

AI tools (Cursor, Claude Code) were used during development of the ES folder.

- `benchmark.py`: largely AI-generated, including the experiment loop, CSV output, the Wilcoxon signed-rank test, plots and the OR-Library gap calculation. It was reviewed and adjusted by the group.
- `base.py`: written together with AI. AI helped implement the formulas from the assignment in numpy (covariance, simplex projection, return repair).
- `population_es.py`: written by the group. AI was used for debugging, review and small fixes (comment and formatting cleanup).
- `one_plus_one.py`: written by the group. AI was used to find the evaluation-counting bug.
- `README.md`: drafted with AI and edited by the group.
