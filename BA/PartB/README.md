# Part B: Continuous Function Optimization with the Bees Algorithm

Minimizes Sphere, Rosenbrock, Rastrigin and Ackley at D = 10 and 30 with the
Bees Algorithm: 10 runs per case, 30,000 evaluations per run.

- **Dependencies:** [`pyproject.toml`](../../pyproject.toml), exact versions
  in [`uv.lock`](../../uv.lock). Run everything with
  [uv](https://docs.astral.sh/uv/).
- **BA settings:** [`bench.py`](bench.py); the settings used for the
  saved results are in [`results/config.json`](results/config.json).
- **Seeds:** generated from the master seed in `bench.py`; every run's seed
  is in [`results/runs.csv`](results/runs.csv).

## Configuration

All experiment settings for the results are in `CONFIG` in [`bench.py`](bench.py):
functions to run, dimensions, number of runs, evaluation budget, master seed,
output folder and the BA settings. Feel free to change them around and 
re-run the experiments :)

## Commands

Run from this folder (`BA/PartB`).

With uv (installs the locked versions automatically):

```
uv run check_functions.py   # check each function at its minimizer and one other point
uv run bench.py             # full experiment set (80 runs), writes to results/
```

Without uv (Python 3.13+, versions from `uv.lock`):

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install numpy==2.5.4 matplotlib==3.11.2

python check_functions.py
python bench.py
```

## Output ([`results/`](results/))

- `summary.csv`: per case, the summary table
- `runs.csv`: per run, the seed, initial best, final best, best vector, evaluations and runtime
- `best_so_far.csv`: one row per evaluation where the best-so-far improved
- `rastrigin_convergence.png`, `final_boxplots_d30.png`: the required plots
