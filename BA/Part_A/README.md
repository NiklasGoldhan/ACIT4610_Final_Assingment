# Discrete Route Planning: Travelling Salesman Problem (TSP) with Bees Algorithm (Part A)

## Setup

For uv:
```bash
uv sync
```

For pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r BA/Part_A/requirements.txt
```

## Run

Run the full benchmark suite across all 6 TSPLIB instances, both neighborhood operators, and 10 random seeds (120 runs total), followed by automated table and plot generation:

```bash
uv run BA/Part_A/experiments.py
```
OR
```bash
python BA/Part_A/experiments.py
```

### Parameters (`experiments.py`)
Configured in `ExperimentConfig` in `BA/Part_A/experiments.py`:
- `max_evals`: Maximum objective function evaluations per run budget (default: `30_000`, strictly terminates at budget)
- `population_size`: Total scout bees (n)
- `selected_sites`: Number of best sites selected (m)
- `elite_sites`: Number of elite sites among selected (e)
- `recruited_elite`: Forager bees recruited for each elite site (ne)
- `recruited_selected`: Forager bees recruited for remaining non-elite sites (ns)
- `seeds`: 10 independent random seeds (default: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`)
- `instances`: Benchmark TSPLIB instances from `BA/Part_A/data/` (`eil51`, `berlin52`, `st70`, `eil76`, `kroA100`, `ch150`)
- `operators`: Local neighborhood search operators evaluated (`swap`, `reversal`)

### Outputs
- `BA/Part_A/results/results_summary.csv`: Summary metrics for each of the 120 trials (instance, operator, seed, best_length, optimal_length, gap_percent, runtime_seconds, evaluations)
- `BA/Part_A/results/table_summary.csv`: Aggregated summary statistics across 10 seeds (instance, cities, operator, optimum, best, mean, std, mean_gap, mean_runtime)
- `BA/Part_A/results/results_detailed.json`: Complete trial records containing evaluation-by-evaluation best-so-far history and best tours (omitted from git tracking due to file size)
- `BA/Part_A/results/convergence_berlin52.png`: Convergence comparison curve for `berlin52` (Evaluations vs. Mean Best-so-Far Tour Length across 10 runs for Swap vs. Reversal with optimal line)
- `BA/Part_A/results/convergence_ch150.png`: Convergence comparison curve for `ch150` (Evaluations vs. Mean Best-so-Far Tour Length across 10 runs for Swap vs. Reversal with optimal line)
- `BA/Part_A/results/best_route_berlin52.png`: 2D closed tour plot of the shortest valid route found for `berlin52` across all runs and operators, labeled with seed, operator, tour length, percentage gap, starting depot, and return leg

### Benchmark Workflow (Pseudocode)

```text
ALGORITHM: RunAllExperiments

1. INITIALIZE CONFIGURATION:
   - Instances   <- [(eil51, 426), (berlin52, 7542), (st70, 675), (eil76, 538), (kroA100, 21282), (ch150, 6528)]
   - Operators   <- [swap, reversal]
   - Seeds       <- [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
   - Parameters  <- n = 20, m = 6, e = 2, ne = 20, ns = 8, max_evals = 30000
   - Total Runs  <- 120 (6 instances times 2 operators times 10 seeds)

2. EXECUTE EXPERIMENTS:
   FOR each (instance_name, file_path) IN Instances:
       - Parse TSPLIB file and validate metadata (TYPE = TSP, EDGE_WEIGHT_TYPE = EUC_2D, DIMENSION)
       - Compute symmetric Euclidean distance matrix using rounded TSPLIB integer formula:
             d(i, j) <- floor( sqrt( (x_i - x_j)^2 + (y_i - y_j)^2 ) + 0.5 )
       
       FOR each operator IN Operators:
           FOR each seed IN Seeds:
               - Initialize RNG seed <- seed
               - Initialize TripEval with evaluation budget <- 30,000
               - Instantiate BeesAlgorithm(num_cities, evaluator, operator, n, m, e, ne, ns)
               - Execute BeesAlgorithm to retrieve (best_trip, best_length, history)
               - Calculate Gap(%) <- 100 * (best_length - optimum) / optimum
               - Record runtime, evaluation count, tour length, and gap

3. EXPORT RESULTS & VISUALIZATIONS:
   - Save full evaluation histories -> "results/results_detailed.json"
   - Save individual trial metrics  -> "results/results_summary.csv"
   - Compute aggregated metrics across seeds -> "results/table_summary.csv"
   - Generate convergence curves for berlin52 and ch150
   - Generate closed 2D route map for shortest berlin52 tour across all runs
```

### Algorithm Workflows (Pseudocode)

#### Bees Algorithm for TSP
```text
ALGORITHM: BeesAlgorithm (TSP)

1. INITIALIZATION:
   - Create initial scout population of size n:
     FOR i FROM 1 TO n:
         - Generate random permutation with City 1 fixed as start: (city 1, city 2, ..., city C)
         - Evaluate tour length of the permutation using distance matrix (including return leg to City 1)
         - Create Scout Bee(trip = permutation, length = tour_length)
   - Sort population in ascending order of tour length (minimization)

2. SEARCH LOOP (Repeat while evaluation budget < 30,000):
   a. ELITE SITES LOCAL SEARCH:
      - For each of the e best sites in the population:
          - Recruit ne forager bees to explore neighborhood:
              FOR j FROM 1 TO ne:
                  - IF budget exhausted, BREAK
                  - Mutate site tour using neighborhood operator (Swap or Reversal)
                  - Evaluate mutant tour length
                  - Keep best solution found in this flower patch
          - Add patch winner to new population

   b. REMAINING SELECTED SITES LOCAL SEARCH:
      - For each of the remaining (m - e) selected sites:
          - Recruit ns forager bees to explore neighborhood:
              FOR j FROM 1 TO ns:
                  - IF budget exhausted, BREAK
                  - Mutate site tour using neighborhood operator (Swap or Reversal)
                  - Evaluate mutant tour length
                  - Keep best solution found in this flower patch
          - Add patch winner to new population

   c. GLOBAL SCOUTING (Random Exploration):
      - For the remaining (n - m) bees:
          - Generate new random scout permutation with City 1 fixed
          - Evaluate tour length
          - Add fresh scout bee to new population

   d. POPULATION UPDATE:
      - Replace current population with the n newly formed bees
      - Sort population by tour length

3. OUTPUT:
   - Return best valid tour found, best tour length, and best-so-far evaluation history
```

#### Neighborhood Operators (Swap & Segment Reversal)
```text
OPERATOR: Swap
1. Input: Tour permutation = (start_city, city_2, ..., city_C)
2. Sample two distinct random indices i, j from range 1 to (num_cities - 1) (leaving start City 1 at index 0 fixed)
3. Swap cities at positions i and j: new_tour[i] <- tour[j], new_tour[j] <- tour[i]
4. Return mutated valid tour

OPERATOR: Segment Reversal (2-opt style)
1. Input: Tour permutation = (start_city, city_2, ..., city_C)
2. Sample two distinct random indices i, j from range 1 to (num_cities - 1) where index i is less than index j (City 1 at index 0 fixed)
3. Reverse the subsequence between i and j: new_tour[i to j] <- reversed segment of tour[i to j]
4. Return mutated valid tour
```

---

## Single Run (Testing Only)

Run a single trial or regenerate plots and summary tables from existing results:

```python
from tsplib_parser import TSPInstance
from distance import TripEval, create_distance_matrix
from operators import reverse_operator
from bees_algorithm import BeesAlgorithm

# Load instance and build distance matrix
instance = TSPInstance()
instance.read_file("BA/Part_A/data/berlin52.tsp")
dist_matrix = create_distance_matrix(instance.dataset)

# Initialize evaluator with 30,000 budget
evaluator = TripEval(dist_matrix)
evaluator.max_evals = 30_000

# Configure and run Bees Algorithm
ba = BeesAlgorithm(
    num_cities=len(instance.dataset),
    evaluator=evaluator,
    operator=reverse_operator,
    population_size=20,
    selected_sites=6,
    elite_sites=2,
    recruited_elite=20,
    recruited_selected=8,
)
best_trip, best_len, history = ba.run()
print(f"Optimal: 7542 | Best Found: {best_len} | Gap: {100.0 * (best_len - 7542) / 7542:.2f}%")
```

OR regenerate the summary tables and figures from previously saved results:

```bash
uv run BA/Part_A/visualization.py
# or: python BA/Part_A/visualization.py
```

### Parameters (`bees_algorithm.py`)
Configured in `BeesAlgorithm.__init__`:
```python
ba = BeesAlgorithm(
    num_cities=52,
    evaluator=evaluator,
    operator=reverse_operator,    # swap_operator or reverse_operator
    population_size=20,           # n: Total scout bees
    selected_sites=6,             # m: Number of best sites selected
    elite_sites=2,                # e: Elite sites subset
    recruited_elite=20,           # ne: Foragers per elite site
    recruited_selected=8          # ns: Foragers per non-elite selected site
)
```

---

## AI Usage Declaration
AI was utilized for documentation, visualization scripts, and benchmark reporting support.
Components that were developed or assisted using AI are detailed below.
They are also clearly marked in the code.
- `BA/Part_A/visualization.py`:
    - Summary table statistical aggregation (`generate_summary_table`)
    - Mean convergence curve generation across seeds with optimal reference lines (`plot_convergence`)
    - 2D champion route mapping with depot and return leg visualization (`plot_best_tour`)
- `BA/Part_A/experiments.py`:
    - Integration of visualization routines and experiment progress logging
- AI was used for pseudocode structuring and README documentation formatting
