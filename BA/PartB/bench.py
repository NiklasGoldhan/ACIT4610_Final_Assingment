import json
import record
from pathlib import Path

from experiment import BAParams, Config, run_all


# Define config for the runs
CONFIG = Config(
    master_seed=42,
    runs=10,
    functions=("sphere", "rosenbrock", "rastrigin", "ackley"),
    dimensions=(10, 30),
    max_evals=30_000,
    output_dir=Path("results"),
    ba=BAParams(
        ns=13, ne=2, nb=6, nre=9, nrb=6,
        ngh=0.1, stlim=20, shrink_factor=0.95,
    )
)


def run_experiment(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_json(), indent=2))

    results = run_all(cfg)

    rows = record.summary_rows(results)
    record.print_summary(rows)
    record.write_summary(rows, out)
    record.write_runs(results, out)
    record.write_best_so_far(results, out)

    # include these mandatories if present
    if ("rastrigin", 10) in results:
        record.plot_convergence(
            results, "rastrigin", 10, out / "rastrigin_convergence.png"
        )
    if 30 in cfg.dimensions:
        record.plot_boxplots(
            results, list(cfg.functions), 30, out / "final_boxplots_d30.png"
        )

    print("\nExperiments completed.")
    print(f"Results saved to: {out.resolve()}")


if __name__ == "__main__":
    run_experiment(CONFIG)
