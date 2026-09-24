"""
robustness_check.py

Tests whether the repair mechanism (not the rule-conditioning /
repair interaction -- see the framing note in adult_german_constraints.py)
generalizes beyond the Bank Marketing weak-multimodal benchmark.

Place this in src/ alongside constraint_repair.py and
adult_german_constraints.py.

Usage (from repo root):

    python src/robustness_check.py \
        --datasets adult_income,german_credit \
        --methods independent,random_forest,xgboost \
        --seeds 42,123,2024 \
        --output-dir data/processed/robustness_check

Produces data/processed/robustness_check/robustness_results.csv and
robustness_summary.csv, same shape as repair_ablation's outputs, so they
can be compared side by side.

What to look for in the results (see the framing note for why this is a
narrower claim than the weak_multimodal ablation):
  - violation_rate_before -> violation_rate_after should drop sharply for
    all methods, on BOTH datasets, if the repair mechanism itself
    generalizes.
  - tstr_auroc should stay roughly flat (within seed noise) for
    random_forest/xgboost with repair on vs off -- same "repair is free"
    result as the weak_multimodal ablation, tested on a different schema.
  - independent + repair may show the same "some utility recovery, but
    far below what conditional generation achieves" pattern seen on
    weak_multimodal -- or may not; these datasets have different amounts
    of inherent logical structure than Bank Marketing does, so this is
    a genuine open question, not a predicted result.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from baseline_independent_sampling import independent_column_sampling
from baseline_random_forest import conditional_random_forest_synthesis
from synth_xgboost import conditional_xgboost_synthesis
from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import evaluate_utility
from prepare_tabular_dataset import clean_tabular_dataset
from config_utils import CONFIG_DIR, load_json_config, resolve_project_path

from constraint_repair import repair_csv, compute_violation_report
from adult_german_constraints import ADULT_INCOME_CONSTRAINTS, GERMAN_CREDIT_CONSTRAINTS


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSHOP_DIR / "data" / "processed" / "robustness_check"
DEFAULT_DATASET_REGISTRY_JSON = CONFIG_DIR / "experiment_datasets.json"

CONSTRAINTS_BY_DATASET = {
    "adult_income": ADULT_INCOME_CONSTRAINTS,
    "german_credit": GERMAN_CREDIT_CONSTRAINTS,
}


def load_experiment_datasets(
    config_json: Path = DEFAULT_DATASET_REGISTRY_JSON,
) -> dict[str, Path]:
    configs = load_json_config(config_json)
    return {name: resolve_project_path(path) for name, path in configs.items()}


DATASETS = load_experiment_datasets()


def run_method(
    method: str, real_csv: Path, synth_csv: Path, seed: int, n_rows: int | None
) -> None:
    if method == "independent":
        independent_column_sampling(
            input_csv=real_csv, output_csv=synth_csv, n_rows=n_rows, random_state=seed
        )
    elif method == "random_forest":
        conditional_random_forest_synthesis(
            input_csv=real_csv,
            output_csv=synth_csv,
            n_rows=n_rows,
            random_state=seed,
            max_training_rows=5000,
            max_trees=30,
            max_condition_cols=12,
        )
    elif method == "xgboost":
        conditional_xgboost_synthesis(
            input_csv=real_csv,
            output_csv=synth_csv,
            n_rows=n_rows,
            random_state=seed,
            max_training_rows=8000,
            max_condition_cols=12,
            n_estimators=80,
            max_depth=6,
        )
    else:
        raise ValueError(
            f"robustness_check.py supports independent/random_forest/xgboost only, got: {method}"
        )


def run_robustness_check(
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    output_dir: Path,
    n_rows: int | None,
    keep_run_artifacts: bool = False,
) -> pd.DataFrame:
    for d in datasets:
        if d not in CONSTRAINTS_BY_DATASET:
            raise ValueError(
                f"No constraint set defined for dataset '{d}'. "
                f"robustness_check.py currently supports: {list(CONSTRAINTS_BY_DATASET)}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for dataset in datasets:
        real_csv = DATASETS[dataset]
        if not real_csv.exists():
            raise FileNotFoundError(f"Missing processed dataset: {real_csv}")
        constraints = CONSTRAINTS_BY_DATASET[dataset]

        for method in methods:
            for seed in seeds:
                for repair_applied in (False, True):
                    tag = f"{method}{'_repaired' if repair_applied else ''}"
                    artifact_dir = output_dir / dataset / tag / f"seed_{seed}"
                    if keep_run_artifacts:
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        run_context = None
                        run_dir = artifact_dir
                    else:
                        run_context = tempfile.TemporaryDirectory(
                            prefix=f"{dataset}_{tag}_{seed}_"
                        )
                        run_dir = Path(run_context.name)

                    try:
                        synth_csv = run_dir / "synthetic.csv"
                        fidelity_csv = run_dir / "fidelity.csv"
                        utility_csv = run_dir / "utility.csv"

                        print(
                            f"Running dataset={dataset} method={method} seed={seed} repair={repair_applied}"
                        )
                        run_method(
                            method=method,
                            real_csv=real_csv,
                            synth_csv=synth_csv,
                            seed=seed,
                            n_rows=n_rows,
                        )

                        if repair_applied:
                            violation_report = repair_csv(
                                synth_csv=synth_csv,
                                real_csv=real_csv,
                                output_csv=None,
                                constraints=constraints,
                            )
                        else:
                            unrepaired_df = pd.read_csv(synth_csv)
                            violation_report = {
                                "before": compute_violation_report(
                                    unrepaired_df, constraints
                                ),
                                "after": None,
                                "passes_run": None,
                            }

                        numeric_df, categorical_df, cross_modal_df = evaluate_fidelity(
                            real_csv=real_csv,
                            synth_csv=synth_csv,
                            output_csv=fidelity_csv,
                        )
                        utility_df = evaluate_utility(
                            real_csv=real_csv,
                            synth_csv=synth_csv,
                            output_csv=utility_csv,
                            random_state=seed,
                        )
                    finally:
                        if run_context is not None:
                            run_context.cleanup()

                    utility_map = utility_df.set_index("setting")
                    tstr = utility_map.loc["train_synthetic_test_real"]
                    trtr = utility_map.loc["train_real_test_real"]

                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "repair_applied": repair_applied,
                            "seed": seed,
                            "trtr_auroc": float(trtr["auroc"]),
                            "tstr_accuracy": float(tstr["accuracy"]),
                            "tstr_f1": float(tstr["f1"]),
                            "tstr_auroc": float(tstr["auroc"]),
                            "violation_rate_before": violation_report["before"][
                                "any_violation_rate"
                            ],
                            "violation_rate_after": (
                                violation_report["after"]["any_violation_rate"]
                                if repair_applied
                                else None
                            ),
                        }
                    )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "robustness_results.csv", index=False)

    summary_df = (
        results_df.groupby(["dataset", "method", "repair_applied"], as_index=False)
        .agg(
            tstr_auroc_mean=("tstr_auroc", "mean"),
            tstr_auroc_std=("tstr_auroc", "std"),
            tstr_f1_mean=("tstr_f1", "mean"),
            violation_rate_before_mean=("violation_rate_before", "mean"),
            violation_rate_after_mean=("violation_rate_after", "mean"),
        )
        .sort_values(["dataset", "method", "repair_applied"])
    )
    summary_df.to_csv(output_dir / "robustness_summary.csv", index=False)
    return summary_df


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the repair mechanism generalizes beyond Bank Marketing."
    )
    parser.add_argument(
        "--datasets", type=parse_csv_list, default=["adult_income", "german_credit"]
    )
    parser.add_argument(
        "--methods",
        type=parse_csv_list,
        default=["independent", "random_forest", "xgboost"],
    )
    parser.add_argument("--seeds", type=parse_csv_list, default=["42", "123", "2024"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-rows", type=int, default=12000)
    parser.add_argument("--keep-run-artifacts", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds]
    for d in args.datasets:
        if d == "adult_income":
            clean_tabular_dataset("adult_income")
        elif d == "german_credit":
            clean_tabular_dataset("german_credit")

    summary_df = run_robustness_check(
        datasets=args.datasets,
        methods=args.methods,
        seeds=seeds,
        output_dir=args.output_dir,
        n_rows=args.n_rows,
        keep_run_artifacts=args.keep_run_artifacts,
    )
    print(f"Saved detailed results to: {args.output_dir / 'robustness_results.csv'}")
    print(f"Saved summary to: {args.output_dir / 'robustness_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
