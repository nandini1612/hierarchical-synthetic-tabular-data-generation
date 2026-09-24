"""
run_repair_ablation.py

Drop this file into the repo's `src/` directory, alongside run_experiments.py.
It reuses the repo's real synthesis/evaluation functions directly (same
imports run_experiments.py uses) and adds exactly one new thing: the
constraint-repair step, applied optionally, with violation-rate metrics
logged alongside the existing fidelity/utility columns.

Usage (from repo root, same convention as the other src/ scripts):

    python src/run_repair_ablation.py \
        --datasets weak_multimodal,weak_multimodal_gemini \
        --methods independent,random_forest,xgboost \
        --seeds 42,123,2024 \
        --n-rows 12000 \
        --output-dir data/processed/repair_ablation

This produces data/processed/repair_ablation/repair_ablation_results.csv
with one row per (dataset, method, seed, repair_applied) — a strict
superset of what experiment_results.csv already has, plus:
    violation_rate_before, violation_rate_after, repair_passes_run

That gives you the full ablation grid directly:
  - method=independent,       repair_applied=False  -> "no rule-conditioning, no repair" baseline
  - method=independent,       repair_applied=True   -> "repair only"
  - method=random_forest/xgboost, repair_applied=False -> "rule-conditioning only" (matches
        existing experiment_summary.csv numbers exactly, since this calls the same
        run_method() logic)
  - method=random_forest/xgboost, repair_applied=True  -> "both" (the proposed combination)

Why `independent` stands in for "no rule-conditioning": it's already the
one existing method that ignores conditional/sequential structure entirely
(baseline_independent_sampling.py samples each column's marginal
independently), so it's the closest existing analogue to "bottom-up
generation without any top-down structure" without writing a whole new
generator. This is a modeling choice worth flagging explicitly in the
writeup, not a hidden assumption.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

# Same imports as the real run_experiments.py (confirmed via audit_repo.py)
from baseline_gaussian_copula import gaussian_copula_synthesis
from baseline_independent_sampling import independent_column_sampling
from baseline_random_forest import conditional_random_forest_synthesis
from baseline_sdv import sdv_synthesis
from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import evaluate_utility
from prepare_tabular_dataset import clean_tabular_dataset
from synth_xgboost import conditional_xgboost_synthesis
from config_utils import CONFIG_DIR, load_json_config, resolve_project_path

# NEW: the repair module. Place constraint_repair.py in src/ alongside this file.
from constraint_repair import (
    BANK_MARKETING_CONSTRAINTS,
    repair_csv,
    compute_violation_report,
)


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSHOP_DIR / "data" / "processed" / "repair_ablation"
DEFAULT_DATASET_REGISTRY_JSON = CONFIG_DIR / "experiment_datasets.json"


def load_experiment_datasets(
    config_json: Path = DEFAULT_DATASET_REGISTRY_JSON,
) -> dict[str, Path]:
    configs = load_json_config(config_json)
    return {name: resolve_project_path(path) for name, path in configs.items()}


DATASETS = load_experiment_datasets()


def ensure_clean_datasets() -> None:
    clean_tabular_dataset("bank_marketing")
    clean_tabular_dataset("adult_income")
    clean_tabular_dataset("german_credit")


def summarize_fidelity(numeric_df, categorical_df, cross_modal_df) -> dict[str, float]:
    return {
        "mean_numeric_abs_mean_diff": float(numeric_df["abs_mean_diff"].mean())
        if not numeric_df.empty
        else float("nan"),
        "mean_numeric_abs_std_diff": float(numeric_df["abs_std_diff"].mean())
        if not numeric_df.empty
        else float("nan"),
        "mean_categorical_tvd": (
            float(categorical_df["total_variation_distance"].mean())
            if not categorical_df.empty
            else float("nan")
        ),
        "mean_cross_modal_abs_diff": (
            float(cross_modal_df["abs_diff"].mean())
            if not cross_modal_df.empty
            else float("nan")
        ),
    }


def run_method(
    method: str, real_csv: Path, synth_csv: Path, seed: int, n_rows: int | None
) -> None:
    """Identical to run_experiments.py's run_method — unchanged."""
    if method == "independent":
        independent_column_sampling(
            input_csv=real_csv, output_csv=synth_csv, n_rows=n_rows, random_state=seed
        )
    elif method == "gaussian_copula":
        gaussian_copula_synthesis(
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
    elif method in {"ctgan", "tvae"}:
        sdv_synthesis(
            input_csv=real_csv,
            output_csv=synth_csv,
            method=method,
            n_rows=n_rows,
            random_state=seed,
            epochs=50,
            metadata_json=synth_csv.with_name("metadata.json"),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def run_repair_ablation(
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    output_dir: Path,
    n_rows: int | None,
    keep_run_artifacts: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for dataset in datasets:
        real_csv = DATASETS[dataset]
        if not real_csv.exists():
            raise FileNotFoundError(f"Missing processed dataset: {real_csv}")

        for method in methods:
            for seed in seeds:
                for repair_mode in ("none", "knn", "canonical"):
                    repair_applied = repair_mode != "none"
                    tag = f"{method}{'_' + repair_mode if repair_applied else ''}"
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
                            f"Running dataset={dataset} method={method} seed={seed} repair_mode={repair_mode}"
                        )
                        run_method(
                            method=method,
                            real_csv=real_csv,
                            synth_csv=synth_csv,
                            seed=seed,
                            n_rows=n_rows,
                        )

                        # violation rate is always computed (even when not repairing) so the
                        # "rule-conditioning only" arm has a comparable before-repair number.
                        # Only actually rewrite synth_csv when repair_applied is True.
                        if repair_applied:
                            violation_report = repair_csv(
                                synth_csv=synth_csv,
                                real_csv=real_csv,
                                output_csv=None,  # repair in place
                                constraints=BANK_MARKETING_CONSTRAINTS,
                                mode=repair_mode,
                            )
                        else:
                            unrepaired_df = pd.read_csv(synth_csv)
                            violation_report = {
                                "before": compute_violation_report(
                                    unrepaired_df, BANK_MARKETING_CONSTRAINTS
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
                    trtr = utility_map.loc["train_real_test_real"]
                    tstr = utility_map.loc["train_synthetic_test_real"]
                    gap = utility_map.loc["gap_tstr_minus_trtr"]

                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "repair_applied": repair_applied,
                            "repair_mode": repair_mode,
                            "seed": seed,
                            "trtr_accuracy": float(trtr["accuracy"]),
                            "trtr_f1": float(trtr["f1"]),
                            "trtr_auroc": float(trtr["auroc"]),
                            "tstr_accuracy": float(tstr["accuracy"]),
                            "tstr_f1": float(tstr["f1"]),
                            "tstr_auroc": float(tstr["auroc"]),
                            "gap_accuracy": float(gap["accuracy"]),
                            "gap_f1": float(gap["f1"]),
                            "gap_auroc": float(gap["auroc"]),
                            "violation_rate_before": violation_report["before"][
                                "any_violation_rate"
                            ],
                            "violation_rate_after": (
                                violation_report["after"]["any_violation_rate"]
                                if repair_applied
                                else None
                            ),
                            "repair_passes_run": violation_report.get("passes_run")
                            if repair_applied
                            else None,
                            **summarize_fidelity(
                                numeric_df, categorical_df, cross_modal_df
                            ),
                        }
                    )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "repair_ablation_results.csv", index=False)

    summary_df = (
        results_df.groupby(["dataset", "method", "repair_mode"], as_index=False)
        .agg(
            tstr_accuracy_mean=("tstr_accuracy", "mean"),
            tstr_accuracy_std=("tstr_accuracy", "std"),
            tstr_f1_mean=("tstr_f1", "mean"),
            tstr_f1_std=("tstr_f1", "std"),
            tstr_auroc_mean=("tstr_auroc", "mean"),
            tstr_auroc_std=("tstr_auroc", "std"),
            violation_rate_before_mean=("violation_rate_before", "mean"),
            violation_rate_after_mean=("violation_rate_after", "mean"),
            mean_cross_modal_abs_diff=("mean_cross_modal_abs_diff", "mean"),
        )
        .sort_values(["dataset", "method", "repair_mode"])
    )
    summary_df.to_csv(output_dir / "repair_ablation_summary.csv", index=False)
    return summary_df


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule-conditioning x repair ablation."
    )
    parser.add_argument(
        "--datasets",
        type=parse_csv_list,
        default=["weak_multimodal", "weak_multimodal_gemini"],
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
    ensure_clean_datasets()
    summary_df = run_repair_ablation(
        datasets=args.datasets,
        methods=args.methods,
        seeds=seeds,
        output_dir=args.output_dir,
        n_rows=args.n_rows,
        keep_run_artifacts=args.keep_run_artifacts,
    )
    print(
        f"Saved detailed results to: {args.output_dir / 'repair_ablation_results.csv'}"
    )
    print(f"Saved summary to: {args.output_dir / 'repair_ablation_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
