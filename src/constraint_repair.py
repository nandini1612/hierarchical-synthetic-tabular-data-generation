"""
constraint_repair.py

New module implementing the "repair" mechanism described in the project
brief (Section 3.2/3.3). This is intentionally self-contained: it operates
on plain pandas DataFrames and does not import anything from the repo's
src/ package, so it works today regardless of the exact internal function
signatures in run_experiments.py / evaluate_utility.py (which we haven't
inspected yet — run audit_repo.py first to get those).

Integration into the pipeline (once audit_repo.py output confirms exact
call sites) is a single insertion point:

    synthetic_df = synthesize(...)                       # existing
    synthetic_df, violation_report = repair_dataframe(     # NEW
        synthetic_df, real_df, constraints=BANK_MARKETING_CONSTRAINTS
    )
    evaluate_fidelity(synthetic_df, real_df)               # existing, unchanged
    evaluate_utility(synthetic_df, real_df)                 # existing, unchanged

--------------------------------------------------------------------------
IMPORTANT CAVEAT ON COLUMN NAMES
--------------------------------------------------------------------------
The constraints below are written against the STANDARD UCI Bank Marketing
schema (age, job, marital, education, default, balance, housing, loan,
contact, day, month, duration, campaign, pdays, previous, poutcome, y).
The repo's `prepare_tabular_dataset.py` may rename or drop columns (the
README confirms it creates a derived binary `target` column and may strip
raw label columns like `y`). Run audit_repo.py's processed-data-schema
section and compare against COLUMN NAMES ASSUMED below before running this
on real processed data — adjust the `COLS` mapping at the top if names
differ. Everything else in this file is written against that COLS mapping,
not hardcoded strings, specifically so a rename is a one-line fix.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# --------------------------------------------------------------------------
# COLUMN NAMES ASSUMED — edit this mapping if audit_repo.py shows different
# names in the processed CSV (e.g. if `y` was renamed to `target`).
# --------------------------------------------------------------------------
COLS = {
    "age": "age",
    "job": "job",
    "marital": "marital",
    "education": "education",
    "default": "default",
    "balance": "balance",
    "housing": "housing",
    "loan": "loan",
    "contact": "contact",
    "day": "day",
    "month": "month",
    "duration": "duration",
    "campaign": "campaign",
    "pdays": "pdays",
    "previous": "previous",
    "poutcome": "poutcome",
}


@dataclasses.dataclass
class Constraint:
    """A single hard logical constraint.

    `check` returns a boolean Series over the dataframe: True where the
    constraint is VIOLATED (not where it holds) — this matches how you'll
    want to aggregate violation rates and select rows for repair.

    `repair_cols` lists which columns should be overwritten (from a nearest
    real-data neighbor) when this constraint is violated on a row. Keep
    this minimal — only the column(s) that actually caused the
    inconsistency, not the whole row, so repair perturbs synthetic rows as
    little as possible. Ignored when `deterministic_fix` is set.

    `deterministic_fix` is an OPTIONAL callable for constraints where the
    correct value is fully known from other columns in the SAME row (e.g.
    education -> education_num is a fixed 1:1 mapping; relationship=Husband
    implies gender=Male by definition). When set, repair uses this instead
    of k-NN lookup: it's exact rather than approximate, and resolves in one
    pass instead of needing multiple convergence passes. Signature:
    `deterministic_fix(df) -> dict[col_name, pd.Series]` where each Series
    is aligned to df's index and gives the correct value for EVERY row
    (repair only applies it to the violating subset, so it's fine to
    compute it for the whole frame). Leave as None for constraints where
    the correct value isn't derivable from the row itself and a
    nearest-real-neighbor lookup is the best available fix.
    """

    id: str
    description: str
    check: Callable[[pd.DataFrame], pd.Series]
    repair_cols: tuple
    deterministic_fix: Callable[[pd.DataFrame], dict] | None = None
    canonical_fix: Callable[[pd.DataFrame], dict] | None = None
    # `canonical_fix` is an OPTIONAL callable used only by repair mode
    # "canonical". Unlike `deterministic_fix` (the UNIQUE correct value that
    # is logically forced by the row, e.g. education->education_num), a
    # canonical fix is a *chosen* internally-consistent default for
    # constraints where several repairs are all logically valid — it uses
    # ONLY the row's own values and injects no real data. Its purpose is to
    # test whether repair's utility effect comes from logical validity per se
    # (canonical) or from copying real neighbours (k-NN). Same signature as
    # deterministic_fix: `canonical_fix(df) -> dict[col_name, pd.Series]`,
    # each Series aligned to df's index; repair applies it only to the
    # violating subset. Ignored in the default "knn" mode.


def _col(df, key):
    name = COLS[key]
    if name not in df.columns:
        raise KeyError(
            f"Expected column '{name}' (mapped from COLS['{key}']) not found "
            f"in dataframe. Columns present: {list(df.columns)}. "
            f"Update COLS at the top of constraint_repair.py to match your "
            f"processed schema (see audit_repo.py output)."
        )
    return df[name]


# --------------------------------------------------------------------------
# Constraint definitions
# --------------------------------------------------------------------------
# These encode genuine logical/temporal dependencies in the Bank Marketing
# schema — not soft statistical correlations. Each one is a real-world
# fact that must hold for a row to represent a coherent record, which is
# exactly the class of error that TSTR/fidelity metrics do not catch
# (a row can have plausible marginal values for every column individually
# while combining them in a way that could not occur in a real record).


def _never_contacted_but_has_history(df: pd.DataFrame) -> pd.Series:
    """pdays == -1 means 'not previously contacted'. That is inconsistent
    with previous > 0 (a prior-contact count) or poutcome indicating a
    previous outcome other than 'unknown'.
    Real poutcome categories (confirmed against processed data): unknown,
    failure, other, success — there is no 'nonexistent' category."""
    pdays = _col(df, "pdays")
    previous = _col(df, "previous")
    poutcome = _col(df, "poutcome")
    never_contacted = pdays == -1
    has_history = (previous > 0) | (poutcome != "unknown")
    return never_contacted & has_history


def _contacted_before_but_no_history(df: pd.DataFrame) -> pd.Series:
    """The inverse: pdays >= 0 (days since last contact is known) implies
    the client WAS previously contacted, so previous should be > 0 and
    poutcome should not be 'unknown'."""
    pdays = _col(df, "pdays")
    previous = _col(df, "previous")
    poutcome = _col(df, "poutcome")
    contacted_before = pdays >= 0
    no_history = (previous == 0) & (poutcome == "unknown")
    return contacted_before & no_history


def _success_without_prior_contact(df: pd.DataFrame) -> pd.Series:
    """poutcome == 'success' cannot occur if previous == 0 — you cannot
    have a successful PRIOR campaign outcome with zero prior contacts."""
    poutcome = _col(df, "poutcome")
    previous = _col(df, "previous")
    return (poutcome == "success") & (previous == 0)


def _implausible_age(df: pd.DataFrame) -> pd.Series:
    """Bank Marketing's real age range is bounded; values outside plausible
    working-adult-to-retiree range indicate a synthesis artifact."""
    age = _col(df, "age")
    return (age < 18) | (age > 95)


def _non_positive_campaign_contacts(df: pd.DataFrame) -> pd.Series:
    """campaign counts contacts made DURING this campaign, including the
    current one — it cannot be zero or negative for a row that exists."""
    campaign = _col(df, "campaign")
    return campaign < 1


def _negative_duration(df: pd.DataFrame) -> pd.Series:
    """Call duration cannot be negative."""
    duration = _col(df, "duration")
    return duration < 0


# --------------------------------------------------------------------------
# Canonical fixes (used only by repair mode="canonical").
# Each returns {col: Series-over-whole-frame}; repair applies to the violating
# subset only. All are row-only and inject no real data. For the three
# temporal constraints the canonical choice is "declare the record never
# previously contacted", which is the internally-consistent state reachable
# from any of these violations by editing only the pdays/previous/poutcome
# triple.
# --------------------------------------------------------------------------


def _canon_never_contacted_but_has_history(df: pd.DataFrame) -> dict:
    # pdays already == -1 on the violating rows; clear the spurious history.
    return {
        COLS["previous"]: pd.Series(0, index=df.index),
        COLS["poutcome"]: pd.Series("unknown", index=df.index),
    }


def _canon_contacted_before_but_no_history(df: pd.DataFrame) -> dict:
    # previous==0 & poutcome=="unknown" on the violating rows; make pdays agree.
    return {COLS["pdays"]: pd.Series(-1, index=df.index)}


def _canon_success_without_prior_contact(df: pd.DataFrame) -> dict:
    # previous==0 on the violating rows; a prior success is impossible -> unknown.
    return {COLS["poutcome"]: pd.Series("unknown", index=df.index)}


def _canon_implausible_age(df: pd.DataFrame) -> dict:
    return {COLS["age"]: _col(df, "age").clip(18, 95)}


def _canon_non_positive_campaign(df: pd.DataFrame) -> dict:
    return {COLS["campaign"]: _col(df, "campaign").clip(lower=1)}


def _canon_negative_duration(df: pd.DataFrame) -> dict:
    return {COLS["duration"]: _col(df, "duration").clip(lower=0)}


BANK_MARKETING_CONSTRAINTS: list[Constraint] = [
    Constraint(
        id="pdays_never_contacted_has_history",
        description="pdays=-1 (never contacted) but previous>0 or poutcome not unknown/nonexistent",
        check=_never_contacted_but_has_history,
        # Copy the FULL (pdays, previous, poutcome) triple from one coherent
        # real neighbour. Including pdays is required for convergence: copying
        # only previous/poutcome from a contacted-before neighbour onto a row
        # that still has pdays=-1 re-creates this same violation (verified: a
        # stable ~6.5% k-NN plateau on Bank Marketing before this fix).
        repair_cols=(COLS["pdays"], COLS["previous"], COLS["poutcome"]),
        canonical_fix=_canon_never_contacted_but_has_history,
    ),
    Constraint(
        id="pdays_contacted_before_no_history",
        description="pdays>=0 (contacted before) but previous=0 and poutcome unknown/nonexistent",
        check=_contacted_before_but_no_history,
        repair_cols=(COLS["pdays"], COLS["previous"], COLS["poutcome"]),
        canonical_fix=_canon_contacted_before_but_no_history,
    ),
    Constraint(
        id="success_without_prior_contact",
        description="poutcome=success but previous=0 (impossible: no prior contacts, prior success)",
        check=_success_without_prior_contact,
        # Same reasoning as the first constraint: repair the whole triple from
        # a single coherent neighbour so the copied poutcome/previous can't
        # conflict with a stale pdays left behind.
        repair_cols=(COLS["pdays"], COLS["poutcome"], COLS["previous"]),
        canonical_fix=_canon_success_without_prior_contact,
    ),
    Constraint(
        id="implausible_age",
        description="age outside plausible [18, 95] range",
        check=_implausible_age,
        repair_cols=(COLS["age"],),
        canonical_fix=_canon_implausible_age,
    ),
    Constraint(
        id="non_positive_campaign",
        description="campaign < 1 (must include at least the current contact)",
        check=_non_positive_campaign_contacts,
        repair_cols=(COLS["campaign"],),
        canonical_fix=_canon_non_positive_campaign,
    ),
    Constraint(
        id="negative_duration",
        description="duration < 0 (call duration cannot be negative)",
        check=_negative_duration,
        repair_cols=(COLS["duration"],),
        canonical_fix=_canon_negative_duration,
    ),
]


# --------------------------------------------------------------------------
# Violation checking / metric
# --------------------------------------------------------------------------


def compute_violation_report(
    df: pd.DataFrame, constraints: Iterable[Constraint] = BANK_MARKETING_CONSTRAINTS
) -> dict:
    """Returns a dict matching the style of evaluate_fidelity.py's report
    (flat dict of scalar metrics) so it can be merged into the same output
    row / CSV without restructuring the existing result schema.

    Reports:
      - one violation RATE per constraint (fraction of rows violating it)
      - `any_violation_rate`: fraction of rows violating >=1 constraint
      - `mean_violations_per_row`: average count of constraints violated per row
    """
    n = len(df)
    report = {}
    violated_any = pd.Series(False, index=df.index)
    total_violations = pd.Series(0, index=df.index)

    for c in constraints:
        mask = c.check(df)
        rate = float(mask.mean()) if n > 0 else float("nan")
        report[f"violation_rate__{c.id}"] = rate
        violated_any = violated_any | mask
        total_violations = total_violations + mask.astype(int)

    report["any_violation_rate"] = float(violated_any.mean()) if n > 0 else float("nan")
    report["mean_violations_per_row"] = (
        float(total_violations.mean()) if n > 0 else float("nan")
    )
    return report


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def _build_nn_index(real_df: pd.DataFrame, feature_cols: list[str]):
    """Fit a k-NN index over the REAL data in a numeric-encoded feature
    space. Categorical columns are ordinal-encoded, numeric columns are
    standardized — this is a fast, dependency-light approach (no need for
    a fitted embedding model) appropriate for a first repair-mechanism
    prototype. Swap for a smarter distance metric later if this proves
    too coarse in practice.
    """
    frame = real_df[feature_cols].copy()
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(frame[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    encoders = {}
    encoded_parts = []

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_cat = enc.fit_transform(frame[cat_cols].astype(str))
        encoders["cat"] = (enc, cat_cols)
        encoded_parts.append(encoded_cat)

    if num_cols:
        scaler = StandardScaler()
        encoded_num = scaler.fit_transform(frame[num_cols].astype(float))
        encoders["num"] = (scaler, num_cols)
        encoded_parts.append(encoded_num)

    X = np.hstack(encoded_parts) if len(encoded_parts) > 1 else encoded_parts[0]
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    return nn, encoders, feature_cols


def _encode_rows(
    rows: pd.DataFrame, encoders: dict, feature_cols: list[str]
) -> np.ndarray:
    parts = []
    if "cat" in encoders:
        enc, cat_cols = encoders["cat"]
        parts.append(enc.transform(rows[cat_cols].astype(str)))
    if "num" in encoders:
        scaler, num_cols = encoders["num"]
        parts.append(scaler.transform(rows[num_cols].astype(float)))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def repair_dataframe(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    constraints: Iterable[Constraint] = BANK_MARKETING_CONSTRAINTS,
    feature_cols: list[str] | None = None,
    max_passes: int = 3,
    random_state: int = 42,
    mode: str = "knn",
) -> tuple[pd.DataFrame, dict]:
    """Repairs constraint violations in synthetic_df.

    `mode` selects the repair strategy for constraints that don't have a
    `deterministic_fix` (which is always applied exactly, in either mode):
      - "knn" (default): nearest-real-neighbour projection — copies the
        implicated columns from the most similar real row. Minimally
        invasive but injects real data, so any downstream utility change
        it produces is confounded with real-data leakage.
      - "canonical": apply each constraint's `canonical_fix` — an
        internally-consistent default computed from the row's OWN values
        only, injecting no real data. Used to isolate the utility effect of
        logical validity itself from the leakage effect of k-NN. Constraints
        with no `canonical_fix` fall back to k-NN so unhandled cases still
        get repaired.

    Original k-NN description follows (applies to mode="knn").

    Strategy per violating row, per constraint:
      1. Find its nearest real-data neighbor (over `feature_cols`, or all
         shared columns if not specified).
      2. Overwrite ONLY the columns implicated by that constraint
         (`constraint.repair_cols`) with that neighbor's values.
      3. Leave every other column in the row untouched — repair should be
         minimally invasive, not a full-row replacement (a full-row
         replacement would just be resampling from real data with extra
         steps, and would defeat the point of testing repair as a
         separate mechanism from bottom-up generation).

    Constraints can share repair_cols (e.g. `previous`/`poutcome` appear in
    three constraints here), so fixing one can transiently re-violate
    another. `max_passes` re-runs the full constraint sweep until either no
    constraints are violated or the pass limit is hit, so the reported
    "after" numbers reflect convergence rather than a single pass. In
    practice this converges in 2 passes on the constraint set defined here.

    Returns (repaired_df, {"before": <violation report>, "after": <violation report>, "passes_run": int}).
    """
    if mode not in {"knn", "canonical"}:
        raise ValueError(f"Unknown repair mode: {mode!r} (expected 'knn' or 'canonical')")

    df = synthetic_df.copy().reset_index(drop=True)
    before_report = compute_violation_report(df, constraints)

    if feature_cols is None:
        feature_cols = [c for c in real_df.columns if c in df.columns]

    # The k-NN index is only needed when at least one constraint will actually
    # fall through to nearest-neighbour repair. In mode="canonical" with a
    # canonical_fix on every constraint (as for Bank Marketing) it is never
    # needed, so build it lazily to avoid the cost.
    real_df_reset = real_df.reset_index(drop=True)
    nn = encoders = None

    def _ensure_nn_index():
        nonlocal nn, encoders, feature_cols
        if nn is None:
            nn, encoders, feature_cols = _build_nn_index(real_df, feature_cols)

    passes_run = 0
    for _ in range(max_passes):
        passes_run += 1
        any_violation_this_pass = False

        for c in constraints:
            mask = c.check(df)
            n_violating = int(mask.sum())
            if n_violating == 0:
                continue
            any_violation_this_pass = True

            # An exact `deterministic_fix` is always applied when present, in
            # either mode — it is the unique correct value, not a strategy choice.
            fix = c.deterministic_fix
            # In canonical mode, prefer the constraint's canonical_fix (row-only,
            # no real data). Fall back to k-NN when neither fix is available.
            if fix is None and mode == "canonical":
                fix = c.canonical_fix

            if fix is not None:
                fix_values = fix(df)
                violating_positions = df.index[mask]
                for col, series in fix_values.items():
                    if col in df.columns:
                        df.loc[violating_positions, col] = series.loc[
                            violating_positions
                        ]
                continue

            _ensure_nn_index()
            violating_rows = df.loc[mask, feature_cols]
            encoded = _encode_rows(violating_rows, encoders, feature_cols)
            _, neighbor_idx = nn.kneighbors(encoded, n_neighbors=1)
            neighbor_idx = neighbor_idx[:, 0]

            violating_positions = df.index[mask]
            for pos, nbr in zip(violating_positions, neighbor_idx):
                for col in c.repair_cols:
                    if col in df.columns and col in real_df_reset.columns:
                        df.at[pos, col] = real_df_reset.at[nbr, col]

        if not any_violation_this_pass:
            break

    after_report = compute_violation_report(df, constraints)
    return df, {
        "before": before_report,
        "after": after_report,
        "passes_run": passes_run,
    }


# --------------------------------------------------------------------------
# CSV-path wrapper — matches the repo's convention. Every existing synthesis
# and evaluation function in this repo (independent_column_sampling,
# conditional_random_forest_synthesis, evaluate_fidelity, evaluate_utility,
# ...) takes `input_csv`/`output_csv` Path arguments and reads/writes CSV
# files directly, rather than passing dataframes around in memory. This
# wrapper matches that convention so it drops into run_experiments.py's
# run_method()/run_experiments() flow with the same calling shape as every
# other step already there.
# --------------------------------------------------------------------------


def repair_csv(
    synth_csv,
    real_csv,
    output_csv=None,
    constraints: Iterable[Constraint] = BANK_MARKETING_CONSTRAINTS,
    max_passes: int = 3,
    mode: str = "knn",
) -> dict:
    """Reads synth_csv and real_csv, repairs synth_csv's violations against
    real_csv, and writes the repaired result to output_csv (or back to
    synth_csv in place if output_csv is None — matching how the existing
    pipeline treats synth_csv as a disposable per-run artifact).

    Returns the violation report dict ({"before": ..., "after": ...,
    "passes_run": ...}) so the caller can log it into the results row
    exactly like summarize_fidelity()'s dict is merged into a row in
    run_experiments.py.
    """
    from pathlib import Path

    synth_csv = Path(synth_csv)
    real_csv = Path(real_csv)
    target = Path(output_csv) if output_csv is not None else synth_csv

    synthetic_df = pd.read_csv(synth_csv)
    real_df = pd.read_csv(real_csv)

    repaired_df, report = repair_dataframe(
        synthetic_df, real_df, constraints=constraints, max_passes=max_passes, mode=mode
    )

    # evaluate_fidelity() requires real and synthetic CSVs to have IDENTICAL
    # columns in the SAME ORDER (see evaluate_fidelity.py's explicit check).
    # repair_dataframe() only overwrites cell values, never adds/drops/
    # reorders columns, so this is preserved automatically — but reassert
    # the real_df column order defensively in case a future constraint
    # touches column structure.
    repaired_df = repaired_df[list(synthetic_df.columns)]

    target.parent.mkdir(parents=True, exist_ok=True)
    repaired_df.to_csv(target, index=False)
    return report


# --------------------------------------------------------------------------
# Standalone smoke test — run this file directly to sanity-check the logic
# on synthetic toy data before wiring it into the real pipeline.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500

    real = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "job": rng.choice(
                ["admin.", "technician", "blue-collar", "management", "student"], n
            ),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["primary", "secondary", "tertiary"], n),
            "default": rng.choice(["yes", "no"], n, p=[0.05, 0.95]),
            "balance": rng.integers(-2000, 20000, n),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n),
            "day": rng.integers(1, 28, n),
            "month": rng.choice(["jan", "feb", "mar", "apr", "may"], n),
            "duration": rng.integers(0, 3000, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.choice([-1] + list(range(1, 400)), n),
            "previous": rng.integers(0, 5, n),
            "poutcome": rng.choice(["unknown", "other", "failure", "success"], n),
        }
    )
    # enforce internal consistency in the "real" reference set
    real.loc[real["pdays"] == -1, "previous"] = 0
    real.loc[real["pdays"] == -1, "poutcome"] = "unknown"
    real.loc[real["previous"] == 0, "poutcome"] = real.loc[
        real["previous"] == 0, "poutcome"
    ].replace("success", "unknown")

    # a deliberately messy "synthetic" set with injected violations
    synth = real.copy()
    corrupt_idx = rng.choice(n, size=80, replace=False)
    synth.loc[corrupt_idx[:20], "previous"] = (
        3  # pdays=-1 rows now have history -> violation
    )
    synth.loc[corrupt_idx[:20], "pdays"] = -1
    synth.loc[corrupt_idx[20:40], "poutcome"] = "success"
    synth.loc[corrupt_idx[20:40], "previous"] = (
        0  # success w/ 0 prior contacts -> violation
    )
    synth.loc[corrupt_idx[40:60], "age"] = rng.integers(100, 150, 20)  # implausible age
    synth.loc[corrupt_idx[60:], "duration"] = -rng.integers(
        1, 500, 20
    )  # negative duration

    repaired, reports = repair_dataframe(synth, real)

    print("BEFORE repair:")
    for k, v in reports["before"].items():
        print(f"  {k}: {v:.4f}")
    print("\nAFTER repair:")
    for k, v in reports["after"].items():
        print(f"  {k}: {v:.4f}")
