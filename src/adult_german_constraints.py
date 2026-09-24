"""
adult_german_constraints.py

Constraint sets for the two tabular-only robustness-check benchmarks
(Adult Income, German Credit). Place this in src/ alongside
constraint_repair.py.

IMPORTANT FRAMING NOTE (read before using results from this file):
Adult Income and German Credit have no rule-conditioning / cross-modal
layer at all in this repo (no text, no alignment rule) -- they exist
purely as tabular-only sanity checks. So running repair on them tests
ONE thing only: whether the repair mechanism itself (violation
detection + k-NN projection) generalizes to a different schema and
still (a) reduces violations and (b) leaves TSTR utility flat/improved.
It does NOT retest the "rule-conditioning vs repair" interaction from
the weak_multimodal ablation, because there's no rule-conditioning axis
here to interact with. Keep this distinction explicit in any writeup.

CONFIDENCE LEVELS (be honest about these when citing results):
- Adult Income constraints below are HIGH confidence: education/
  education_num is a fixed, standard, well-documented 1:1 mapping in
  the UCI Adult dataset, and relationship-implies-gender for
  Husband/Wife is a matter of definition, not a data-dependent
  assumption.
- German Credit constraints below are DELIBERATELY CONSERVATIVE: this
  processed file uses the numeric Statlog encoding, and the exact
  semantics of several codes (e.g. what personal_status_sex=3 means)
  vary slightly across documented versions of this dataset online. Not
  confident enough to encode cross-field semantic rules without risking
  a WRONG constraint that silently corrupts good rows while looking
  like repair. So only range/known-category-membership checks are used
  here -- nothing that assumes a specific cross-field meaning.
"""

from __future__ import annotations
import pandas as pd
from constraint_repair import Constraint


# --------------------------------------------------------------------------
# ADULT INCOME
# --------------------------------------------------------------------------
# Standard UCI Adult education -> education_num mapping (fixed, documented).
_EDUCATION_NUM_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def _education_num_mismatch(df):
    expected = df["education"].map(_EDUCATION_NUM_MAP)
    # rows where education is an unrecognized string (e.g. "unknown") can't
    # be checked against the fixed map -- don't flag those as violations,
    # since we have no ground truth for what education_num "should" be.
    checkable = expected.notna()
    return checkable & (expected != df["education_num"])


def _fix_education_num(df):
    # Exact, known mapping -- no need to guess via nearest neighbor.
    # Rows with unrecognized education strings are left as NaN here, but
    # they're excluded from the violation mask above so they're never
    # selected for repair in the first place.
    return {"education_num": df["education"].map(_EDUCATION_NUM_MAP)}


def _husband_wrong_gender(df):
    return (df["relationship"] == "Husband") & (df["gender"] != "Male")


def _fix_husband_gender(df):
    return {"gender": pd.Series("Male", index=df.index)}


def _wife_wrong_gender(df):
    return (df["relationship"] == "Wife") & (df["gender"] != "Female")


def _fix_wife_gender(df):
    return {"gender": pd.Series("Female", index=df.index)}


def _implausible_age_adult(df):
    # UCI Adult's real age range is bounded; standard bounds are ~17-90
    return (df["age"] < 17) | (df["age"] > 90)


def _implausible_hours(df):
    # hours_per_week is bounded 1-99 in the standard dataset (99 = "99+" bucket)
    return (df["hours_per_week"] < 1) | (df["hours_per_week"] > 99)


ADULT_INCOME_CONSTRAINTS: list[Constraint] = [
    Constraint(
        id="education_education_num_mismatch",
        description="education string doesn't match its fixed education_num code (UCI standard mapping)",
        check=_education_num_mismatch,
        repair_cols=(
            "education_num",
        ),  # unused when deterministic_fix is set; kept for reporting clarity
        deterministic_fix=_fix_education_num,
    ),
    Constraint(
        id="husband_wrong_gender",
        description="relationship=Husband but gender != Male",
        check=_husband_wrong_gender,
        repair_cols=("gender",),
        deterministic_fix=_fix_husband_gender,
    ),
    Constraint(
        id="wife_wrong_gender",
        description="relationship=Wife but gender != Female",
        check=_wife_wrong_gender,
        repair_cols=("gender",),
        deterministic_fix=_fix_wife_gender,
    ),
    Constraint(
        id="implausible_age",
        description="age outside plausible [17, 90] range",
        check=_implausible_age_adult,
        repair_cols=("age",),
    ),
    Constraint(
        id="implausible_hours_per_week",
        description="hours_per_week outside plausible [1, 99] range",
        check=_implausible_hours,
        repair_cols=("hours_per_week",),
    ),
]


# --------------------------------------------------------------------------
# GERMAN CREDIT (numeric Statlog encoding -- conservative, range/category-only)
# --------------------------------------------------------------------------
_GERMAN_CATEGORY_DOMAINS = {
    "status": {1, 2, 3, 4},
    "credit_history": {0, 1, 2, 3, 4},
    "savings": {1, 2, 3, 4, 5},
    "employment_duration": {1, 2, 3, 4, 5},
    "installment_rate": {1, 2, 3, 4},
    "personal_status_sex": {1, 2, 3, 4},
    "other_debtors": {1, 2, 3},
    "present_residence": {1, 2, 3, 4},
    "property": {1, 2, 3, 4},
    "other_installment_plans": {1, 2, 3},
    "housing": {1, 2, 3},
    "number_credits": {1, 2, 3, 4},
    "job": {1, 2, 3, 4},
    "people_liable": {1, 2},
    "telephone": {1, 2},
    "foreign_worker": {1, 2},
}


def _make_category_domain_check(col: str, domain: set):
    def _check(df):
        return ~df[col].isin(domain)

    return _check


def _implausible_age_german(df):
    # UCI German Credit's real age range is 19-75; give a little slack
    # since this is the coarse robustness check, not a precision claim
    return (df["age"] < 18) | (df["age"] > 90)


def _non_positive_duration(df):
    return df["duration"] < 1


def _non_positive_amount(df):
    return df["amount"] < 1


GERMAN_CREDIT_CONSTRAINTS: list[Constraint] = [
    Constraint(
        id=f"invalid_category__{col}",
        description=f"{col} value falls outside its documented category set {sorted(domain)}",
        check=_make_category_domain_check(col, domain),
        repair_cols=(col,),
    )
    for col, domain in _GERMAN_CATEGORY_DOMAINS.items()
] + [
    Constraint(
        id="implausible_age",
        description="age outside plausible [18, 90] range",
        check=_implausible_age_german,
        repair_cols=("age",),
    ),
    Constraint(
        id="non_positive_duration",
        description="duration (months) < 1",
        check=_non_positive_duration,
        repair_cols=("duration",),
    ),
    Constraint(
        id="non_positive_amount",
        description="credit amount < 1",
        check=_non_positive_amount,
        repair_cols=("amount",),
    ),
]
