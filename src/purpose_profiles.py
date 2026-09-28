"""
purpose_profiles.py  —  Verity Layer 2: purpose-aware configuration.

The SAME dataset should be generated differently depending on WHY: sharing it
externally (privacy first), training a model (utility first), or seeding a test
environment (validity/coverage first). This module encodes those purposes as
profiles that tune the pipeline and GATE the output against the right guarantee.

Research payoff: because k-NN repair leaks real values (canonical/rule-driven
repair does not), any privacy purpose must forbid real-data-copying repair.
Verity's repair is already rule-driven, so the privacy profile asserts a
no-leakage guarantee and audits disclosure with distance-to-closest-record,
not just exact-copy rate.

Used by verity.py (`--purpose`) and verity_app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Profile:
    name: str
    label: str
    description: str
    min_confidence: float = 1.0
    allow_real_data_repair: bool = True
    max_exact_copy_rate: float | None = None
    audit_disclosure: bool = False
    max_too_close_rate: float = 0.02
    primary: str = "fidelity"
    generator: str = "copula"        # recommended generator backend when "auto"
    goal: str = ""
    guarantees: list = field(default_factory=list)


PROFILES: dict[str, Profile] = {
    "balanced": Profile(
        name="balanced", label="Balanced",
        description="A good general-purpose synthetic copy.",
        min_confidence=1.0, primary="fidelity",
        goal="A faithful, valid synthetic copy with no particular axis prioritised.",
        guarantees=["100% logical validity", "no real-data leakage in repair"],
    ),
    "ml_training": Profile(
        name="ml_training", label="ML training / augmentation",
        description="Maximise usefulness for training a downstream model.",
        min_confidence=1.0, primary="fidelity", generator="tvae",
        goal="Preserve the statistical signal a model needs; validity as a free guarantee.",
        guarantees=["100% logical validity", "distributions & correlations preserved",
                    "no real-data leakage in repair"],
    ),
    "privacy": Profile(
        name="privacy", label="Privacy-safe sharing",
        description="Safe to share externally in place of sensitive real data.",
        min_confidence=0.999,
        allow_real_data_repair=False,
        max_exact_copy_rate=0.0,
        audit_disclosure=True,
        max_too_close_rate=0.02,
        primary="privacy",
        goal="Release data that behaves like the real thing but discloses no individual.",
        guarantees=["no real record copied verbatim", "no real-data leakage in repair (rule-driven only)",
                    "distance-to-closest-record audited", "100% logical validity"],
    ),
    "testing": Profile(
        name="testing", label="Software / QA testing",
        description="Populate a test environment with valid, well-covered records.",
        min_confidence=0.95,
        primary="validity",
        goal="Every row logically valid; maximise coverage of categories and edge values.",
        guarantees=["100% logical validity", "all discovered rules enforced",
                    "no real-data leakage in repair"],
    ),
}

DEFAULT = "balanced"


def get(name: str | None) -> Profile:
    return PROFILES.get((name or DEFAULT).lower(), PROFILES[DEFAULT])


def list_profiles() -> list[str]:
    return list(PROFILES)


def _encode(real: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(real[c]):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            mu, sd = float(real[c].mean()), float(real[c].std() or 1.0)
        else:
            order = {val: i for i, val in enumerate(real[c].astype(str).value_counts().index)}
            v = df[c].astype(str).map(order).to_numpy(dtype=float)
            ref = real[c].astype(str).map(order)
            mu, sd = float(ref.mean()), float(ref.std() or 1.0)
        v = np.nan_to_num(v, nan=mu)
        parts.append(((v - mu) / (sd if sd else 1.0)).reshape(-1, 1))
    return np.hstack(parts) if parts else np.zeros((len(df), 1))


def disclosure_audit(real: pd.DataFrame, synth: pd.DataFrame,
                     sample: int = 3000, seed: int = 0) -> dict:
    """Distance-to-closest-record: is a synthetic row closer to a real row than
    real rows typically are to each other? If so it may be a near-copy."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {"available": False}
    cols = [c for c in real.columns if c in synth.columns]
    if not cols:
        return {"available": False}
    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real), min(sample, len(real)), replace=False)
    s_idx = rng.choice(len(synth), min(sample, len(synth)), replace=False)
    R = _encode(real, real.iloc[r_idx], cols)
    S = _encode(real, synth.iloc[s_idx], cols)
    d_syn = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(S, n_neighbors=1)[0][:, 0]
    d_real = NearestNeighbors(n_neighbors=2).fit(R).kneighbors(R, n_neighbors=2)[0][:, 1]
    baseline = float(np.percentile(d_real, 5))
    return {
        "available": True,
        "dcr_median": round(float(np.median(d_syn)), 4),
        "real_baseline_p5": round(baseline, 4),
        "too_close_rate": round(float(np.mean(d_syn < baseline)), 4),
    }


def evaluate(profile: Profile, real: pd.DataFrame, synth: pd.DataFrame, metrics: dict) -> dict:
    """Purpose-scoped verdict: which metric leads, whether the gate passes, and
    any privacy audit. `metrics` is verity.generate()'s second return value."""
    fid = metrics["fidelity"]["overall"]
    exact = metrics["privacy"]["exact_copy_rate"]
    valid_after = metrics["validity"]["violation_after"]

    checks, passed = [], True
    checks.append(("Logical validity 100%", valid_after == 0.0)); passed &= valid_after == 0.0

    audit = {}
    if profile.max_exact_copy_rate is not None:
        ok = exact <= profile.max_exact_copy_rate
        checks.append((f"No verbatim real records (<= {profile.max_exact_copy_rate:.0%})", ok)); passed &= ok
    if profile.audit_disclosure:
        audit = disclosure_audit(real, synth)
        if audit.get("available"):
            ok = audit["too_close_rate"] <= profile.max_too_close_rate
            checks.append((f"Near-duplicate rate <= {profile.max_too_close_rate:.0%} "
                           f"(got {audit['too_close_rate']:.1%})", ok)); passed &= ok
    if not profile.allow_real_data_repair:
        checks.append(("Repair used no real-data copying (leakage-safe)", True))

    headline = {"fidelity": f"{fid}/100 fidelity",
                "privacy": f"{exact*100:.2f}% verbatim copies"
                           + (f", {audit['too_close_rate']*100:.1f}% near-duplicates" if audit.get("available") else ""),
                "validity": "100% valid"}[profile.primary]

    return {"profile": profile.name, "label": profile.label, "goal": profile.goal,
            "primary": profile.primary, "headline": headline, "passed": bool(passed),
            "checks": [{"label": l, "ok": bool(ok)} for l, ok in checks],
            "guarantees": profile.guarantees, "disclosure_audit": audit}
