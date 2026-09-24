# Project Context: Logical Constraint Repair for Hierarchical Synthetic Tabular Data Generation

## 1. Baseline paper being extended

**Title**: "Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework"
**Authors**: Nie, Junfeng; Jin, Alvin; Chen, Xiaohui (AnyFluxion / University of Southern California)
**Venue**: PMLR 306, ICML 2026
**Repo**: https://github.com/junfengn-ctrl/hierarchical-synthetic-tabular

### What the baseline (H-TDBU) proposes
A hybrid framework combining:
- **Top-down (TD) path**: domain rules and cross-modal alignment constraints.
- **Bottom-up (BU) path**: lightweight learned generators (Gaussian copula, RandomForest, XGBoost, plus neural baselines CTGAN/TVAE) capturing local statistical patterns from real seed data.
- The two paths are combined via a synthesis engine, with a "reconciliation loop" that checks TSTR utility and cross-modal alignment (XModal) to adjust either side.

The paper's abstract explicitly names **two** goals for the top-down path: *"structure-driven logical constraints and cross-modal alignment rules."*

### What's actually implemented (confirmed by reading the real repo source, not just the paper)
- **Cross-modal alignment — IMPLEMENTED.** A rule-provider JSON (manual, or LLM/Gemini-generated) encodes a weak alignment rule between the tabular target column and attached sentiment text (e.g. target=1 pairs with positive financial-news text). This is measured via a dedicated metric, **XModal** (mean absolute difference between real and synthetic joint distributions of target and text sentiment).
- **Structure-driven logical constraints — NOT IMPLEMENTED.** No mechanism anywhere in the code checks or enforces internal logical consistency among a row's own tabular columns (independent of text). This is asserted in the abstract but never built or measured. **This is the gap this project addresses.**

### Baseline benchmarks and datasets
- **weak_multimodal** (manual rule) and **weak_multimodal_gemini** (LLM-generated rule): Bank Marketing (UCI, Moro et al. 2014, ~45K rows) + FinancialPhraseBank (Kaggle sentiment) combined via the weak alignment rule.
- **Adult Income** (UCI/Kaggle, ~48K rows) and **German Credit** (UCI/Statlog, 1,000 rows, numeric Statlog encoding): tabular-only sanity-check benchmarks, no text/rule layer at all.

### Baseline's own published headline result (reproduced by us within seed noise)
Conditional tree-based generators (RandomForest, XGBoost) match or beat neural baselines (CTGAN, TVAE) on downstream TSTR utility, at far lower compute cost; independent-column sampling performs poorly, confirming cross-column dependency preservation matters more than model capacity.

---

## 2. The research gap (this project's motivation)

**Statistical fidelity / downstream utility ≠ logical validity.** A synthetic row can score well on TSTR AUROC and fidelity metrics while combining column values that could never co-occur in a real record — because every existing metric evaluates columns individually or in aggregate, never a row's internal logical consistency.

**Canonical example (Bank Marketing schema)**: a row with `pdays = -1` ("never previously contacted") but also `previous = 3` ("contacted 3 times before") and `poutcome = success` ("a previous campaign succeeded"). Every field individually looks like a plausible value; together they describe something impossible. No existing H-TDBU metric would flag this.

**Why it matters practically**: financial ML systems (fraud detection, credit scoring) trained on logically incoherent synthetic data risk learning spurious patterns; compliance/audit-sensitive domains may need internal record coherence, not just statistical plausibility, to trust synthetic data for model development.

---

## 3. What we built (the contribution)

A **constraint-repair module**, inserted between synthesis and evaluation in the existing pipeline — strictly additive, no existing code modified or removed.

### Components
1. **Constraint definitions** — hand-written, deterministic logical rules per dataset schema (not soft statistical correlations). Each returns a boolean mask (True = violated).
2. **Violation detection** — checks every synthetic row against every constraint; produces a new metric, `violation_rate` (and per-constraint rates), matching the output style of the existing `evaluate_fidelity.py`.
3. **Repair — two strategies**:
   - **Deterministic override**: when the correct value is fully derivable from other columns in the same row (e.g. `education` → `education_num` is a fixed 1:1 UCI-standard mapping; `relationship=Husband` implies `gender=Male` by definition). Applied directly, no approximation.
   - **k-NN nearest-real-neighbor projection**: when the correct value isn't derivable from the row itself. Finds the most similar real row (via `scikit-learn` `NearestNeighbors` over ordinal-encoded categoricals + standardized numerics) and copies **only** the implicated column(s) — minimally invasive, not a full-row replacement.
4. **Iterative convergence** — constraints can share columns (e.g. `previous`/`poutcome` appear in three Bank Marketing constraints), so fixing one can transiently re-violate another; repair re-runs the full constraint sweep (default `max_passes=3`) until no violations remain or the pass limit is hit.

### Design rationale (why not alternatives)
- **Why repair, not discard**: discarding wastes generation cost and biases the remaining sample; for high-violation methods (independent sampling: 40–88% initial violation rate depending on dataset) discarding would gut most of the dataset.
- **Why insert repair as a separate stage, not inside the generator**: keeps it generator-agnostic (works identically regardless of upstream method) and keeps the ablation clean — same repair code toggled on/off around any generator.
- **Why deterministic fix as a separate path from k-NN**: k-NN is an *approximation* even for rules with a known exact answer, and (as the correction below shows) that imprecision can silently inflate results.

### Constraint sets built, per dataset
- **Bank Marketing** (6 constraints): `pdays`/`previous`/`poutcome` temporal consistency (3 rules), plausible age range, non-negative call duration, positive campaign-contact count. All k-NN repaired (no deterministic rule available for this schema).
- **Adult Income** (5 constraints): `education`↔`education_num` mismatch (deterministic fix — verified against real audit sample data), `Husband`/`Wife` relationship↔`gender` (deterministic fix), plausible age range, plausible hours-per-week range.
- **German Credit** (deliberately conservative — see caveat below): category-domain-membership checks for 16 coded columns (verified against real data), plus plausible age/duration/amount ranges. No cross-field semantic rules, due to uncertainty about the exact Statlog numeric codebook.

### Verification discipline applied throughout
- Constraint category values and mappings (e.g. `education_num` codes, German Credit category domains) were checked against **real sample rows from the actual processed dataset** (via an audit script run against the cloned repo), not assumed from memory or documentation.
- The repair module was smoke-tested against both ordinary and deliberately worst-case synthetic scenarios before being trusted on real experiments.
- A real implementation bug was caught mid-project (see Section 5) and corrected before finalizing results.

---

## 4. Experimental design

### Ablation logic
Three/four-way comparison, isolating each mechanism's individual contribution:
| Arm | What it is |
|---|---|
| Rule-conditioning only | RandomForest / XGBoost (existing, conditional generators paired with the existing rule provider), no repair |
| Repair only | Independent-column sampling (existing method with zero conditioning — stands in for "no rule-conditioning") + repair |
| Both | RandomForest / XGBoost + repair |

### Generators used (of the 6 available in the repo)
Used: `independent`, `random_forest`, `xgboost`.
Not used: `gaussian_copula`, `ctgan`, `tvae` — deliberately out of scope, since the paper's own reproduced finding showed RF/XGBoost already matching or beating the neural baselines at lower cost; testing repair against the strongest validated options was the higher-value use of limited scope.

### Protocol
- 3 seeds per (dataset, method, repair on/off): 42, 123, 2024 — matches the baseline paper's own seeds for direct comparability.
- Metrics logged: violation rate (new), TSTR/TRTR accuracy/F1/AUROC (existing), XModal (existing, weak-multimodal only), fidelity (existing).
- CPU-only throughout — no GPU used or required. Repair itself does zero machine learning (no training): violation checks are vectorized boolean operations over columns; k-NN repair uses a pre-built index (not a trained model) and only runs on the violating row subset, not the full dataset; deterministic fixes are dictionary lookups. Full repair pass over ~12,000 rows runs in well under a second on a laptop CPU.

### Datasets tested
Bank Marketing weak-multimodal (manual + Gemini rule variants), Adult Income, German Credit — the same four benchmark files used in the original paper, enabling direct comparison to its published numbers.

---

## 5. Results

### Phase 1 — Baseline reproduction (validation step, before building anything)
Reproduced the paper's own Tables 1–3 within seed noise across both weak-multimodal benchmarks and the XGBoost ablation (training rows / conditioning columns), confirming the baseline was trustworthy before extending it.

### Phase 3 — Main ablation (Bank Marketing weak-multimodal)
| Arm | Violation rate | TSTR AUROC |
|---|---|---|
| Rule-conditioning only (RF / XGB) | ~11–14% | 0.919 / 0.925 |
| Repair only (independent + repair) | ~40% → 6.5% (incomplete convergence at the time) | 0.489 → 0.587 |
| Both (RF / XGB + repair) | ~11–14% → **~0%** | 0.919 / 0.925 (unchanged, within noise) |

Key findings:
- Repair eliminates nearly all logical violations for the already-structured generators, at negligible utility cost (deltas within seed noise).
- **XModal is byte-identical before and after repair, for every method** — proof repair and cross-modal alignment are genuinely non-interfering mechanisms (by construction: repair constraints touch only tabular columns, never `target`/`text_sentiment`).
- Repair-only recovers some utility over no-conditioning-no-repair, but stays far below what rule-conditioning achieves alone.

### Phase 3b — Robustness check (Adult Income, German Credit)
Confirmed the "repair is free" pattern (violations → ~0%, utility flat) generalizes to two structurally different schemas — not a Bank-Marketing-specific artifact.

Honest caveat: German Credit's `independent` arm showed 0% violations even *before* repair — not because the generator was perfect, but because the deliberately conservative constraint set (category-domain checks only, due to codebook uncertainty) never gets violated by marginal-sampling methods. This dataset provides a weaker robustness test than the other two, and this is stated explicitly rather than left implicit.

### Phase 6 — The correction (important, and the strongest evidence of rigor in the project)
**First run** (Adult Income, `independent + repair`, using k-NN for all constraints including `education`/`education_num`, which is actually a known deterministic rule): violation rate 84.4% → 29.0% (incomplete), TSTR AUROC 0.567 → **0.662** (+0.095 gain).

**Investigation**: the incomplete convergence pointed to a design flaw — a fully deterministic rule was being approximated via k-NN instead of applied exactly.

**Fix**: added a `deterministic_fix` capability to the `Constraint` class — for rules where the correct value is exactly derivable from the row itself, repair applies the known formula directly instead of nearest-neighbor lookup. Applied to `education`/`education_num` and `Husband`/`Wife`-`gender` constraints. Stress-tested against a worse-than-real corruption scenario (88.5% violation rate) — resolved to exactly 0%, with every row verified exactly correct, not approximately correct.

**Re-run with fix**: violation rate 84.4% → **0.0%** (fully resolved). TSTR AUROC 0.567 → **0.570** (+0.004 — essentially flat, within noise).

**Interpretation**: the original +0.095 gain was an artifact. The imprecise k-NN repair wasn't just fixing the one flagged inconsistency — it was pulling in a whole real neighbor row's value, incidentally leaking that neighbor's broader real-data coherence into the row as a side effect. Once repair became exact and minimal (touching only what the constraint logically required), that side-effect gain disappeared.

**Consequence for the project's claims**: the finding "repair alone recovers meaningful utility" is **withdrawn** as an established result. The more conservative, verified finding stands: **precise repair does not reliably improve downstream utility on its own** — its value is specifically and only in enforcing logical validity, at zero cost, without interfering with rule-conditioning's utility contribution. This is treated as a strength of the project's process (catching and correcting an inflated result before finalizing), not a weakness of the finding.

---

## 6. Current honest thesis (post-correction)

> Logical validity and statistical/semantic utility are two separable problems in synthetic tabular data generation, requiring two separable mechanisms. Rule-conditioning (already existing in H-TDBU) drives downstream utility; repair (this project's contribution) drives logical validity. Neither substitutes for the other. Combining them costs nothing — repair does not measurably harm utility or cross-modal alignment when layered on top of an already-good generator. Repair alone, even when made maximally precise, does not reliably recover meaningful utility on its own.

This is a narrower claim than the pre-correction version, and is presented as such — the corrected, smaller finding, not the earlier inflated one.

---

## 7. Known limitations (to state proactively, not wait to be asked)
- Constraint sets are hand-authored per schema, not automatically derived — doesn't scale to new datasets without manual constraint-writing effort.
- German Credit's constraint set is deliberately conservative (category/range checks only) due to uncertainty about the exact Statlog numeric codebook — a weaker robustness test than Adult Income or Bank Marketing.
- Bank Marketing's constraints are still entirely k-NN-based (no deterministic rule available for that schema) — so the same imprecision-inflation risk identified and fixed for Adult Income has not been separately verified as absent there; this is an open question, not yet checked.
- Repair has only been tested downstream of tree-based/statistical generators (independent sampling, RandomForest, XGBoost) — not tested after LLM-based autoregressive generators (GReaT, TabuLa-style), which the literature notes have a *weaker* built-in guarantee of row-level structural consistency than conditional tree-based generation, making this a plausible and higher-value next test, not a completed one.

---

## 8. Literature landscape (for the lit review section)

| Line of work | Examples | Approach | Limitation relevant to this project |
|---|---|---|---|
| Classical generative tabular models | CTGAN, TVAE (Xu et al. 2019; Zhao et al. 2023) | GAN/VAE-learned distribution, sampled | Mode collapse; miss rare/tail events; no logical-constraint awareness |
| Tabular diffusion models | STaSy, SOS, TabDDPM | Iterative denoising | Same statistical-only focus; heavier compute |
| LLM-based tabular synthesis | GReaT, REaLTabFormer, TabuLa | Row-as-text, autoregressive generation via LLM in-context learning | Autoregressive generation doesn't guarantee structural consistency or functional dependencies between columns (explicitly noted in H-TDBU's own related-work framing) |
| This project's baseline | H-TDBU (Nie, Jin & Chen, ICML 2026) | Hybrid top-down rules + bottom-up tree/neural generators | Names "logical constraints" as an objective, implements only cross-modal alignment |
| Adjacent, uncited field | Constraint-based data cleaning / data repair literature | Detect and repair records violating known integrity constraints (functional dependencies, denial constraints) | This project borrows its detect-and-repair framing from here, applied to synthetic rather than dirty real data |

---

## 9. Working title for the paper

**Primary**: *"Disentangling Logical Validity from Statistical Utility in Synthetic Tabular Data: A Constraint-Repair Extension to Hierarchical Generation"*

Chosen because it accurately reflects the corrected finding — it claims the project *tested and separated* these two properties, not that repair *improves* utility (which the corrected results do not support). Avoids "Improving" or "Novel" in the title for the same honesty reason.

Alternatives considered:
- "Logical Constraint Repair for Hierarchical Synthetic Tabular Data Generation: An Empirical Ablation Study" (safer, more conventional)
- "Is Logical Repair Necessary Alongside Rule-Conditioned Synthesis? An Ablation Study on Hierarchical Tabular Generation" (question-framed)

---

## 10. Project status (as of the presentation)

| Stage | Status |
|---|---|
| Phase 0 — repo setup, environment | Done |
| Phase 1 — baseline reproduction (matches paper's Tables 1–3) | Done |
| Phase 2 — repo audit (confirmed real function signatures, CSV-based I/O convention, real schemas) | Done |
| Phase 3 — repair module + violation metric (Bank Marketing) | Done |
| Phase 4 — main 3-way ablation (weak-multimodal) | Done |
| Phase 5 — robustness check (Adult Income, German Credit) | Done |
| Phase 6 — bug found, deterministic-fix correction, re-verified | Done |
| Presentation (slides, code walkthrough, Q&A prep) | Ready |
| **Written paper draft** | **Not started — next deliverable** |

All experimental work is complete and verified. The manuscript (abstract, full lit review, methodology writeup, results section with figures, discussion, limitations, conclusion) has not yet been drafted.

---

## 11. Implementation file inventory (for reference)

- `audit_repo.py` — dumps repo source, config, and processed-data schema for verifying integration points against real code rather than assumption.
- `constraint_repair.py` — core module: `Constraint` dataclass (with `check`, `repair_cols`, optional `deterministic_fix`), `compute_violation_report()`, `repair_dataframe()`, `repair_csv()` (CSV-path wrapper matching the repo's I/O convention), Bank Marketing constraint set.
- `adult_german_constraints.py` — Adult Income and German Credit constraint sets, with confidence levels stated explicitly per dataset.
- `run_repair_ablation.py` — drop-in extension of the repo's real `run_experiments.py`; runs the rule-conditioning × repair ablation grid, logging violation rate alongside existing fidelity/utility metrics.
- `robustness_check.py` — parallel runner for the Adult Income / German Credit generalization test.
