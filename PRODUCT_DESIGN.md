# Product Design (quick) — Verity: a constraint-aware synthetic data platform

*Working name: **Verity**. Authors: Nandini Saxena, Snigdha Sarkar. Built on the constraint-repair research (extension of H-TDBU).*

## 1. Problem & opportunity
Synthetic-data tools (SDV, Gretel, Mostly AI, Tonic) all sell **fidelity + privacy**. None of them:
- guarantee that individual rows are **logically valid** (no impossible records), or
- warn that a naive "repair/impute" step can **leak real data** into supposedly-synthetic output.

Our research proves both problems are real and fixable. That is the wedge.

## 2. Vision
> Point Verity at a real dataset and say what the data is *for*. It **learns the business rules**, generates synthetic data with the right generator, **repairs it to be logically valid**, and tunes everything to that purpose — with validity and privacy as guarantees, not afterthoughts. Output: valid synthetic data plus a trust report scoped to the purpose.

## 3. Differentiators (the moat)
1. **Validity guarantee** — "0% violations of the approved rule set," a claim competitors can't make.
2. **Leakage-safe repair** — our k-NN-vs-canonical finding becomes a safety feature: for privacy use, Verity forbids the data-borrowing (k-NN) repair and flags it as leaky. Research-backed, unique.
3. **Purpose-aware** — the same data is generated differently for sharing vs. training vs. testing.

## 4. Architecture (5 layers)
```
Input: real data + schema + column descriptions + PURPOSE
  1. RULE INTELLIGENCE   auto-discover + LLM-elicit + human-approve → versioned rule set
  2. PURPOSE PROFILE     picks generator, repair mode, metrics, privacy budget
  3. GENERATOR           independent / RF / XGBoost  (pluggable: CTGAN, LLM)
  4. REPAIR ENGINE       deterministic / canonical / k-NN   [already built]
  5. EVALUATION          validity + fidelity + utility + privacy → report card
Output: valid synthetic data + trust report
```
Layers 3–5 already exist from the research. The product work is **layers 1–2**.

## 5. The intelligence layer (how it "knows business rules")
Three mechanisms, least-to-most human involvement:
1. **Automatic discovery** — mine functional dependencies (unique repair), category domains, plausible ranges, and denial-constraint candidates (forbidden combinations / sentinel-value rules) directly from the data. *Proposes, never decides.*
2. **LLM-assisted elicitation** — give an LLM the schema + column descriptions + a sample; it proposes human-readable semantic rules that pure data-mining misses.
3. **Human-in-the-loop approval** — a reviewer accepts/edits/rejects candidates; approved rules become a **versioned rule set** reusable across datasets of the same schema.

Combining all three is what makes it "intelligent enough" without pretending to be safely fully-automatic (it isn't).

## 6. Purpose profiles (where the leakage finding pays off)
| Purpose | Generator | Repair mode | Rationale |
|---|---|---|---|
| Privacy-safe sharing | RF/XGBoost | **canonical / deterministic only** | k-NN would leak real values — forbidden here |
| ML training / augmentation | RF/XGBoost + rule-conditioning | any | maximize downstream utility (TSTR) |
| Software / QA testing | any | hard-enforce validity | valid edge-case coverage |
| Benchmarking | controlled | deterministic | reproducible properties |

## 7. MVP scope (priority order)
1. **Rule discovery + review** — auto-mine candidate constraints from a dataset; human approves. *(prototype now)*
2. **LLM elicitation** — plug an LLM into the same review flow for semantic rules.
3. **Purpose profiles** — a config layer over the existing generator + repair.
4. **Privacy check** — detect/flag k-NN leakage and copied-record risk.
5. Later: FD/DC mining at scale, pluggable neural/LLM generators, full report card.

## 8. What exists vs. new
- **Exists:** repair engine (3 modes), violation metric, generators (independent/RF/XGBoost), evaluation, showcase.
- **New:** rule-discovery, LLM elicitation, purpose profiles, privacy check, review UI.

## 9. Risks (be honest)
- Auto rule-discovery over-produces spurious rules and doesn't scale trivially → keep a human gate; sell it as *assisted*.
- "No impossible rows" is only true for the **approved** rule set → state that precisely.
- Fully automatic rule-knowing is not safely achievable today → MVP centers on discovery + approval, not autonomy.

## 10. What the prototype proves
That the "hard part" is tractable: **Verity can automatically rediscover the constraints we wrote by hand** (e.g. the exact `education → education_num` dependency, the `pdays = -1` sentinel rule, category domains, plausible ranges) from the raw data — turning hand-authored rules into a learned, reviewable rule set.
