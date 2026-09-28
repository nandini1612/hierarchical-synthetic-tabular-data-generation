# Verity — validity-guaranteed synthetic tabular data

Verity is the product built on this repository's research. Point it at any tabular
CSV and it learns the data's distributions **and its business rules**, generates
realistic synthetic rows, **repairs every logically impossible record to valid**,
and reports the four things that matter: **validity, fidelity, privacy, and utility**.

It is the applied form of the paper's finding that logical validity and statistical
utility are separable — so Verity guarantees validity as a first-class property that
mainstream generators (SDV, CTGAN, etc.) do not, and it treats the leakage finding as
a privacy safety feature.

---

## Install

```bash
python -m pip install -r requirements.txt
```

Core (`pandas numpy scikit-learn scipy sdv flask`) is required. `anthropic` and
`groq` are optional — only for the AI rule features. Everything else is CPU-only.

## Quick start

**Command line — generate from any CSV:**
```bash
python src/verity.py --input your_data.csv --rows 5000
```

**Web app — upload, review rules, generate, download:**
```bash
python src/verity_app.py            # then open http://127.0.0.1:5000
```

Both produce a synthetic CSV plus a quality report (JSON, and HTML with `--html`).

---

## The quality report

Every run reports four axes:

| Axis | What it means | How it's measured |
|---|---|---|
| **Validity** | no logically impossible rows | share of rows violating any enforced rule → repaired to 0% |
| **Fidelity** | looks like the real data | KS (numeric) + total-variation (categorical) + correlation match, scored /100 |
| **Privacy** | discloses no individual | exact-copy rate + distance-to-closest-record audit (privacy purpose) |
| **Utility** | useful to train on | TSTR: train a model on synthetic, test on real, vs a real-trained baseline |

Example:
```
Fidelity   [##########] 98/100
Validity   [##########] 100%   (16 rules enforced; 4016 impossible rows repaired)
Privacy    exact copies of real records: 0.00%
Utility    TSTR AUROC 0.829  vs real 0.906   (target: income)
```

---

## How it works (pipeline)

```
Real CSV → learn rules (mine + optional AI + your typed rules)
         → generate (copula | TVAE | CTGAN)
         → repair every row to logically valid
         → reproduce real missing-value rates
         → report validity / fidelity / privacy / utility
         → synthetic CSV + report
```

Repair is rule-driven (canonical), so it never copies real rows — validity is
guaranteed on top of *any* generator, and it's leakage-safe by construction.

---

## CLI reference

```bash
python src/verity.py --input data.csv [options]
```

| Option | Default | Purpose |
|---|---|---|
| `--input PATH` | (required) | the real CSV to learn from |
| `--rows N` | 5000 | how many synthetic rows to generate |
| `--output PATH` | `<input>_synthetic.csv` | where to write the synthetic CSV |
| `--exclude a,b` | — | drop columns (IDs, free text) before learning |
| `--purpose P` | balanced | `balanced` \| `ml_training` \| `privacy` \| `testing` (see below) |
| `--generator G` | auto | `auto` (match purpose) \| `copula` \| `tvae` \| `ctgan` |
| `--epochs N` | 50 | training epochs for tvae/ctgan |
| `--target COL` | auto-detect | label column for the utility (TSTR) check |
| `--rules-file F` | — | a text file of business rules in plain English (needs an AI key) |
| `--llm` | off | also ask an AI for semantic rules (needs an AI key) |
| `--local-only` | off | **guarantee no data leaves this machine** (disables all AI) |
| `--descriptions-only` | off | if using AI, send only column names/types — never real rows |
| `--no-missing` | off | do not reproduce the real data's missing-value rates |
| `--min-confidence F` | 1.0 | only enforce mined rules at/above this confidence |
| `--html` | off | also write a shareable HTML quality report |

---

## Purpose profiles

The **purpose** tunes the rule threshold, the generator, and the quality gate:

| Purpose | Generator | Leads on | Special behaviour |
|---|---|---|---|
| `balanced` | copula | fidelity | general-purpose |
| `ml_training` | **TVAE** | fidelity/utility | stronger generator for downstream models |
| `privacy` | copula | privacy | **forbids real-data-copying repair; gates exact-copy=0; audits distance-to-closest-record; fails if near-duplicate rate > 2%** |
| `testing` | copula | validity | lower rule threshold → enforce more rules |

The privacy profile is where the research pays off: it turns the leakage finding into
an enforced guarantee and reports a PASS/REVIEW verdict with per-check results.

## Generators

| Backend | Speed | When |
|---|---|---|
| `copula` (Gaussian copula) | fast, CPU | default; preserves marginals + correlations |
| `tvae` / `ctgan` (SDV neural) | slower (minutes) | complex, non-linear, heavy-tailed data |

Repair guarantees validity regardless of backend, so a stronger generator only
raises fidelity/utility — it can never produce invalid output.

---

## Rule intelligence

Verity gets its business rules three ways (all optional except the first):

1. **Mining (always on, 100% local).** Discovers functional dependencies, sentinel
   rules (`pdays=-1 ⇒ no prior contact`), denial constraints, category domains, and
   ranges directly from the data. Run it standalone: `python src/rule_discovery.py`.
2. **AI elicitation (`--llm`).** An LLM proposes *semantic* rules mining can miss.
   **Every proposal is validated against your data**, so hallucinations are dropped.
3. **Plain-language rules (`--rules-file`, or the web textarea).** Type rules your
   domain knows ("discharge date ≥ admission date"); the AI converts them to enforced
   constraints, each checked against your data.

### Setting up the AI (optional)
```bash
# Groq (free tier):
pip install groq
export GROQ_API_KEY=...            # Windows: setx GROQ_API_KEY ...
# Anthropic:
pip install anthropic
export ANTHROPIC_API_KEY=...
```
Provider is auto-detected (Groq preferred). Override the model with
`VERITY_LLM_MODEL` (Groq default `openai/gpt-oss-120b`) or force a provider with
`VERITY_LLM_PROVIDER=groq|anthropic`.

---

## Data privacy — where your data goes

- **Generation is always 100% local.** The copula/TVAE/CTGAN training and repair run
  entirely on your machine; the real CSV never leaves it.
- **The only exception is the optional AI features.** With `--llm` or plain-language
  rules, Verity sends the schema **and a small sample of real rows** to the AI provider.
- Two controls make this safe for sensitive data:
  - **`--local-only`** — hard guarantee that *nothing* leaves the machine (all AI off).
    Mining still discovers most rules with no network.
  - **`--descriptions-only`** — if you do use AI, send only column names and types,
    never real rows or values.

For finance/healthcare data, run `--local-only`, or `--descriptions-only` if you want
AI help without exposing records.

---

## File map

| File | Role |
|---|---|
| `src/verity.py` | the engine + CLI: generators, repair, quality report, utility |
| `src/verity_app.py` | the Flask web app |
| `src/rule_discovery.py` | rule mining + AI elicitation + plain-language parsing |
| `src/purpose_profiles.py` | purpose profiles + the privacy disclosure audit |
| `src/constraint_repair.py` | the original research repair engine (Bank Marketing) |
| `PRODUCT_DESIGN.md` | the product design / architecture |
| `README.md` | the underlying research |

## Honest limitations

- "100% valid" means valid against the **rules Verity discovered or you approved** —
  not a guarantee of every unstated domain rule. Review the rules.
- The copula default is strong on tidy data; on complex, heavy-tailed data prefer
  `--generator tvae` (slower).
- Neural backends need enough epochs to reach high fidelity, and shine on complex
  data, not tidy benchmarks.
- Privacy is empirical (exact-copy + distance-to-closest-record), not yet a formal
  differential-privacy guarantee.
- Validated on public benchmarks; a real regulated-domain dataset should be tested
  before production use.
