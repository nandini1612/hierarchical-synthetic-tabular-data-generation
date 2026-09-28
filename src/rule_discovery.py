"""
rule_discovery.py  —  Verity prototype: the "rule intelligence" layer.

Automatically mines candidate business rules (constraints) from a raw dataset,
so a human can review/approve them instead of hand-authoring every rule. This
is Layer 1 (mechanism 1: automatic discovery) of the product design; it also
builds the prompt that Layer 1 / mechanism 2 (LLM elicitation) would send.

Proof-of-concept goal: rediscover, straight from the data, the same kinds of
rules we hand-wrote in constraint_repair.py and adult_german_constraints.py —
functional dependencies (education -> education_num), sentinel-value rules
(pdays == -1 implies previous == 0 and poutcome == "unknown"), denial
constraints (poutcome == "success" requires previous >= 1), category domains,
and plausible numeric ranges.

Pure pandas/numpy, CPU-only, no training. Run directly:

    python src/rule_discovery.py

Outputs a human-readable review report and writes candidate rules to
data/processed/discovered_rules/<dataset>_rules.json for approval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "processed" / "discovered_rules"


# --------------------------------------------------------------------------
@dataclass
class CandidateRule:
    """A machine-readable candidate constraint for human review."""
    id: str
    kind: str              # fd | sentinel_implies | denial | category_domain | range | compare
    description: str       # human-readable
    columns: list          # columns involved
    spec: dict             # machine form (enough to generate a check)
    support: int           # rows the rule is based on
    confidence: float      # 1.0 = holds on every applicable row
    approved: bool = field(default=False)
    source: str = field(default="mining")   # mining | llm


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


# --------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame) -> dict:
    prof = {}
    for c in df.columns:
        s = df[c]
        col = {"dtype": "numeric" if _is_numeric(s) else "categorical",
               "n_unique": int(s.nunique(dropna=True)),
               "null_rate": float(s.isna().mean())}
        if col["dtype"] == "numeric":
            col.update(min=float(s.min()), max=float(s.max()),
                       mean=float(s.mean()))
        else:
            vc = s.value_counts()
            col["domain"] = [str(v) for v in vc.index[:25]]
        prof[c] = col
    return prof


def discover_functional_dependencies(df, max_determinant_card=80, min_conf=0.999):
    """A -> B when B is (almost) constant within each group of A."""
    rules = []
    cols = list(df.columns)
    for a in cols:
        if df[a].nunique(dropna=True) > max_determinant_card:
            continue
        g = df.groupby(a, observed=True)
        sizes = g.size()
        for b in cols:
            if a == b:
                continue
            # fraction of rows whose A-group maps to a single B value
            single = g[b].nunique(dropna=True) == 1
            conf = float(sizes[single].sum() / len(df))
            if conf >= min_conf and df[b].nunique(dropna=True) > 1:
                mapping = (df.dropna(subset=[a, b]).groupby(a, observed=True)[b]
                           .agg(lambda x: x.mode().iloc[0]).to_dict())
                rules.append(CandidateRule(
                    id=f"fd__{a}__to__{b}", kind="fd",
                    description=f"{a} determines {b} (functional dependency) — "
                                f"correct {b} is fixed by {a}.",
                    columns=[a, b],
                    spec={"determinant": a, "dependent": b,
                          "mapping": {str(k): (int(v) if isinstance(v, (np.integer,)) else v)
                                      for k, v in list(mapping.items())[:200]}},
                    support=len(df), confidence=round(conf, 5)))
    return rules


def discover_category_domains(df, max_card=25):
    rules = []
    for c in df.columns:
        if _is_numeric(df[c]):
            continue
        vals = df[c].dropna().value_counts().index.tolist()  # ordered by frequency (mode first)
        if 1 < len(vals) <= max_card:
            rules.append(CandidateRule(
                id=f"domain__{c}", kind="category_domain",
                description=f"{c} must be one of {len(vals)} known categories.",
                columns=[c], spec={"column": c, "allowed": [str(v) for v in vals]},
                support=len(df), confidence=1.0))
    return rules


def discover_ranges(df, sentinels):
    rules = []
    for c in df.columns:
        if not _is_numeric(df[c]):
            continue
        s = df[c]
        sval = sentinels.get(c)
        core = s[s != sval] if sval is not None else s
        lo, hi = float(core.min()), float(core.max())
        rules.append(CandidateRule(
            id=f"range__{c}", kind="range",
            description=f"{c} lies in [{lo:g}, {hi:g}]"
                        + (f" (excluding sentinel {sval:g})" if sval is not None else "") + ".",
            columns=[c], spec={"column": c, "min": lo, "max": hi, "sentinel": sval},
            support=len(df), confidence=1.0))
    return rules


def detect_sentinels(df, min_share=0.10):
    """A sentinel is a boundary value (the column min) that is a large spike
    AND sits apart from the rest of the distribution — e.g. pdays == -1 meaning
    'not applicable'. Separation is judged two ways: an out-of-range value
    (negative while the rest are non-negative), or a Tukey low-side outlier
    relative to the non-sentinel values. This catches pdays=-1 while rejecting
    ordinary minimums like campaign=1."""
    sentinels = {}
    for c in df.columns:
        if not _is_numeric(df[c]):
            continue
        s = df[c]
        mn = float(s.min())
        share = float((s == mn).mean())
        others = s[s != mn]
        if share < min_share or len(others) == 0:
            continue
        omin = float(others.min())
        q1, q3 = float(others.quantile(0.25)), float(others.quantile(0.75))
        iqr = q3 - q1
        out_of_range = mn < 0 <= omin
        tukey_low = iqr > 0 and mn < q1 - 1.5 * iqr
        if out_of_range or tukey_low:
            sentinels[c] = mn
    return sentinels


def discover_sentinel_implications(df, sentinels, min_conf=0.999):
    """When col == sentinel, which other columns become (almost) constant?"""
    rules = []
    for c, sval in sentinels.items():
        sub = df[df[c] == sval]
        if len(sub) < 20:
            continue
        for other in df.columns:
            if other == c:
                continue
            vc = sub[other].value_counts(normalize=True, dropna=True)
            if len(vc) == 0:
                continue
            top_val, top_frac = vc.index[0], float(vc.iloc[0])
            if isinstance(top_val, (np.integer,)):
                top_val = int(top_val)
            elif isinstance(top_val, (np.floating,)):
                top_val = float(top_val)
            disp = f"'{top_val}'" if isinstance(top_val, str) else f"{top_val}"
            if top_frac >= min_conf and df[other].nunique(dropna=True) > 1:
                rules.append(CandidateRule(
                    id=f"sentinel__{c}_{sval:g}__implies__{other}", kind="sentinel_implies",
                    description=f"{c} == {sval:g} implies {other} == {disp} "
                                f"(a 'not-applicable' sentinel rule).",
                    columns=[c, other],
                    spec={"if_column": c, "if_value": (int(sval) if float(sval).is_integer() else sval),
                          "then_column": other,
                          "then_value": (int(top_val) if isinstance(top_val, (np.integer,)) else
                                         (str(top_val) if not isinstance(top_val, (int, float)) else top_val))},
                    support=int(len(sub)), confidence=round(top_frac, 5)))
    return rules


def discover_positivity_denials(df, min_support=30, min_conf=0.999):
    """categorical value V requires numeric column N to be non-zero/positive —
    e.g. poutcome == 'success' requires previous >= 1."""
    rules = []
    cat_cols = [c for c in df.columns if not _is_numeric(df[c]) and df[c].nunique() <= 25]
    num_cols = [c for c in df.columns if _is_numeric(df[c]) and (df[c] == 0).any()]
    for cc in cat_cols:
        for v, sub in df.groupby(cc, observed=True):
            if len(sub) < min_support:
                continue
            for nc in num_cols:
                frac_pos = float((sub[nc] > 0).mean())
                overall_zero = float((df[nc] == 0).mean())
                if frac_pos >= min_conf and overall_zero > 0.02:
                    rules.append(CandidateRule(
                        id=f"denial__{cc}_{v}__requires__{nc}_positive", kind="denial",
                        description=f"{cc} == {v!r} requires {nc} >= 1 "
                                    f"(forbidden: {cc}=={v!r} AND {nc}==0).",
                        columns=[cc, nc],
                        spec={"if_column": cc, "if_value": str(v),
                              "require_column": nc, "require": ">=1"},
                        support=int(len(sub)), confidence=round(frac_pos, 5)))
    return rules


# --------------------------------------------------------------------------
def elicit_rules(df: pd.DataFrame, use_llm: bool = False, descriptions: dict | None = None,
                 sample_n: int = 8, include_values: bool = True) -> list:
    """Run every discovery mechanism and return ranked candidate rules.

    Mechanism 1 (mining) always runs and is 100% local. Mechanism 2 (LLM
    elicitation) runs only when use_llm=True AND credentials are available; every
    LLM-proposed rule is validated against the data before inclusion. sample_n=0 /
    include_values=False makes the LLM prompt descriptions-only (no real rows or
    value domains). Falls back silently to mining-only."""
    sentinels = detect_sentinels(df)
    rules = []
    rules += discover_functional_dependencies(df)
    rules += discover_sentinel_implications(df, sentinels)
    rules += discover_positivity_denials(df)
    rules += discover_category_domains(df)
    rules += discover_ranges(df, sentinels)

    if use_llm:
        existing = {(r.kind, json.dumps(r.spec, sort_keys=True, default=str)) for r in rules}
        for r in llm_elicit_rules(df, descriptions, sample_n=sample_n, include_values=include_values):
            if (r.kind, json.dumps(r.spec, sort_keys=True, default=str)) not in existing:
                rules.append(r)

    priority = {"compare": 0, "fd": 0, "sentinel_implies": 1, "denial": 2, "range": 3, "category_domain": 4}
    rules.sort(key=lambda r: (0 if r.source == "llm" else 1, priority.get(r.kind, 9),
                              -r.confidence, -r.support))
    return rules, sentinels


# --------------------------------------------------------------------------
# Shared rule semantics: a single mask function used for discovery validation,
# reporting, and repair (imported by verity.py). Covers every rule kind,
# including `compare` (cross-column inequality) which the LLM can propose.
# --------------------------------------------------------------------------
def rule_mask(df: pd.DataFrame, kind: str, spec: dict) -> pd.Series:
    """Boolean Series: True where the row VIOLATES this rule."""
    if kind == "range":
        m = (df[spec["column"]] < spec["min"]) | (df[spec["column"]] > spec["max"])
        if spec.get("sentinel") is not None:
            m &= df[spec["column"]] != spec["sentinel"]
        return m
    if kind == "category_domain":
        return ~df[spec["column"]].astype(str).isin(set(spec["allowed"])) & df[spec["column"]].notna()
    if kind == "fd":
        expect = df[spec["determinant"]].astype(str).map(spec["mapping"])
        return expect.notna() & (df[spec["dependent"]].astype(str) != expect.astype(str))
    if kind == "sentinel_implies":
        return (df[spec["if_column"]] == spec["if_value"]) & (df[spec["then_column"]].astype(str) != str(spec["then_value"]))
    if kind == "denial":
        return (df[spec["if_column"]].astype(str) == str(spec["if_value"])) & (df[spec["require_column"]] <= 0)
    if kind == "compare":
        left, right, both = _compare_operands(df, spec)
        op = spec["op"]
        viol = {">=": left < right, ">": left <= right, "<=": left > right,
                "<": left >= right, "==": left != right}.get(op, pd.Series(False, index=df.index))
        return viol & both
    return pd.Series(False, index=df.index)


def _compare_operands(df, spec):
    left = pd.to_numeric(df[spec["left"]], errors="coerce")
    if spec["right"] in df.columns:
        right = pd.to_numeric(df[spec["right"]], errors="coerce")
        both = left.notna() & right.notna()
    else:
        right = pd.Series(float(spec["right"]), index=df.index)
        both = left.notna()
    return left, right, both


def _applicable_count(df, kind, spec) -> int:
    if kind == "sentinel_implies":
        return int((df[spec["if_column"]] == spec["if_value"]).sum())
    if kind == "denial":
        return int((df[spec["if_column"]].astype(str) == str(spec["if_value"])).sum())
    if kind == "compare":
        _, _, both = _compare_operands(df, spec)
        return int(both.sum())
    return len(df)


# --------------------------------------------------------------------------
# Mechanism 2: LLM elicitation (optional, validated, graceful)
# --------------------------------------------------------------------------
LLM_INSTRUCTION = """

Propose logical rules that EVERY valid row must satisfy. Use ONLY these rule kinds
and return STRICT JSON — an object {"rules": [ ... ]} and nothing else:

  {"kind":"compare","description":"...","left":"<col>","op":">="|">"|"<="|"<"|"==","right":"<col or number>"}
      e.g. a discharge date >= an admission date, or an amount <= a credit limit.
  {"kind":"sentinel_implies","description":"...","if_column":"<col>","if_value":<v>,"then_column":"<col>","then_value":<v>}
  {"kind":"denial","description":"...","if_column":"<col>","if_value":"<v>","require_column":"<numeric col>"}
      (meaning: if_column==if_value requires require_column >= 1)

Only propose rules that are logically necessary from the meaning of the columns.
Do not invent columns. If unsure, propose fewer. Output JSON only."""


def _parse_rules_json(text: str) -> list:
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    rules = obj.get("rules", obj) if isinstance(obj, dict) else obj
    return rules if isinstance(rules, list) else []


def validate_proposals(df, proposals, min_conf=0.99, min_support=20) -> list:
    """Keep only LLM proposals the DATA actually supports (drops hallucinations)."""
    out = []
    for i, p in enumerate(proposals):
        kind = p.get("kind")
        if kind not in ("sentinel_implies", "denial", "compare"):
            continue
        spec = {k: v for k, v in p.items() if k not in ("kind", "description")}
        try:
            applicable = _applicable_count(df, kind, spec)
            if applicable < min_support:
                continue
            viol = int(rule_mask(df, kind, spec).sum())
            conf = 1 - viol / applicable if applicable else 0.0
        except Exception:
            continue
        if conf < min_conf:
            continue
        out.append(CandidateRule(
            id=f"llm_{kind}_{i}", kind=kind,
            description=p.get("description", f"{kind} rule"),
            columns=[v for v in spec.values() if isinstance(v, str) and v in df.columns],
            spec=spec, support=int(applicable), confidence=round(float(conf), 5), source="llm"))
    return out


def _is_number(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _resolve_provider() -> str:
    """Pick the LLM provider. VERITY_LLM_PROVIDER wins; else auto-detect from
    whichever API key is present (Groq preferred as the free option)."""
    import os
    p = os.environ.get("VERITY_LLM_PROVIDER", "").lower()
    if p:
        return p
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return ""


def _call_groq(prompt: str, model: str | None) -> str:
    import os
    from groq import Groq
    # Default to a broadly-capable free Groq model; override with VERITY_LLM_MODEL.
    model = model or os.environ.get("VERITY_LLM_MODEL", "openai/gpt-oss-120b")
    resp = Groq().chat.completions.create(          # reads GROQ_API_KEY from env
        model=model, max_tokens=4000, temperature=0,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str, model: str | None) -> str:
    import os
    import anthropic
    model = model or os.environ.get("VERITY_LLM_MODEL", "claude-opus-5")
    resp = anthropic.Anthropic().messages.create(   # reads ANTHROPIC_API_KEY from env
        model=model, max_tokens=4000,
        messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in resp.content
                   if getattr(b, "type", None) == "text")


def llm_elicit_rules(df, descriptions=None, model=None, min_conf=0.99,
                     sample_n=8, include_values=True) -> list:
    """Ask an LLM for SEMANTIC rules, then validate every one against the data.
    Provider-agnostic (Groq or Anthropic, auto-detected from the key present).
    sample_n=0 / include_values=False sends descriptions only (no real rows).
    Returns [] gracefully if no SDK/credentials or the call fails."""
    provider = _resolve_provider()
    if not provider:
        print("  [llm] no LLM API key set (GROQ_API_KEY or ANTHROPIC_API_KEY); using mined rules only.")
        return []
    prompt = build_llm_elicitation_prompt(df, descriptions, sample_n=sample_n,
                                          include_values=include_values) + LLM_INSTRUCTION
    try:
        text = _call_groq(prompt, model) if provider == "groq" else _call_anthropic(prompt, model)
        proposals = _parse_rules_json(text)
    except Exception as e:
        print(f"  [llm] {provider} elicitation unavailable ({type(e).__name__}: {e}); using mined rules only.")
        return []
    validated = validate_proposals(df, proposals, min_conf=min_conf)
    print(f"  [llm] {provider} proposed {len(proposals)} rules; {len(validated)} survived data validation.")
    return validated


def build_user_rule_prompt(df, text: str, descriptions=None, include_values=True) -> str:
    """Prompt to convert a user's plain-English business rules into structured rules.
    Never sends sample rows; include_values=False also omits real value domains."""
    lines = ["Convert the user's plain-English business rules into structured rules.",
             "Use ONLY the column names listed below; one structured rule per stated rule.",
             "", "COLUMNS:"]
    lines += _schema_lines(df, descriptions, include_values)
    lines += ["", "USER RULES (plain English):", text.strip()]
    return "\n".join(lines) + LLM_INSTRUCTION


def parse_user_rules(df, text: str, descriptions=None, model=None, min_conf=0.0,
                     include_values=True) -> list:
    """Turn a user's plain-English business rules into validated CandidateRules.
    Unlike mined/LLM-proposed rules, typed rules are kept even if the data only
    partially satisfies them (min_conf=0 by default) — the user is asserting how
    valid data SHOULD look — but each rule's confidence on the real data is
    reported so a mismatch is visible. Needs an LLM key; returns [] gracefully."""
    if not text or not text.strip():
        return []
    provider = _resolve_provider()
    if not provider:
        print("  [rules] plain-language rules need GROQ_API_KEY or ANTHROPIC_API_KEY; skipped.")
        return []
    prompt = build_user_rule_prompt(df, text, descriptions, include_values=include_values)
    try:
        out = _call_groq(prompt, model) if provider == "groq" else _call_anthropic(prompt, model)
        proposals = _parse_rules_json(out)
    except Exception as e:
        print(f"  [rules] {provider} parse failed ({type(e).__name__}: {e}); no typed rules added.")
        return []
    parsed = validate_proposals(df, proposals, min_conf=min_conf, min_support=1)
    for r in parsed:
        r.source = "typed"
        r.id = "typed_" + r.id
    print(f"  [rules] parsed {len(proposals)} plain-language rules; kept {len(parsed)} "
          f"(validated against your data).")
    return parsed


def _schema_lines(df, descriptions=None, include_values=True) -> list:
    """Schema block for an LLM prompt. When include_values is False, emit only
    column name + type + user description — no real value domains or min/max —
    so no information derived from individual real records leaves the machine."""
    prof = profile_dataset(df)
    lines = []
    for c, p in prof.items():
        d = f" — {descriptions[c]}" if descriptions and c in descriptions else ""
        if p["dtype"] == "numeric":
            rng = f", {p['min']:g}..{p['max']:g}" if include_values else ""
            lines.append(f"  {c} (numeric{rng}){d}")
        else:
            if include_values:
                dom = ", ".join(p.get("domain", [])[:8])
                lines.append(f"  {c} (categorical: {dom}{'…' if p['n_unique'] > 8 else ''}){d}")
            else:
                lines.append(f"  {c} (categorical){d}")
    return lines


def build_llm_elicitation_prompt(df, descriptions=None, sample_n=8, include_values=True) -> str:
    """Mechanism 2 scaffold: the prompt Verity sends an LLM to propose SEMANTIC
    rules mining can miss. sample_n=0 / include_values=False = descriptions-only
    (no real rows or value domains are sent). (No API call here.)"""
    lines = ["You are a data-integrity expert. Given this table's schema"
             + (" and a sample" if sample_n else "") + ", propose logical business",
             "rules that every valid row must satisfy (functional dependencies,",
             "forbidden value combinations, plausible ranges).",
             "", "SCHEMA:"]
    lines += _schema_lines(df, descriptions, include_values)
    if sample_n and include_values:
        lines += ["", "SAMPLE ROWS:",
                  df.sample(min(sample_n, len(df)), random_state=0).to_csv(index=False).strip()]
    return "\n".join(lines)


def _print_report(name, rules, sentinels):
    print("=" * 74)
    print(f"DISCOVERED CANDIDATE RULES — {name}")
    print(f"(sentinels detected: { {k: float(v) for k, v in sentinels.items()} })")
    print("=" * 74)
    shown = [r for r in rules if r.kind in ("fd", "sentinel_implies", "denial")]
    for r in shown:
        print(f"  [{r.kind:16}] conf={r.confidence:<6} support={r.support:<6} {r.description}")
    n_dom = sum(r.kind == "category_domain" for r in rules)
    n_rng = sum(r.kind == "range" for r in rules)
    print(f"  ... plus {n_dom} category-domain and {n_rng} range constraints (auto-derived).")
    print(f"  TOTAL candidate rules: {len(rules)}  (semantic: {len(shown)})")


def run_on(name, csv_name, descriptions=None, exclude=None):
    df = pd.read_csv(REPO / "data" / "processed" / csv_name)
    # Exclude label / derived columns from constraint discovery: they are the
    # prediction target, not schema fields, and produce trivial label-leakage
    # FDs (e.g. y <-> target). In the product the user names the label.
    if exclude:
        df = df.drop(columns=[c for c in exclude if c in df.columns])
    rules, sentinels = elicit_rules(df)
    _print_report(name, rules, sentinels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_rules.json"
    out.write_text(json.dumps([asdict(r) for r in rules], indent=2, default=str))
    print(f"  -> wrote {len(rules)} candidates to {out.relative_to(REPO)}\n")
    return df


if __name__ == "__main__":
    print("\nVerity — automatic business-rule discovery (prototype)\n")

    df_bank = run_on("bank_marketing", "bank_marketing_clean.csv", exclude=["target", "y"])
    run_on("adult_income", "adult_income_clean.csv", exclude=["target"])

    print("-" * 74)
    print("LLM ELICITATION PROMPT (mechanism 2 scaffold) — Bank Marketing, first 22 lines:")
    print("-" * 74)
    prompt = build_llm_elicitation_prompt(
        df_bank,
        descriptions={"pdays": "days since last contact; -1 means never contacted",
                      "previous": "number of prior contacts",
                      "poutcome": "outcome of the previous campaign"})
    print("\n".join(prompt.splitlines()[:22]))
    print("...\n[plug an LLM call here to get semantic candidate rules for review]")
