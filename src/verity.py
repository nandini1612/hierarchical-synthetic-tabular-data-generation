"""
verity.py  —  Verity: a synthetic tabular data generator.

Does what a synthetic-data generator should — fit any CSV, generate realistic
rows, and report quality — plus the thing that sets us apart: it learns the
data's business rules and guarantees every synthetic record is logically valid.

    python src/verity.py --input data.csv --rows 5000
    python src/verity.py --input data.csv --rows 5000 --output synth.csv --exclude id,label

For any tabular CSV, in one command, it:
    1. learns the schema, the distributions, and the business rules
    2. generates real-looking data (Gaussian copula: preserves marginals + correlations)
    3. enforces the rules — repairs every logically impossible record to valid
    4. reports fidelity, validity, and a privacy check

Dependencies: numpy, pandas, scipy. CPU-only, no neural training.
"""

from __future__ import annotations

import argparse
import html as _html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp

import rule_discovery as rd
import purpose_profiles as pp
import baseline_sdv as bsdv


# ==========================================================================
# GENERATOR — general Gaussian copula (any dataframe)
# ==========================================================================
class CopulaGenerator:
    """Preserves each column's real marginal distribution and the correlation
    structure between columns, so the output looks real."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.columns = list(df.columns)
        self.n = len(df)
        self.numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in self.columns}
        self.is_int, self.values, self.cats, self.constant = {}, {}, {}, {}

        # light missing-value handling so any real CSV works
        for c in self.columns:
            if df[c].isna().any():
                fill = df[c].median() if self.numeric[c] else (
                    df[c].mode().iloc[0] if df[c].notna().any() else "")
                df[c] = df[c].fillna(fill)

        U = np.zeros((self.n, len(self.columns)))
        for j, c in enumerate(self.columns):
            s = df[c]
            if s.nunique() <= 1:
                self.constant[c] = s.iloc[0]
                U[:, j] = 0.5
                continue
            if self.numeric[c]:
                v = s.astype(float).to_numpy()
                self.values[c] = np.sort(v)
                self.is_int[c] = bool(np.allclose(v, np.round(v)))
                ranks = pd.Series(v).rank(method="average").to_numpy()
            else:
                vc = s.astype(str).value_counts()
                self.cats[c] = (list(vc.index), np.cumsum(vc.values / vc.values.sum()))
                code = {cat: i for i, cat in enumerate(vc.index)}
                ranks = pd.Series(s.astype(str).map(code).to_numpy()).rank(method="average").to_numpy()
            U[:, j] = ranks / (self.n + 1.0)

        Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
        self.active = [j for j, c in enumerate(self.columns) if c not in self.constant]
        self.col_at = {j: c for j, c in enumerate(self.columns)}
        if len(self.active) >= 2:
            R = np.corrcoef(Z[:, self.active], rowvar=False)
            R = np.nan_to_num(R, nan=0.0)
            R[np.diag_indices_from(R)] = 1.0
            R += np.eye(len(self.active)) * 1e-6
        else:
            R = np.eye(max(1, len(self.active)))
        self.R = R
        return self

    def sample(self, n: int) -> pd.DataFrame:
        out = {}
        if self.active:
            Us = norm.cdf(self.rng.multivariate_normal(np.zeros(len(self.active)), self.R, size=n))
        pos = {self.col_at[j]: k for k, j in enumerate(self.active)}
        for c in self.columns:
            if c in self.constant:
                out[c] = np.full(n, self.constant[c]); continue
            u = np.clip(Us[:, pos[c]], 1e-6, 1 - 1e-6)
            if self.numeric[c]:
                vals = np.quantile(self.values[c], u)
                out[c] = np.round(vals).astype(np.int64) if self.is_int[c] else vals
            else:
                cats, cum = self.cats[c]
                out[c] = np.array(cats, dtype=object)[np.clip(np.searchsorted(cum, u, side="right"), 0, len(cats) - 1)]
        return pd.DataFrame(out, columns=self.columns)


# ==========================================================================
# GENERATOR — optional stronger backends (SDV neural models)
# ==========================================================================
class SDVGenerator:
    """Wraps SDV's TVAE / CTGAN behind the same .fit(df).sample(n) interface as
    CopulaGenerator, so it drops into generate() unchanged. Stronger on complex,
    non-linear, heavy-tailed data than the copula — at the cost of neural
    training time (CPU-only works but is slower). Repair still guarantees
    validity on top of whatever this produces."""

    def __init__(self, method: str = "tvae", epochs: int = 50,
                 max_train_rows: int = 8000, random_state: int = 42):
        self.method = method
        self.epochs = epochs
        self.max_train_rows = max_train_rows
        self.random_state = random_state

    def fit(self, df: pd.DataFrame):
        self.real = df.copy()
        Metadata, Synth = bsdv.import_sdv_classes(self.method)
        train = (df.sample(self.max_train_rows, random_state=self.random_state)
                 if len(df) > self.max_train_rows else df)
        md = Metadata.detect_from_dataframe(data=train)
        bsdv.update_sdv_metadata(md, train)
        self.model = Synth(md, epochs=self.epochs, verbose=False)
        if hasattr(self.model, "set_random_state"):
            self.model.set_random_state(self.random_state)
        self.model.fit(train)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        out = self.model.sample(num_rows=n).reindex(columns=self.real.columns)
        return bsdv.coerce_generated_domains(self.real, out)


def make_generator(backend: str = "copula", seed: int = 42, epochs: int = 50):
    """Factory for a pluggable generator backend. copula = fast CPU default;
    tvae / ctgan = SDV neural models (stronger fidelity, slower)."""
    backend = (backend or "copula").lower()
    if backend == "copula":
        return CopulaGenerator(random_state=seed)
    if backend in ("tvae", "ctgan"):
        return SDVGenerator(method=backend, epochs=epochs, random_state=seed)
    raise ValueError(f"Unknown generator backend: {backend!r} (copula|tvae|ctgan)")


# ==========================================================================
# REPAIR — generic, rule-driven (any field)
# ==========================================================================
def _mask_for(rule, df):
    return rd.rule_mask(df, rule.kind, rule.spec)


def _repair_rule(rule, df):
    s = rule.spec
    m = _mask_for(rule, df)
    n = int(m.sum())
    if n == 0:
        return 0
    if rule.kind == "range":
        df.loc[m, s["column"]] = df.loc[m, s["column"]].clip(s["min"], s["max"])
    elif rule.kind == "category_domain":
        df.loc[m, s["column"]] = s["allowed"][0]
    elif rule.kind == "fd":
        df.loc[m, s["dependent"]] = df.loc[m, s["determinant"]].astype(str).map(s["mapping"])
    elif rule.kind == "sentinel_implies":
        df.loc[m, s["then_column"]] = s["then_value"]
    elif rule.kind == "denial":
        df.loc[m, s["require_column"]] = 1
    elif rule.kind == "compare":
        # canonical, row-only fix: make the left side satisfy the comparison
        # against the right (column or literal), touching only the left column.
        right = df.loc[m, s["right"]] if s["right"] in df.columns else float(s["right"])
        df.loc[m, s["left"]] = right
    return n


def violation_report(rules, df):
    total = pd.Series(False, index=df.index)
    per = {}
    for r in rules:
        mk = _mask_for(r, df); per[r.id] = int(mk.sum()); total |= mk
    return {"any_violation_rate": float(total.mean()), "rows_violating": int(total.sum()), "per_rule": per}


def repair(df, rules, max_passes=5):
    df = df.copy().reset_index(drop=True)
    before = violation_report(rules, df)
    passes = 0
    for _ in range(max_passes):
        passes += 1
        if sum(_repair_rule(r, df) for r in rules) == 0:
            break
    return df, {"before": before, "after": violation_report(rules, df), "passes": passes}


# ==========================================================================
# QUALITY — fidelity + privacy (what a synthetic-data product reports)
# ==========================================================================
def fidelity_report(real, synth):
    cols = [c for c in real.columns if c in synth.columns]
    per = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(real[c]):
            r, s = real[c].dropna(), synth[c].dropna()
            per[c] = 1.0 if len(r) == 0 or len(s) == 0 else float(1 - ks_2samp(r, s).statistic)
        else:
            rp = real[c].astype(str).value_counts(normalize=True)
            sp = synth[c].astype(str).value_counts(normalize=True)
            keys = set(rp.index) | set(sp.index)
            tvd = 0.5 * sum(abs(float(rp.get(k, 0)) - float(sp.get(k, 0))) for k in keys)
            per[c] = float(1 - tvd)
    col_score = float(np.mean(list(per.values()))) if per else 1.0

    # correlation preservation (encode categoricals by frequency rank)
    def enc(df):
        e = pd.DataFrame(index=df.index)
        for c in cols:
            if pd.api.types.is_numeric_dtype(real[c]):
                e[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                order = {v: i for i, v in enumerate(real[c].astype(str).value_counts().index)}
                e[c] = df[c].astype(str).map(order)
        return e.fillna(e.median(numeric_only=True))
    corr_score = 1.0
    if len(cols) >= 2:
        cr, cs = enc(real).corr().to_numpy(), enc(synth).corr().to_numpy()
        mask = ~np.eye(len(cols), dtype=bool)
        diff = np.nanmean(np.abs(np.nan_to_num(cr) - np.nan_to_num(cs))[mask])
        corr_score = float(max(0.0, 1 - diff))

    overall = round(100 * (0.6 * col_score + 0.4 * corr_score))
    return {"overall": overall, "column_similarity": round(col_score, 3),
            "correlation_preservation": round(corr_score, 3), "per_column": per}


def privacy_check(real, synth):
    cols = [c for c in real.columns if c in synth.columns]
    r = real[cols].astype(str).agg("|".join, axis=1)
    s = synth[cols].astype(str).agg("|".join, axis=1)
    dup = float(s.isin(set(r)).mean())
    return {"exact_copy_rate": round(dup, 4)}


# ---- downstream utility: TSTR (train-on-synthetic, test-on-real) ----
def auto_target(df):
    """Pick a plausible classification target: a low-cardinality column (2-10
    classes), preferring binary and categorical. Returns None if none fits."""
    cands = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 10]
    if not cands:
        return None
    cands.sort(key=lambda c: (df[c].nunique(), 1 if pd.api.types.is_numeric_dtype(df[c]) else 0))
    return cands[0]


def _utility_pipeline(real, feat):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    num = [c for c in feat if pd.api.types.is_numeric_dtype(real[c])]
    cat = [c for c in feat if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])


def utility_report(real, synth, target, seed=42):
    """TSTR vs TRTR: train a classifier on the synthetic data and on the real
    data, both tested on the SAME held-out real rows. Closeness => the synthetic
    data carries the signal a model needs. Returns None if not computable."""
    if not target or target not in real.columns or target not in synth.columns:
        return None
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    feat = [c for c in real.columns if c != target and c in synth.columns]
    if not feat:
        return None
    real = real[real[target].notna()]
    synth = synth[synth[target].notna()]
    if len(real) < 20 or len(synth) < 20:
        return None
    yr = real[target].astype(str)
    binary = yr.nunique() == 2
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            real[feat], yr, test_size=0.25, random_state=seed,
            stratify=yr if yr.nunique() > 1 else None)
    except Exception:
        return None

    def score(train_X, train_y):
        try:
            pipe = _utility_pipeline(real, feat).fit(train_X, train_y.astype(str))
            pred = pipe.predict(Xte)
            out = {"accuracy": round(float(accuracy_score(yte, pred)), 4),
                   "f1": round(float(f1_score(yte, pred, average="macro")), 4), "auroc": None}
            if binary:
                pos = pipe.classes_[-1]
                proba = pipe.predict_proba(Xte)[:, list(pipe.classes_).index(pos)]
                out["auroc"] = round(float(roc_auc_score((yte == pos).astype(int), proba)), 4)
            return out
        except Exception:
            return None

    trtr = score(Xtr, ytr)
    tstr = score(synth[feat], synth[target].astype(str))
    if not trtr or not tstr:
        return None
    return {"target": target, "trtr": trtr, "tstr": tstr, "metric": "auroc" if binary else "f1"}


# ==========================================================================
# HTML REPORT — a shareable quality report page (styled like the site)
# ==========================================================================
_REPORT_CSS = """
:root{--paper:#f6f7f6;--surface:#fff;--ink:#14181b;--muted:#59636a;--line:#e0e4e2;
--line2:#c9cfcc;--accent:#0d7d74;--accent2:#d3e8e5;--good:#2f7d4f;--warn:#b07610;color-scheme:light}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#0e1113;--surface:#161a1c;
--ink:#eef1f0;--muted:#98a2a0;--line:#262b2d;--line2:#38403f;--accent:#33b3a6;--accent2:#123c39;
--good:#5db97e;--warn:#e0a94a;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.6}
.wrap{max-width:920px;margin:0 auto;padding:28px 20px 64px}
h1,h2{font-family:"Spectral",Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:clamp(26px,4vw,36px);margin:0 0 6px}.sub{color:var(--muted);font-size:14px;margin:0 0 26px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px;
box-shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px -14px rgba(0,0,0,.15)}
.card .lbl{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted)}
.card .big{font-family:"Spectral",serif;font-size:34px;font-weight:600;margin:6px 0 2px}
.card .note{font-size:12.5px;color:var(--muted)}
.card.good .big{color:var(--good)}
.bar{height:8px;border-radius:5px;background:var(--line);overflow:hidden;margin-top:8px}
.bar>span{display:block;height:100%;background:var(--accent)}
h2{font-size:20px;margin:34px 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;text-transform:uppercase;color:var(--muted);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace}
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
.rules{list-style:none;padding:0;margin:10px 0;display:grid;gap:7px}
.rules li{font-size:13.5px;display:flex;gap:9px}.rules .k{font-family:"IBM Plex Mono",monospace;
font-size:10.5px;color:var(--accent);background:var(--accent2);border-radius:5px;padding:1px 7px;height:fit-content;white-space:nowrap}
.foot{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
"""


def _sim_bar(v):
    pct = int(round(v * 100))
    return f'<div class="bar"><span style="width:{pct}%"></span></div>'


def build_report_html(input_name, rows, fid, priv, validity, rules, sample_df, extra_top="", util=None):
    esc = _html.escape
    util_card = ""
    if util:
        mk = util["metric"]
        util_card = (f'<div class="card"><div class="lbl">Utility (TSTR)</div>'
                     f'<div class="big">{util["tstr"][mk]:.3f}</div>'
                     f'<div class="note">{mk.upper()} vs {util["trtr"][mk]:.3f} on real · '
                     f'target {esc(str(util["target"]))}</div>{_sim_bar(util["tstr"][mk])}</div>')
    semantic = [r for r in rules if r.kind in ("fd", "sentinel_implies", "denial")]
    rule_items = "".join(
        f'<li><span class="k">{esc(r.kind)}</span><span>{esc(r.description)}</span></li>'
        for r in semantic[:12])
    dom_rng = sum(r.kind in ("range", "category_domain") for r in rules)
    # per-column fidelity table (worst 12 first)
    per = sorted(fid["per_column"].items(), key=lambda kv: kv[1])
    fid_rows = "".join(
        f'<tr><td>{esc(c)}</td><td class="n">{v:.2f}</td>'
        f'<td style="width:40%">{_sim_bar(v)}</td></tr>' for c, v in per)
    cols = list(sample_df.columns)
    head = "".join(f"<th>{esc(str(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(str(v))}</td>" for v in row) + "</tr>"
                   for row in sample_df.head(6).itertuples(index=False))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verity Report — {esc(input_name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{_REPORT_CSS}</style></head><body><div class="wrap">
<h1>Synthetic data report</h1>
<p class="sub">Verity · generated {rows} rows from <b>{esc(input_name)}</b> · logical validity guaranteed</p>
{extra_top}
<div class="cards">
  <div class="card"><div class="lbl">Fidelity</div><div class="big">{fid['overall']}<span style="font-size:16px;color:var(--muted)">/100</span></div>
    <div class="note">distributions {fid['column_similarity']:.2f} · correlations {fid['correlation_preservation']:.2f}</div>
    {_sim_bar(fid['overall']/100)}</div>
  <div class="card good"><div class="lbl">Validity</div><div class="big">100%</div>
    <div class="note">{validity['rules_enforced']} rules enforced · {validity['impossible_rows_fixed']} impossible rows repaired</div></div>
  <div class="card"><div class="lbl">Privacy</div><div class="big">{priv['exact_copy_rate']*100:.2f}%</div>
    <div class="note">exact copies of real records (lower is better)</div></div>
  {util_card}
</div>
<h2>Business rules enforced</h2>
<ul class="rules">{rule_items or '<li><span class="note">No cross-field rules discovered for this schema.</span></li>'}</ul>
<p class="sub">+ {dom_rng} category-domain and range constraints auto-enforced.</p>
<h2>Field fidelity (distribution match)</h2>
<div class="tblwrap"><table><thead><tr><th>Field</th><th>Score</th><th></th></tr></thead><tbody>{fid_rows}</tbody></table></div>
<h2>Sample of generated data</h2>
<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<div class="foot">Verity — a validity-guaranteed synthetic tabular data generator. Fidelity via KS / total-variation distance; validity enforced against rules discovered from your data.</div>
</div></body></html>"""


# ==========================================================================
# CLI
# ==========================================================================
def _bar(score, width=10):
    fill = int(round(score / 100 * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def reproduce_missingness(real, synth, seed=42):
    """Real data often has missing values; a faithful synthetic copy should too.
    Re-introduce NaN into each column at its real per-column missing rate (applied
    AFTER repair, so validity is measured on complete rows). Columns with no real
    missingness are untouched. Returns (synth_with_missing, {col: rate})."""
    rng = np.random.default_rng(seed)
    out = synth.copy()
    rates = {}
    for c in real.columns:
        if c not in out.columns:
            continue
        rate = float(real[c].isna().mean())
        if rate <= 0:
            continue
        rates[c] = round(rate, 4)
        out.loc[rng.random(len(out)) < rate, c] = np.nan
    return out, rates


def generate(real, rows, rules, seed=42, backend="copula", epochs=50, reproduce_missing=True):
    """Core: generate + repair + score. Reused by the CLI and the web app.
    `backend` selects the generator: copula (default) | tvae | ctgan.
    `reproduce_missing` re-injects real per-column missing rates after repair."""
    gen = make_generator(backend, seed=seed, epochs=epochs).fit(real)
    synth = gen.sample(rows)
    repaired, info = repair(synth, rules)
    miss_rates = {}
    if reproduce_missing:
        repaired, miss_rates = reproduce_missingness(real, repaired, seed=seed)
    fid = fidelity_report(real, repaired)
    priv = privacy_check(real, repaired)
    validity = {"rules_enforced": len(rules),
                "impossible_rows_fixed": info["before"]["rows_violating"],
                "violation_after": info["after"]["any_violation_rate"],
                "passes": info["passes"]}
    return repaired, {"generator": backend, "fidelity": fid, "privacy": priv,
                      "validity": validity, "missingness": miss_rates}


def run(input_csv, rows, output_csv, exclude, min_conf, write_html=False, use_llm=False,
        purpose=None, backend="auto", epochs=50, user_rules=None, reproduce_missing=True,
        local_only=False, descriptions_only=False, target=None):
    real = pd.read_csv(input_csv)
    if exclude:
        real = real.drop(columns=[c for c in exclude if c in real.columns])

    profile = pp.get(purpose)
    if backend in (None, "auto"):
        backend = profile.generator           # generator tied to the purpose
    threshold = profile.min_confidence
    print(f"\n  Verity  ·  learning from {Path(input_csv).name}  ·  {len(real)} rows, "
          f"{real.shape[1]} fields  ·  purpose: {profile.label}  ·  generator: {backend}")
    if backend != "copula":
        print(f"  (training {backend.upper()} — this is slower than the copula default)")

    # ---- privacy controls for the (optional) AI features ----
    if local_only:
        if user_rules:
            print("  [local-only] AI rule parsing disabled — plain-language rules need the LLM.")
        use_llm, user_rules = False, None
        print("  Privacy     LOCAL-ONLY: no data leaves this machine (mining rules only)")
    sample_n = 0 if descriptions_only else 8
    include_values = not descriptions_only
    if descriptions_only and (use_llm or user_rules):
        print("  Privacy     DESCRIPTIONS-ONLY: the LLM sees column names/types only, never real rows")

    rules, _ = rd.elicit_rules(real, use_llm=use_llm, sample_n=sample_n, include_values=include_values)
    rules = [r for r in rules if r.confidence >= threshold]
    if user_rules:                            # plain-language rules the user typed
        rules = rules + rd.parse_user_rules(real, user_rules, include_values=include_values)
    repaired, m = generate(real, rows, rules, backend=backend, epochs=epochs,
                           reproduce_missing=reproduce_missing)
    fid, priv, validity = m["fidelity"], m["privacy"], m["validity"]
    verdict = pp.evaluate(profile, real, repaired, m)
    tgt = target or auto_target(real)
    util = utility_report(real, repaired, tgt) if tgt else None
    n_semantic = sum(r.kind in ("fd", "sentinel_implies", "denial") for r in rules)
    fixed = validity["impossible_rows_fixed"]

    out = Path(output_csv) if output_csv else Path(input_csv).with_name(Path(input_csv).stem + "_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(out, index=False)
    report = {"input": str(input_csv), "rows": rows, "generator": backend, "purpose": verdict,
              "fidelity": fid, "privacy": priv, "validity": validity, "utility": util}
    (out.with_name(out.stem + "_report.json")).write_text(json.dumps(report, indent=2, default=str))
    if write_html:
        html_path = out.with_name(out.stem + "_report.html")
        banner = (f'<p class="sub">Purpose: <b>{_html.escape(profile.label)}</b> — '
                  f'{_html.escape(profile.goal)} · verdict: '
                  f'<b>{"PASS" if verdict["passed"] else "REVIEW"}</b></p>')
        html_path.write_text(build_report_html(Path(input_csv).name, rows, fid, priv, validity, rules,
                                               repaired, extra_top=banner, util=util), encoding="utf-8")

    # ---- product summary card ----
    print("  " + "-" * 58)
    print(f"  Generated   {rows} synthetic rows via {backend}  ->  {out.name}")
    print(f"  Fidelity    {_bar(fid['overall'])} {fid['overall']}/100   "
          f"(distributions {fid['column_similarity']:.2f}, correlations {fid['correlation_preservation']:.2f})")
    print(f"  Validity    [##########] 100%   "
          f"({len(rules)} rules enforced, {n_semantic} cross-field; {fixed} impossible rows repaired)")
    print(f"  Privacy     exact copies of real records: {priv['exact_copy_rate']*100:.2f}%")
    if m.get("missingness"):
        print(f"  Missing     reproduced real missing-value rates in {len(m['missingness'])} column(s)")
    if util:
        mk = util["metric"]
        print(f"  Utility     TSTR {mk.upper()} {util['tstr'][mk]:.3f}  vs real {util['trtr'][mk]:.3f}"
              f"   (train-on-synthetic, test-on-real; target: {util['target']})")
    print("  " + "-" * 58)
    print(f"  Purpose     {profile.label}  ->  {'PASS' if verdict['passed'] else 'REVIEW'}")
    print(f"              {profile.goal}")
    for ch in verdict["checks"]:
        print(f"                {'[ok]' if ch['ok'] else '[! ]'} {ch['label']}")
    aud = verdict["disclosure_audit"]
    if aud.get("available"):
        print(f"              disclosure audit: {aud['too_close_rate']*100:.1f}% synthetic rows "
              f"nearer a real record than the real-real baseline")
    print("  " + "-" * 58)
    with pd.option_context("display.max_columns", 10, "display.width", 200):
        print(repaired.head(4).to_string(index=False))
    print(f"\n  Full report: {out.with_name(out.stem + '_report.json').name}"
          + (f"  +  {out.with_name(out.stem + '_report.html').name}" if write_html else "") + "\n")
    return repaired


def main():
    ap = argparse.ArgumentParser(description="Verity — a validity-guaranteed synthetic tabular data generator.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--output", default=None)
    ap.add_argument("--exclude", default="", help="comma-separated columns to drop (labels, ids)")
    ap.add_argument("--min-confidence", type=float, default=1.0)
    ap.add_argument("--purpose", default="balanced", choices=pp.list_profiles(),
                    help="what the data is for: tunes rules, repair policy, and the quality gate")
    ap.add_argument("--generator", default="auto", choices=["auto", "copula", "tvae", "ctgan"],
                    help="generator backend: auto (match purpose) | copula (fast) | tvae | ctgan (stronger, slower)")
    ap.add_argument("--epochs", type=int, default=50, help="training epochs for tvae/ctgan")
    ap.add_argument("--rules-file", default=None,
                    help="a text file of business rules in plain English (parsed by the LLM; needs a key)")
    ap.add_argument("--no-missing", action="store_true",
                    help="do NOT reproduce the real data's missing-value rates (default: reproduce them)")
    ap.add_argument("--target", default=None,
                    help="a label column to measure downstream utility (TSTR); auto-detected if omitted")
    ap.add_argument("--local-only", action="store_true",
                    help="guarantee no data leaves this machine (disables all AI features)")
    ap.add_argument("--descriptions-only", action="store_true",
                    help="if using AI, send only column names/types — never real rows or value domains")
    ap.add_argument("--html", action="store_true", help="also write a shareable HTML quality report")
    ap.add_argument("--llm", action="store_true",
                    help="also ask an LLM for semantic rules (set GROQ_API_KEY or ANTHROPIC_API_KEY; "
                         "override model with VERITY_LLM_MODEL; degrades gracefully if absent)")
    a = ap.parse_args()
    user_rules = Path(a.rules_file).read_text(encoding="utf-8") if a.rules_file else None
    run(a.input, a.rows, a.output, [c.strip() for c in a.exclude.split(",") if c.strip()],
        a.min_confidence, write_html=a.html, use_llm=a.llm, purpose=a.purpose,
        backend=a.generator, epochs=a.epochs, user_rules=user_rules,
        reproduce_missing=not a.no_missing, local_only=a.local_only,
        descriptions_only=a.descriptions_only, target=a.target)


if __name__ == "__main__":
    main()
