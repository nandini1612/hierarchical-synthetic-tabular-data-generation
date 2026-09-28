"""
verity_app.py  —  Verity web app (clickable product).

A tiny local web UI over the Verity engine:
    1. upload any CSV
    2. review the business rules Verity discovered, and APPROVE the ones to enforce
    3. generate synthetic data and see the quality report
    4. download the synthetic CSV

Run:
    python src/verity_app.py            # then open http://127.0.0.1:5000

Everything runs locally; uploaded files stay in a temp folder on your machine.
"""

from __future__ import annotations

import html as _html
import json
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, request, send_file, redirect, url_for, abort

import rule_discovery as rd
import verity
import purpose_profiles as pp

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB uploads
SESS = Path(tempfile.gettempdir()) / "verity_sessions"
SESS.mkdir(parents=True, exist_ok=True)

CSS = verity._REPORT_CSS + """
.btn{display:inline-block;background:var(--accent);color:#fff;text-decoration:none;border:0;
font-family:"IBM Plex Sans",sans-serif;font-size:14px;font-weight:500;padding:11px 20px;
border-radius:10px;cursor:pointer}.btn:hover{filter:brightness(1.07)}
.btn.ghost{background:transparent;color:var(--ink);border:1px solid var(--line2)}
label{font-size:14px}input[type=text],input[type=number]{font-family:"IBM Plex Mono",monospace;
background:var(--surface);color:var(--ink);border:1px solid var(--line2);border-radius:9px;padding:9px 11px;font-size:14px}
.drop{border:1.5px dashed var(--line2);border-radius:14px;padding:34px;text-align:center;background:var(--surface)}
.field{display:flex;flex-direction:column;gap:6px;margin:16px 0}
.rule{display:flex;gap:12px;align-items:flex-start;background:var(--surface);border:1px solid var(--line);
border-radius:11px;padding:12px 14px;margin:8px 0}.rule input{margin-top:3px}
.rule .k{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--accent);background:var(--accent2);
border-radius:5px;padding:1px 7px;white-space:nowrap}.rule .meta{font-size:11.5px;color:var(--muted);margin-top:3px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}
"""


def page(body, title="Verity"):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><div class="wrap">{body}</div></body></html>"""


@app.route("/")
def index():
    purpose_opts = "".join(
        f'<option value="{n}">{_html.escape(pp.get(n).label)}</option>' for n in pp.list_profiles())
    return page(f"""
<p class="eyebrow">Verity</p>
<h1>Synthetic data, guaranteed logically valid</h1>
<p class="sub">Upload a CSV. Verity learns its distributions and business rules, generates real-looking
data, and repairs every impossible record — then shows you a quality report.</p>
<form method="post" action="/discover" enctype="multipart/form-data">
  <div class="drop">
    <input type="file" name="file" accept=".csv" required>
    <p class="sub" style="margin:12px 0 0">any tabular .csv</p>
  </div>
  <div class="field"><label>Rows to generate</label><input type="number" name="rows" value="5000" min="10" max="200000"></div>
  <div class="field"><label>What is this data for?</label>
    <select name="purpose" style="font-family:'IBM Plex Mono',monospace;background:var(--surface);color:var(--ink);border:1px solid var(--line2);border-radius:9px;padding:9px 11px;font-size:14px">
      {purpose_opts}
    </select>
    <span class="sub">Privacy-safe sharing forbids real-data copying and audits disclosure; others lead on fidelity/validity.</span></div>
  <div class="field"><label>Generator</label>
    <select name="generator" style="font-family:'IBM Plex Mono',monospace;background:var(--surface);color:var(--ink);border:1px solid var(--line2);border-radius:9px;padding:9px 11px;font-size:14px">
      <option value="auto">Auto — match the purpose (default)</option>
      <option value="copula">Gaussian copula — fast</option>
      <option value="tvae">TVAE — neural, stronger, slower</option>
      <option value="ctgan">CTGAN — neural, stronger, slower</option>
    </select>
    <span class="sub">Auto picks the right generator for your purpose (e.g. ML training uses TVAE). Validity is guaranteed regardless; neural backends can take minutes.</span></div>
  <div class="field"><label>Columns to exclude (labels / IDs, comma-separated) — optional</label>
    <input type="text" name="exclude" placeholder="e.g. id"></div>
  <div class="field"><label>Label column for a usefulness (ML) check — optional</label>
    <input type="text" name="target" placeholder="auto-detected if left blank">
    <span class="sub">Trains a model on the synthetic data and tests on real (TSTR) to prove it's useful, not just valid. Don't also exclude this column.</span></div>
  <div class="field"><label>Business rules in plain English — optional</label>
    <textarea name="user_rules" rows="3" placeholder="e.g. discharge date must be on or after admission date; a successful outcome requires at least one prior contact" style="font-family:'IBM Plex Sans',sans-serif;background:var(--surface);color:var(--ink);border:1px solid var(--line2);border-radius:9px;padding:10px 12px;font-size:14px;resize:vertical"></textarea>
    <span class="sub">Type rules your domain knows; the AI turns them into enforced constraints (needs an API key). Each is checked against your data.</span></div>
  <label class="field" style="flex-direction:row;align-items:center;gap:8px">
    <input type="checkbox" name="use_llm" value="1">
    <span>Also use AI to suggest semantic rules <span class="sub">(set GROQ_API_KEY or ANTHROPIC_API_KEY; every suggestion is validated against your data)</span></span></label>
  <div class="field" style="gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:12px 14px">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)">Data privacy</div>
    <label style="display:flex;align-items:center;gap:8px;font-size:13.5px"><input type="checkbox" name="local_only" value="1">
      <span><b>Local-only mode</b> — guarantee no data leaves this machine (disables all AI features)</span></label>
    <label style="display:flex;align-items:center;gap:8px;font-size:13.5px"><input type="checkbox" name="descriptions_only" value="1">
      <span><b>Descriptions-only</b> — if using AI, send only column names &amp; types, never real rows or values</span></label>
  </div>
  <button class="btn" type="submit">Discover rules &rarr;</button>
</form>
""", "Verity")


@app.route("/discover", methods=["POST"])
def discover():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return page('<h1>Please upload a .csv file.</h1><p><a class="btn ghost" href="/">Back</a></p>')
    token = uuid.uuid4().hex[:12]
    d = SESS / token
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / "input.csv"
    f.save(csv_path)
    rows = max(10, min(200000, int(request.form.get("rows", 5000) or 5000)))
    exclude = [c.strip() for c in request.form.get("exclude", "").split(",") if c.strip()]
    purpose = request.form.get("purpose", "balanced")
    generator = request.form.get("generator", "copula")

    use_llm = request.form.get("use_llm") == "1"
    user_rules = request.form.get("user_rules", "").strip()
    local_only = request.form.get("local_only") == "1"
    descriptions_only = request.form.get("descriptions_only") == "1"
    if local_only:                       # hard guarantee: nothing leaves the machine
        use_llm, user_rules = False, ""
    sample_n = 0 if descriptions_only else 8
    include_values = not descriptions_only
    try:
        real = pd.read_csv(csv_path)
    except Exception as e:
        return page(f'<h1>Could not read that CSV.</h1><p class="sub">{_html.escape(str(e))}</p>'
                    '<p><a class="btn ghost" href="/">Back</a></p>')
    real = real.drop(columns=[c for c in exclude if c in real.columns], errors="ignore")
    if real.empty or real.shape[1] == 0:
        return page('<h1>That file has no usable rows or columns.</h1>'
                    '<p><a class="btn ghost" href="/">Back</a></p>')
    rules, _ = rd.elicit_rules(real, use_llm=use_llm, sample_n=sample_n, include_values=include_values)
    if user_rules:
        rules = rules + rd.parse_user_rules(real, user_rules, include_values=include_values)
    target = request.form.get("target", "").strip() or None
    (d / "meta.json").write_text(json.dumps({"rows": rows, "exclude": exclude, "purpose": purpose,
                                             "generator": generator, "target": target,
                                             "name": f.filename, "cols": list(real.columns)}))
    (d / "rules.json").write_text(json.dumps([{**vars(r)} for r in rules], default=str))

    esc = _html.escape
    semantic = [r for r in rules if r.kind in ("fd", "sentinel_implies", "denial")]
    other = [r for r in rules if r.kind in ("range", "category_domain")]

    def rule_row(r, checked=True):
        src = getattr(r, "source", "mining")
        if src == "llm":
            badge = ' <span class="k" style="color:var(--warn);background:var(--leak-soft,#f0e2c6)">AI</span>'
        elif src == "typed":
            badge = ' <span class="k">TYPED</span>'
        else:
            badge = ""
        return f"""<label class="rule"><input type="checkbox" name="approve" value="{esc(r.id)}" {'checked' if checked else ''}>
<div><div><span class="k">{esc(r.kind)}</span>{badge} &nbsp;{esc(r.description)}</div>
<div class="meta">confidence {r.confidence} · support {r.support} rows · {esc(getattr(r, 'source', 'mining'))}</div></div></label>"""

    sem_html = "".join(rule_row(r) for r in semantic) or '<p class="sub">No cross-field rules found for this schema.</p>'
    oth_html = "".join(rule_row(r) for r in other)

    return page(f"""
<p class="eyebrow">Step 2 · review &amp; approve</p>
<h1>Verity found {len(rules)} candidate rules</h1>
<p class="sub">From <b>{esc(f.filename)}</b> ({len(real)} rows, {real.shape[1]} fields). Untick any rule you don't
want enforced — you are the human in the loop. Approved rules will be guaranteed in the synthetic data.</p>
<form method="post" action="/generate">
  <input type="hidden" name="token" value="{token}">
  <div class="actions"><button type="button" class="btn ghost" onclick="all(true)">Select all</button>
    <button type="button" class="btn ghost" onclick="all(false)">Select none</button></div>
  <h2>Cross-field logic ({len(semantic)})</h2>{sem_html}
  <h2>Domains &amp; ranges ({len(other)})</h2>{oth_html}
  <div class="actions"><button class="btn" type="submit">Generate {rows} rows &rarr;</button>
    <a class="btn ghost" href="/">Start over</a></div>
</form>
<script>function all(v){{document.querySelectorAll('input[name=approve]').forEach(c=>c.checked=v)}}</script>
""", "Verity · review rules")


@app.route("/generate", methods=["POST"])
def generate():
    token = request.form.get("token", "")
    d = SESS / token
    if not (d / "input.csv").exists():
        abort(404)
    meta = json.loads((d / "meta.json").read_text())
    approved = set(request.form.getlist("approve"))
    all_rules = [rd.CandidateRule(**{k: v for k, v in r.items()})
                 for r in json.loads((d / "rules.json").read_text())]
    rules = [r for r in all_rules if r.id in approved]

    real = pd.read_csv(d / "input.csv")
    real = real.drop(columns=[c for c in meta["exclude"] if c in real.columns], errors="ignore")
    profile = pp.get(meta.get("purpose"))
    backend = meta.get("generator", "auto")
    if backend in (None, "auto"):
        backend = profile.generator
    repaired, m = verity.generate(real, meta["rows"], rules, backend=backend)
    repaired.to_csv(d / "synthetic.csv", index=False)

    tgt = meta.get("target") or verity.auto_target(real)
    util = verity.utility_report(real, repaired, tgt) if tgt else None
    verdict = pp.evaluate(profile, real, repaired, m)
    esc = _html.escape
    checks = "".join(
        f'<li>{"✓" if c["ok"] else "✗"} {esc(c["label"])}</li>' for c in verdict["checks"])
    ok_color = "var(--good)" if verdict["passed"] else "var(--warn)"
    banner = (f'<div class="card" style="margin:0 0 8px">'
              f'<div class="lbl">Purpose · {esc(profile.label)}</div>'
              f'<div style="font-size:14px;margin:4px 0 6px">{esc(profile.goal)}</div>'
              f'<div style="font-size:14px">Verdict: <b style="color:{ok_color}">'
              f'{"PASS" if verdict["passed"] else "REVIEW"}</b></div>'
              f'<ul style="font-size:13px;margin:8px 0 0;padding-left:18px">{checks}</ul></div>')
    bar = (f'<div class="actions"><a class="btn" href="/download/{token}">Download synthetic CSV &darr;</a>'
           f'<a class="btn ghost" href="/">New dataset</a></div>')
    html = verity.build_report_html(meta["name"], meta["rows"], m["fidelity"], m["privacy"],
                                    m["validity"], rules, repaired, extra_top=banner + bar, util=util)
    return html


@app.route("/download/<token>")
def download(token):
    p = SESS / token / "synthetic.csv"
    if not p.exists():
        abort(404)
    name = json.loads((SESS / token / "meta.json").read_text()).get("name", "data.csv")
    return send_file(p, as_attachment=True, download_name=Path(name).stem + "_synthetic.csv")


if __name__ == "__main__":
    print("\n  Verity web app  ->  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
