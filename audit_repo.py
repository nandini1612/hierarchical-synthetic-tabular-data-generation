"""
audit_repo.py

Run this from the ROOT of the cloned hierarchical-synthetic-tabular repo,
AFTER you've run `python src/main.py prepare` at least once (so
data/processed/*.csv exist).

    python audit_repo.py > audit_output.txt

It does NOT modify anything. It only reads and prints. Paste the resulting
audit_output.txt back so integration code (the repair-module hook into
run_experiments.py, and the exact evaluate_utility/evaluate_fidelity calls)
can be written against your actual function signatures instead of the
README's documented-but-not-guaranteed-exact interface.

What it reports, in order:
  1. Repository file tree under src/ and configs/
  2. Full source of every .py file under src/ (so signatures, imports,
     and internal helper names are all visible at once)
  3. Contents of every .json file under configs/
  4. Column names + dtypes + a 5-row sample of every processed CSV under
     data/processed/ (this is what the constraint set needs to be built
     against precisely — real column names, real category strings)
"""

import inspect
import json
import os
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed"


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_tree(path: Path, label: str):
    section(f"FILE TREE: {label}")
    if not path.exists():
        print(f"  (missing: {path})")
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            print(" ", p.relative_to(ROOT))


def print_all_src_files():
    section("SOURCE: full contents of every src/*.py file")
    if not SRC.exists():
        print(f"  (missing: {SRC})")
        return
    for p in sorted(SRC.glob("*.py")):
        print(f"\n----- {p.relative_to(ROOT)} " + "-" * max(0, 60 - len(str(p))))
        try:
            print(p.read_text())
        except Exception as e:
            print(f"  [could not read: {e}]")


def print_all_configs():
    section("CONFIGS: contents of every configs/*.json file")
    if not CONFIGS.exists():
        print(f"  (missing: {CONFIGS})")
        return
    for p in sorted(CONFIGS.glob("*.json")):
        print(f"\n----- {p.relative_to(ROOT)} -----")
        try:
            print(json.dumps(json.loads(p.read_text()), indent=2))
        except Exception as e:
            print(f"  [could not parse: {e}]")
    for p in sorted(CONFIGS.glob("*.md")):
        print(f"\n----- {p.relative_to(ROOT)} -----")
        try:
            print(p.read_text())
        except Exception as e:
            print(f"  [could not read: {e}]")


def print_processed_schema():
    section("PROCESSED DATA SCHEMA: columns, dtypes, sample rows")
    if not PROCESSED.exists():
        print(f"  (missing: {PROCESSED} — run `python src/main.py prepare` first)")
        return
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed — cannot inspect CSV schema")
        return
    for p in sorted(PROCESSED.rglob("*.csv")):
        print(f"\n----- {p.relative_to(ROOT)} -----")
        try:
            df = pd.read_csv(p, nrows=2000)
            print(f"shape (first 2000 rows read): {df.shape}")
            print("\ndtypes:")
            print(df.dtypes.to_string())
            print("\nnunique per column:")
            print(df.nunique().to_string())
            print("\nsample rows:")
            print(df.head(5).to_string())
            # For likely categorical columns, show the actual category strings —
            # this is the part constraint-writing needs most and can't be guessed.
            print("\ncategory values for low-cardinality columns (<=15 uniques):")
            for col in df.columns:
                if df[col].nunique() <= 15:
                    print(
                        f"  {col}: {sorted(df[col].dropna().astype(str).unique().tolist())}"
                    )
        except Exception as e:
            print(f"  [could not read: {e}]")


def print_function_signatures():
    section("PYTHON-INTROSPECTED FUNCTION SIGNATURES (src/*.py)")
    if not SRC.exists():
        print(f"  (missing: {SRC})")
        return
    sys.path.insert(0, str(SRC))
    for p in sorted(SRC.glob("*.py")):
        mod_name = p.stem
        try:
            spec = importlib.util.spec_from_file_location(mod_name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"\n----- {p.name}: [import failed: {e}] -----")
            continue
        print(f"\n----- {p.name} -----")
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and obj.__module__ == mod_name:
                try:
                    print(f"  def {name}{inspect.signature(obj)}")
                except (ValueError, TypeError):
                    print(f"  def {name}(...)")


if __name__ == "__main__":
    print_tree(SRC, "src/")
    print_tree(CONFIGS, "configs/")
    print_all_configs()
    print_processed_schema()
    print_all_src_files()
    # Signature introspection last and best-effort — importing every module
    # can fail on missing optional deps (e.g. sdv); that's fine, the full
    # source dump above already gives everything needed even if this fails.
    try:
        print_function_signatures()
    except Exception as e:
        print(f"\n[signature introspection failed entirely: {e}]")
