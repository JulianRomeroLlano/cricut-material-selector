"""
harmonize_materials.py — make each material name's physics identical across rows.

Why: predict.js feeds the model ONE physics set per name (material_lookup =
per-name medians of GSM / density / shore / texture / adhesive) plus the
thickness the user enters. Training used per-row values, so any name whose rows
disagreed on category, GSM, density or shore was trained on inputs the site
never produces (e.g. "Felt, Wool Bonded": Fabric/150 gsm on Joy Xtra vs
Others/200 gsm on Explore 3 / Maker 3).

What it does (AI-predicted rows only — Smart materials and Pens untouched):
  1. Category: rows of the names in CATEGORY_FIX are set to one category.
  2. GSM / Density / Shore: for every name whose rows disagree, all rows get the
     median of the rows already in the (harmonised) category. Thickness variants
     (Magnetic Sheet 0.5/1.0 mm, Copy Paper 0.075/0.09/0.12 mm …) keep their
     explicit thickness_mm — that is the feature the site can actually vary.
  3. thickness_mm: names mixing blank and explicit values get their blank rows
     filled with the value the pipeline already inferred (category default), so
     the live test / site pass the same thickness training saw.

Run: source venv/bin/activate && python scripts/harmonize_materials.py [--dry-run]
"""
import os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV  = os.path.join(ROOT, "assets", "data", "Material List (Combined).csv")
DRY  = "--dry-run" in sys.argv

# Chosen by majority across machines; ties resolved by product family.
CATEGORY_FIX = {
    "Faux Suede":                       "Leather",
    "Felt, Wool Bonded":                "Fabric",
    "Flexible Stencil Film":            "Vinyl",
    "Printable Iron-On, Dark":          "Iron-On",
    "Printable Iron-On, Light":         "Iron-On",
    "Printable Magnetic Sheet":         "Printable Materials",
    "Printable Temporary Tattoo Paper": "Printable Materials",
    "Printable Vinyl, Specialty (Gold/Silver)":                 "Printable Materials",
    "Printable Vinyl, Transparent":                             "Printable Materials",
    "Printable Vinyl, White (Green Liner Printing)":            "Printable Materials",
    "Printable Waterproof Sticker Set - Transparent":           "Printable Materials",
    "Printable Waterproof Sticker Set - Transparent Holographic": "Printable Materials",
    "Printable Waterproof Sticker Set - White":                 "Printable Materials",
    "Printable Waterproof Sticker Set - White Holographic":     "Printable Materials",
    "Vellum":                           "Paper",
    "Washi Tape":                       "Paper",
    "Watercolor Cards":                 "Cardstock",
}
THICKNESS_DEFAULTS = {   # must match train_model_v3.py / build_materials_json.py
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10, "Vinyl": 0.08,
    "Smart Materials": 0.10, "Printable Materials": 0.12, "Infusible Ink": 0.10,
    "Board/Cardboard": 1.0, "Leather": 1.6, "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
}
NUMERIC = ["GSM", "Density (kg/m3)", "Shore Hardness A"]

def infer_thickness(name, category):
    m = re.search(r"(\d+\.?\d*)\s*mm", name, re.IGNORECASE)
    if m: return float(m.group(1))
    return THICKNESS_DEFAULTS.get(category, 0.5)

df = pd.read_csv(CSV, encoding="utf-8-sig")
orig = df.copy()
name = df["Material Name (EN)"].str.strip()
ai = df["Cutting Pressure"].notna() & (df["Category"] != "Pens & Markers") & ~name.str.startswith("Smart")
changes = []

# 1. category
for n, cat in CATEGORY_FIX.items():
    m = ai & (name == n) & (df["Category"] != cat)
    for i in df.index[m]:
        changes.append((n, df.at[i, "Machine"], "Category", df.at[i, "Category"], cat))
    df.loc[m, "Category"] = cat

# 2. numeric physics — one value per name
for n, idx in df[ai].groupby(name[ai]).groups.items():
    sub = df.loc[idx]
    cat = CATEGORY_FIX.get(n)
    # reference rows = those that were ALREADY in the chosen category before step 1
    ref = sub[orig.loc[idx, "Category"] == cat] if cat else sub
    for col in NUMERIC:
        if sub[col].nunique() > 1:
            val = float(np.median(ref[col].values))
            for i in idx:
                if df.at[i, col] != val:
                    changes.append((n, df.at[i, "Machine"], col, df.at[i, col], val))
            df.loc[idx, col] = val

# 3. thickness fill for mixed blank/explicit names
for n, idx in df[ai].groupby(name[ai]).groups.items():
    th = df.loc[idx, "thickness_mm"]
    if th.isna().any() and th.notna().any():
        for i in idx:
            if pd.isna(df.at[i, "thickness_mm"]):
                val = infer_thickness(n, df.at[i, "Category"])
                changes.append((n, df.at[i, "Machine"], "thickness_mm", np.nan, val))
                df.at[i, "thickness_mm"] = val

# report
print(f"{len(changes)} cell changes across {len({c[0] for c in changes})} names:")
cur = None
for n, mach, col, old, new in changes:
    if n != cur:
        print(f"\n  {n}"); cur = n
    print(f"     {mach:<16} {col:<17} {old!s:>22} → {new}")

# sanity: every AI name now has a single value per physics column
name2 = df["Material Name (EN)"].str.strip()
for col in ["Category"] + NUMERIC:
    bad = [n for n, g in df[ai].groupby(name2[ai]) if g[col].nunique() > 1]
    assert not bad, (col, bad)
assert len(df) == len(orig) and list(df.columns) == list(orig.columns)
unchanged_cols = [c for c in df.columns if c not in ["Category", "thickness_mm"] + NUMERIC]
assert df[unchanged_cols].equals(orig[unchanged_cols]), "non-physics columns must not change"

if DRY:
    print("\n--dry-run: CSV not written")
else:
    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {os.path.relpath(CSV, ROOT)}")
