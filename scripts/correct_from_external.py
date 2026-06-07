"""
correct_from_external.py
========================
1. Apply tamayuzucraft corrections (pressure / blade / multi-cut) to the CSV
   for the ~50 entries that differ between our data and the external source.
2. Normalize Joy blade name: "ファインポイント" → "ファインポイントブレード" everywhere.
3. Add a new "Plastic" category by reclassifying plastic-film materials that
   are currently scattered across "Others" and "Paper".
"""

import re, json, sys
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_CSV  = ROOT / "assets" / "data" / "Material List (Combined).csv"
EXT_HTML  = list(Path.home().joinpath(
    ".claude/projects/-home-julian-Projects-cricut-material-selector"
).rglob("bwcvji0m2.txt"))[0]

# ── External data parsing ──────────────────────────────────────────────────────
with open(EXT_HTML, encoding="utf-8") as f:
    html = f.read()
m = re.search(r"const DATA = (\[.*?\]);", html, re.DOTALL)
if not m:
    sys.exit("Could not find DATA array in external HTML")
ext = json.loads(m.group(1))

MACHINE_MAP = {"Joy": "Cricut Joy", "Explorer3": "Explore 3", "Maker3": "Maker 3"}
BLADE_MAP = {
    "-":                    "ファインポイントブレード",
    "ファインポイントブレード":  "ファインポイントブレード",
    "ディープポイントブレード":  "ディープポイントブレード",
    "ナイフブレード":          "ナイフの刃",
    "ロータリーブレード":       "ロータリーブレード",
    "該当なし":               None,
}
MCUT_MAP = {
    "Off": "-", "1x": "-", "2x": "2倍", "3x": "3倍", "3ｘ": "3倍",
    "4x": "4倍", "5x": "5倍", "6x": "6倍", "7x": "7倍", "8x": "8倍",
    "12x": "12倍", "14x": "14倍", "16x": "16倍", "17x": "17倍",
    "18x": "18倍", "24x": "24倍", "該当なし": None,
}

def extract_en(name: str) -> str:
    if " / " in name:
        return name.split(" / ")[-1].strip()
    last_jp = -1
    for i, c in enumerate(name):
        cp = ord(c)
        if (0x3000 <= cp <= 0x9FFF) or (0xFF00 <= cp <= 0xFFEF):
            last_jp = i
    return name[last_jp + 1:].strip() if last_jp >= 0 else name.strip()

# Build lookup: (machine_en, en_name_lower) → {pressure, blade, multicut}
ext_lookup: dict = {}
for d in ext:
    machine = MACHINE_MAP.get(d["machine"])
    if not machine:
        continue
    blade = BLADE_MAP.get(d["blade"])
    if blade is None:
        continue
    mcut = MCUT_MAP.get(d["multicut"])
    if mcut is None:
        continue
    try:
        pressure = float(d["pressure"])
    except (ValueError, TypeError):
        continue
    en = extract_en(d["name"])
    if not en:
        continue
    ext_lookup[(machine, en.lower())] = {
        "pressure": pressure,
        "blade":    blade,
        "multicut": mcut,
    }

# ── Load CSV ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
original_len = len(df)
print(f"Loaded {original_len} rows from {DATA_CSV.name}")

# ── Step 1: Global Joy blade normalisation ────────────────────────────────────
# Our Joy entries imported before the tamayuzucraft run use the short form.
# train_model.py already fixes this at runtime, but the source CSV should be clean.
joy_mask = df["Machine"].str.strip() == "Cricut Joy"
pre = (df.loc[joy_mask, "Blade Type"] == "ファインポイント").sum()
df.loc[joy_mask & (df["Blade Type"] == "ファインポイント"), "Blade Type"] = "ファインポイントブレード"
print(f"  Normalised Joy blade name in {pre} rows")

# ── Step 2: Apply tamayuzucraft corrections ───────────────────────────────────
n_pressure = n_blade = n_multicut = 0
conflict_log = []

for idx, row in df.iterrows():
    machine = str(row["Machine"]).strip()
    en_name = str(row["Material Name (EN)"]).strip()
    key = (machine, en_name.lower())
    if key not in ext_lookup:
        continue
    e = ext_lookup[key]

    old_p = float(row["Cutting Pressure"])
    old_b = str(row["Blade Type"]).strip()
    old_m = str(row["Multi-Cut"]).strip()

    changed = []
    if abs(old_p - e["pressure"]) > 0.5:
        df.at[idx, "Cutting Pressure"] = e["pressure"]
        changed.append(f"pressure {int(old_p)}→{int(e['pressure'])}")
        n_pressure += 1
    if old_b != e["blade"]:
        df.at[idx, "Blade Type"] = e["blade"]
        changed.append(f"blade {old_b}→{e['blade']}")
        n_blade += 1
    if old_m != e["multicut"]:
        df.at[idx, "Multi-Cut"] = e["multicut"]
        changed.append(f"multicut {old_m}→{e['multicut']}")
        n_multicut += 1
    if changed:
        conflict_log.append(f"  [{machine}] {en_name}: {' | '.join(changed)}")

print(f"\nApplied tamayuzucraft corrections:")
print(f"  {n_pressure} pressure changes, {n_blade} blade changes, {n_multicut} multi-cut changes")
if conflict_log:
    for line in conflict_log:
        print(line)

# ── Step 3: Reclassify plastic materials → "Plastic" category ─────────────────
# Matches on English material name (case-insensitive substring).
# We only reclassify from "Others" and "Paper" — not from clearly correct
# categories like Vinyl (Smart Stencil) or Smart Materials.
PLASTIC_PATTERNS = [
    "acetate",            # Acetate, Foil Acetate
    "mylar",              # Mylar (polyester film)
    "transparency",       # Transparency sheet
    "stencil film",       # Stencil Film (0.4mm), Flexible Stencil Film — in Others only
    "cutting mat protector",
    "plastic packaging",
    "plastic canvas",
    "window cling",       # Window Cling, Window Clinging, Window Crying (typo)
    "sandblasting stencil",
]

RECLASSIFY_FROM = {"Others", "Paper"}

n_plastic = 0
for idx, row in df.iterrows():
    if str(row["Category"]).strip() not in RECLASSIFY_FROM:
        continue
    name_lower = str(row["Material Name (EN)"]).lower()
    if any(pat in name_lower for pat in PLASTIC_PATTERNS):
        df.at[idx, "Category"] = "Plastic"
        n_plastic += 1

print(f"\nReclassified {n_plastic} rows → 'Plastic' category")

# Show what moved
plastic_rows = df[df["Category"] == "Plastic"][["Machine", "Material Name (EN)"]].drop_duplicates()
print("  Plastic entries now:")
for _, r in plastic_rows.iterrows():
    print(f"    [{r['Machine']}] {r['Material Name (EN)']}")

# ── Step 4: Category counts ────────────────────────────────────────────────────
print(f"\nCategory distribution (non-Pens):")
for cat, cnt in df[df["Category"] != "Pens & Markers"]["Category"].value_counts().items():
    print(f"  {cat:<25} {cnt}")

# ── Step 5: Save ───────────────────────────────────────────────────────────────
df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(df)} rows to {DATA_CSV.name}")
