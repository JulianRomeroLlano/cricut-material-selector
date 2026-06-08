"""
build_materials_json.py — Convert combined CSV to assets/data/materials.json.

Produces a compact JSON array where every entry has:
  machine, category, name_en, name_jp, pressure (int), multicut,
  blade_en, blade_jp, thickness_mm (float, rounded to 2dp)

Run with:  source venv/bin/activate && python scripts/build_materials_json.py
"""
import os, json, re
import pandas as pd

ROOT     = os.path.join(os.path.dirname(__file__), "..")
SRC      = os.path.join(ROOT, "assets", "data", "Material List (Combined).csv")
DST      = os.path.join(ROOT, "assets", "data", "materials.json")

BLADE_JP_TO_EN = {
    "ディープポイントブレード":      "Deep-Point Blade",
    "ナイフの刃":                 "Knife Blade",
    "ファインポイントブレード":        "Fine-Point Blade",
    "ファインポイント":             "Fine-Point Blade",
    "ボンデッドファブリックブレード":   "Bonded Fabric Blade",
    "ロータリーブレード":            "Rotary Blade",
}

MCUT_DISPLAY = {
    "-":    "1×",
    "2倍":  "2×",  "3倍":  "3×",  "4倍":  "4×",  "5倍":  "5×",
    "6倍":  "6×",  "7倍":  "7×",  "8倍":  "8×",  "10倍": "10×",
    "12倍": "12×", "14倍": "14×", "16倍": "16×",
    "17倍": "17×", "18倍": "18×", "24倍": "24×",
}

THICKNESS_DEFAULTS = {
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10, "Vinyl": 0.08,
    "Smart Materials": 0.10, "Printable Materials": 0.12, "Infusible Ink": 0.10,
    "Board/Cardboard": 1.0, "Leather": 1.6, "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
}

def _lb_to_mm(lb):
    pts = [(60, 0.15), (65, 0.18), (80, 0.22), (100, 0.27), (140, 0.38)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        if x0 <= lb <= x1:
            return round(y0 + (y1-y0) * (lb-x0) / (x1-x0), 2)
    return 0.27 if lb > 100 else 0.15

def infer_thickness(name, category):
    m = re.search(r"(\d+\.?\d*)\s*mm", name, re.IGNORECASE)
    if m: return round(float(m.group(1)), 2)
    m = re.search(r"(\d+)\s*lb", name, re.IGNORECASE)
    if m: return _lb_to_mm(int(m.group(1)))
    m = re.search(r"(\d+)\s*gsm", name, re.IGNORECASE)
    if m: return round(max(0.04, int(m.group(1)) * 0.001), 2)
    return THICKNESS_DEFAULTS.get(category, 0.5)

df = pd.read_csv(SRC, encoding="utf-8-sig")
df = df[df["Category"] != "Pens & Markers"].copy()
df = df[df["Cutting Pressure"].notna()].copy()
df["Cutting Pressure"] = df["Cutting Pressure"].astype(float)
df["Blade Type"] = df["Blade Type"].replace("ファインポイント", "ファインポイントブレード")

out = []
for _, row in df.iterrows():
    blade_jp = str(row["Blade Type"]).strip()
    blade_en = BLADE_JP_TO_EN.get(blade_jp, blade_jp)
    mc_raw   = str(row["Multi-Cut"]).strip()
    mc_disp  = MCUT_DISPLAY.get(mc_raw, mc_raw)
    name_en  = str(row["Material Name (EN)"]).strip()
    category = str(row["Category"]).strip()
    out.append({
        "machine":      str(row["Machine"]).strip(),
        "category":     category,
        "name_en":      name_en,
        "name_jp":      str(row["Material Name (JP)"]).strip(),
        "pressure":     int(row["Cutting Pressure"]),
        "multicut":     mc_disp,
        "blade_en":     blade_en,
        "blade_jp":     blade_jp,
        "thickness_mm": infer_thickness(name_en, category),
    })

with open(DST, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

print(f"✓ {len(out)} materials → {DST}  ({os.path.getsize(DST)//1024} KB)")