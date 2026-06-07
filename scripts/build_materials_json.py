"""
build_materials_json.py — Convert combined CSV to assets/data/materials.json.

Produces a compact JSON array where every entry has:
  machine, category, name_en, name_jp, pressure (int), multicut, blade_en, blade_jp

Run with:  source venv/bin/activate && python scripts/build_materials_json.py
"""
import os, json
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
    out.append({
        "machine":   str(row["Machine"]).strip(),
        "category":  str(row["Category"]).strip(),
        "name_en":   str(row["Material Name (EN)"]).strip(),
        "name_jp":   str(row["Material Name (JP)"]).strip(),
        "pressure":  int(row["Cutting Pressure"]),
        "multicut":  mc_disp,
        "blade_en":  blade_en,
        "blade_jp":  blade_jp,
    })

with open(DST, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

print(f"✓ {len(out)} materials → {DST}  ({os.path.getsize(DST)//1024} KB)")