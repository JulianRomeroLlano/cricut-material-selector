"""
import_custom_materials.py — Promote browser-exported custom materials into the
training dataset (Material List (Combined).csv).

Usage:
  source venv/bin/activate
  python scripts/import_custom_materials.py path/to/cricut_custom_materials.json

Workflow per material:
  1. Infer missing physics (GSM, Density, Shore Hardness A, Surface Texture,
     Has Adhesive) from category medians/mode of existing rows.
  2. Show the proposed CSV row.
  3. Prompt: [A]ccept / [E]dit field / [S]kip.
  4. Append accepted rows; warn on duplicates.

Run build_materials_json.py then retrain after importing.
"""
import sys, json, re
import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
CSV_SRC = ROOT / "assets" / "data" / "Material List (Combined).csv"

# ── Display-label → CSV raw multicut ─────────────────────────────────────────
# MC_LABELS in app.js:  ["1×","2×","3×","4–5×","6–8×","10+×"]
# Bucket ranges → pick the lower bound as a safe default (user can edit).
MCUT_LABEL_TO_RAW = {
    "1×":   "-",
    "2×":   "2倍",
    "3×":   "3倍",
    "4–5×": "4倍",   # lower bound of bucket
    "6–8×": "6倍",
    "10+×": "10倍",
    # also accept raw values passed through unchanged
    "-": "-",
}
# Build inverse for raw → display (used in duplicate check messaging)
for k, v in list(MCUT_LABEL_TO_RAW.items()):
    MCUT_LABEL_TO_RAW.setdefault(v, v)

BLADE_EN_TO_JP = {
    "Fine-Point Blade":    "ファインポイントブレード",
    "Deep-Point Blade":    "ディープポイントブレード",
    "Rotary Blade":        "ロータリーブレード",
    "Bonded Fabric Blade": "ボンデッドファブリックブレード",
    "Knife Blade":         "ナイフの刃",
}

CATEGORIES_WITH_ADHESIVE = {"Iron-On", "Vinyl"}  # default Has Adhesive = 1

# ── ANSI colours (degrade gracefully on Windows) ──────────────────────────────
try:
    import os; os.get_terminal_size()
    GRN, YEL, CYN, RST = "\033[32m", "\033[33m", "\033[36m", "\033[0m"
    BLD = "\033[1m"
except Exception:
    GRN = YEL = CYN = RST = BLD = ""


def load_csv():
    df = pd.read_csv(CSV_SRC, encoding="utf-8-sig")
    return df


def physics_defaults(df: pd.DataFrame, category: str) -> dict:
    """Compute median GSM, Density, Shore and mode Surface Texture for a category."""
    cat_rows = df[df["Category"] == category]
    if cat_rows.empty:
        cat_rows = df  # fallback to global

    def safe_median(col):
        if col not in cat_rows.columns:
            return np.nan
        vals = pd.to_numeric(cat_rows[col], errors="coerce").dropna()
        return round(float(vals.median()), 1) if len(vals) else np.nan

    def safe_mode(col):
        if col not in cat_rows.columns:
            return "plain"
        vals = cat_rows[col].dropna()
        return str(vals.mode().iloc[0]) if len(vals) else "plain"

    return {
        "GSM":              safe_median("GSM"),
        "Density (kg/m3)":  safe_median("Density (kg/m3)"),
        "Shore Hardness A": safe_median("Shore Hardness A"),
        "Surface Texture":  safe_mode("Surface Texture"),
        "Has Adhesive":     1 if category in CATEGORIES_WITH_ADHESIVE else 0,
    }


def is_duplicate(df: pd.DataFrame, name_en: str, category: str,
                 machine: str, thickness_mm: float) -> bool:
    mask = (
        (df["Category"] == category) &
        (df["Machine"] == machine) &
        (df["Material Name (EN)"].str.strip().str.lower() == name_en.strip().lower())
    )
    return bool(mask.any())


def prompt_edit(row: dict) -> dict:
    """Let the user edit individual fields interactively."""
    fields = list(row.keys())
    print(f"\n{CYN}Fields:{RST}")
    for i, k in enumerate(fields):
        print(f"  {i+1}. {k}: {BLD}{row[k]}{RST}")
    while True:
        raw = input("  Enter field number to edit (or blank to finish): ").strip()
        if not raw:
            break
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(fields):
                k = fields[idx]
                new_val = input(f"  New value for '{k}' [{row[k]}]: ").strip()
                if new_val:
                    # preserve numeric types where applicable
                    if k in ("Cutting Pressure",):
                        row[k] = int(new_val)
                    elif k in ("GSM", "Density (kg/m3)", "Shore Hardness A", "thickness_mm"):
                        row[k] = float(new_val)
                    elif k == "Has Adhesive":
                        row[k] = int(new_val)
                    else:
                        row[k] = new_val
        except (ValueError, IndexError):
            print("  Invalid choice.")
    return row


def process_material(entry: dict, df: pd.DataFrame) -> dict | None:
    """Interactive review of one custom material. Returns CSV row dict or None to skip."""
    print(f"\n{'─'*60}")
    print(f"{BLD}{CYN}Material:{RST} {entry.get('name_en', '?')}  "
          f"[{entry.get('machine', '?')} / {entry.get('category', '?')}]")

    machine   = str(entry.get("machine", "")).strip()
    category  = str(entry.get("category", "")).strip()
    name_en   = str(entry.get("name_en", "")).strip()
    name_jp   = str(entry.get("name_jp", name_en)).strip()
    pressure  = int(entry.get("pressure", 0))
    thickness = float(entry.get("thickness_mm", 0.5))
    blade_en  = str(entry.get("blade_en", "Fine-Point Blade")).strip()
    blade_jp  = BLADE_EN_TO_JP.get(blade_en, entry.get("blade_jp", blade_en))
    multicut_label = str(entry.get("multicut", "1×")).strip()
    multicut_raw   = MCUT_LABEL_TO_RAW.get(multicut_label, multicut_label)

    # Duplicate check
    if is_duplicate(df, name_en, category, machine, thickness):
        print(f"{YEL}  ⚠ Duplicate — already in CSV (same name, category, machine). Skipping.{RST}")
        return None

    # Infer missing physics
    phys = physics_defaults(df, category)

    row = {
        "Machine":            machine,
        "Category":           category,
        "Material Name (JP)": name_jp,
        "Material Name (EN)": name_en,
        "Cutting Pressure":   pressure,
        "Multi-Cut":          multicut_raw,
        "Blade Type":         blade_jp,
        "GSM":                phys["GSM"],
        "Surface Texture":    phys["Surface Texture"],
        "Has Adhesive":       phys["Has Adhesive"],
        "Density (kg/m3)":    phys["Density (kg/m3)"],
        "Shore Hardness A":   phys["Shore Hardness A"],
        "thickness_mm":       thickness,   # explicit — prevents infer_thickness guessing wrong default
    }

    # Show proposed row
    print(f"\n{GRN}Proposed CSV row:{RST}")
    for k, v in row.items():
        src = ""
        if k in ("GSM", "Density (kg/m3)", "Shore Hardness A", "Surface Texture", "Has Adhesive"):
            src = f"  {YEL}← inferred from category median{RST}"
        print(f"  {k}: {BLD}{v}{RST}{src}")

    # Prompt
    while True:
        choice = input(f"\n{BLD}[A]ccept  [E]dit  [S]kip:{RST} ").strip().lower()
        if choice in ("a", ""):
            return row
        elif choice == "e":
            row = prompt_edit(row)
            return row
        elif choice == "s":
            print("  Skipped.")
            return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/cricut_custom_materials.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        print("JSON must be an array of custom material objects.")
        sys.exit(1)

    if not entries:
        print("No custom materials found in the file.")
        sys.exit(0)

    df = load_csv()
    print(f"{GRN}Loaded CSV:{RST} {len(df)} rows  |  "
          f"{GRN}Custom materials in file:{RST} {len(entries)}")

    accepted = []
    for entry in entries:
        result = process_material(entry, df)
        if result:
            accepted.append(result)

    if not accepted:
        print(f"\n{YEL}No materials accepted. CSV unchanged.{RST}")
        sys.exit(0)

    print(f"\n{'═'*60}")
    print(f"{BLD}{GRN}Appending {len(accepted)} material(s) to CSV…{RST}")

    new_rows = pd.DataFrame(accepted)
    df_updated = pd.concat([df, new_rows], ignore_index=True)

    # Write back without BOM (matches original encoding)
    raw = df_updated.to_csv(index=False, encoding="utf-8-sig")
    # Strip BOM so the file stays consistent with the rest of the repo
    with open(CSV_SRC, "wb") as f:
        f.write(raw.encode("utf-8-sig"))

    print(f"✓ CSV updated: {len(df)} → {len(df_updated)} rows  ({CSV_SRC})")
    print(f"\n{CYN}Next steps:{RST}")
    print("  1. source venv/bin/activate")
    print("  2. python scripts/build_materials_json.py")
    print("  3. python scripts/train_model_v3.py")
    print("  4. cp assets/model/preprocessor_v3.json assets/model/preprocessor.json")
    print("  5. git add, commit, push")


if __name__ == "__main__":
    main()
