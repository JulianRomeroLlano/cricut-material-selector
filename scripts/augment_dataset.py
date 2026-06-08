"""
augment_dataset.py — Material Name Normalization + GSM Augmentation

For each original row:
  1. Normalize base name: strip trailing parenthetical weight/gsm/mode suffixes
     e.g. "Value Cardstock - 65 lb (176gsm)" → "Value Cardstock - 65 lb"
          "Copy Paper (20lb/75gsm)"           → "Copy Paper"
  2. Generate 21 augmented samples at GSM × [0.90, 0.91, ..., 1.10] (1% steps)
     Pressure scaled proportionally: p_aug = p_orig × (gsm_aug / gsm_orig)
     All other fields (blade, multicut, machine, category, texture, etc.) unchanged.

Output: assets/data/Material List (Augmented).csv

Run: source venv/bin/activate && python scripts/augment_dataset.py
"""
import os, re, csv
import pandas as pd
import numpy as np

ROOT    = os.path.join(os.path.dirname(__file__), "..")
IN_CSV  = os.path.join(ROOT, "assets", "data", "Material List (Combined).csv")
OUT_CSV = os.path.join(ROOT, "assets", "data", "Material List (Augmented).csv")

AUG_FACTORS = [round(0.90 + i * 0.01, 2) for i in range(21)]  # 0.90 … 1.10


# Only strip parentheticals that are purely measurement/weight info (digits + units).
# Descriptive parentheticals like "(Mosaic)", "(Green Liner)", "(Intricate Cuts)" are kept.
_MEAS_PAREN = re.compile(
    r'''\s*\(\s*
        \d                                     # must start with a digit
        [\d\s./\-]*                            # more digits / separators
        (?:gsm|lbs?|oz\.?|mm|cm|              # optional primary unit
           inch(?:es)?|gauge)?
        (?:\s*/\s*                             # optional second measurement
           [\d\s./\-]+
           (?:gsm|lbs?|oz\.?|mm|cm|
              inch(?:es)?|gauge)?
        )?
        \s*\)\s*$''',
    re.IGNORECASE | re.VERBOSE,
)

def normalize_name(name: str) -> str:
    """Strip trailing measurement-only parentheticals, keep descriptive ones."""
    cleaned = _MEAS_PAREN.sub('', name.strip()).strip()
    return cleaned or name.strip()


def main():
    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {os.path.basename(IN_CSV)}")

    # Drop rows without pressure
    df = df[df["Cutting Pressure"].notna()].copy()
    df["Cutting Pressure"] = df["Cutting Pressure"].astype(float)
    df["GSM"] = pd.to_numeric(df["GSM"], errors="coerce").fillna(100.0)
    print(f"  After cleaning: {len(df)} rows")

    # Add normalized base name column
    df["Material Name Base"] = df["Material Name (EN)"].apply(normalize_name)

    # Show normalization stats
    changed = (df["Material Name Base"] != df["Material Name (EN)"]).sum()
    print(f"  Names normalized: {changed} rows changed")
    print(f"  Unique base names: {df['Material Name Base'].nunique()}")

    # Generate augmented rows
    records = []
    for _, row in df.iterrows():
        gsm_orig = float(row["GSM"])
        p_orig   = float(row["Cutting Pressure"])
        if gsm_orig <= 0:
            gsm_orig = 100.0  # safety fallback

        for factor in AUG_FACTORS:
            gsm_aug = gsm_orig * factor
            p_aug   = p_orig * factor  # linear pressure scaling with weight

            rec = row.to_dict()
            rec["GSM"]              = round(gsm_aug, 4)
            rec["Cutting Pressure"] = round(p_aug, 2)
            rec["aug_factor"]       = factor
            records.append(rec)

    out_df = pd.DataFrame(records)
    print(f"\nAugmented dataset: {len(out_df)} rows  ({len(df)} × {len(AUG_FACTORS)} = {len(df)*len(AUG_FACTORS)})")
    print(f"  Machines: {dict(out_df['Machine'].value_counts())}")
    print(f"  GSM range: {out_df['GSM'].min():.1f} – {out_df['GSM'].max():.1f}")
    print(f"  Pressure range: {out_df['Cutting Pressure'].min():.1f} – {out_df['Cutting Pressure'].max():.1f}")

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {os.path.basename(OUT_CSV)}")

    # Print a few examples
    print("\nExample augmented entries for 'Value Cardstock - 65 lb (176gsm)':")
    sample = out_df[out_df["Material Name (EN)"].str.contains("Value Cardstock - 65 lb", na=False)
                    & (out_df["Machine"] == "Maker 3")].head(5)
    for _, r in sample.iterrows():
        print(f"  base={r['Material Name Base']!r}  gsm={r['GSM']:.1f}  pressure={r['Cutting Pressure']:.1f}  factor={r['aug_factor']}")


if __name__ == "__main__":
    main()
