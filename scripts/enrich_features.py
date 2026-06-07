"""
enrich_features.py — Add 5 physical property columns to material CSVs.

New columns:
  GSM              — grammage (g/m²), float
  Surface Texture  — plain/matte/glossy/shimmer/foil/metallic/holographic/glitter/flock/textured, str
  Has Adhesive     — 0 or 1 (int)
  Density (kg/m3)  — float
  Shore Hardness A — float (Shore A equivalent, 0-100 scale)

Run with:  source venv/bin/activate && python scripts/enrich_features.py
"""
import os, re
import pandas as pd

ROOT     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "assets", "data")

COMBINED_CSV   = os.path.join(DATA_DIR, "Material List (Combined).csv")
MACHINE_CSVS = [
    os.path.join(DATA_DIR, "Material List (Settings) - Cricut Joy 2.csv"),
    os.path.join(DATA_DIR, "Material List (Settings) - Cricut Joy Xtra.csv"),
]

NEW_COLS = ["GSM", "Surface Texture", "Has Adhesive", "Density (kg/m3)", "Shore Hardness A"]

# ─── Reference tables ─────────────────────────────────────────────────────────

GSM_DEFAULTS = {
    "Paper": 80,
    "Cardstock": 216,
    "Iron-On": 100,
    "Vinyl": 120,
    "Smart Materials": 120,
    "Infusible Ink": 75,
    "Printable Materials": 100,
    "Board/Cardboard": None,   # computed from thickness × density
    "Leather": None,           # computed from thickness × density
    "Fabric": 150,
    "Plastic": 150,            # typical 100µm PET/acetate film
    "Others": 200,
}

DENSITY_DEFAULTS = {
    "Paper": 750,
    "Cardstock": 800,
    "Iron-On": 1050,
    "Vinyl": 1300,
    "Smart Materials": 1300,
    "Infusible Ink": 800,
    "Printable Materials": 900,
    "Board/Cardboard": 850,
    "Leather": 900,
    "Fabric": 280,
    "Plastic": 1350,           # PET/acetate density
    "Others": 150,
}

# Shore A equivalent defaults per category
SHORE_DEFAULTS = {
    "Paper": 15,
    "Cardstock": 30,
    "Iron-On": 45,
    "Vinyl": 65,
    "Smart Materials": 65,
    "Infusible Ink": 20,
    "Printable Materials": 25,
    "Board/Cardboard": 65,
    "Leather": 55,
    "Fabric": 10,
    "Plastic": 70,             # stiff plastic film
    "Others": 40,
}

# Surface texture keywords — matched in order (most specific first to avoid partial matches)
# Each entry: (keyword, texture_label, shore_a_bonus, density_multiplier)
SURFACE_KEYWORDS = [
    ("glitter cardstock",  "glitter",     18, 1.15),
    ("glitter iron",       "glitter",     18, 1.12),
    ("glitter vinyl",      "glitter",     18, 1.15),
    ("glitter",            "glitter",     16, 1.12),
    ("flocked",            "flock",       14, 1.05),
    ("flock",              "flock",       14, 1.05),
    ("foil poster",        "foil",        10, 1.05),
    ("foil iron",          "foil",        10, 1.00),
    ("foil",               "foil",        10, 1.05),
    ("metallic iron",      "metallic",    10, 1.05),
    ("metallic",           "metallic",    10, 1.05),
    ("chrome",             "metallic",    12, 1.10),
    ("mirror",             "metallic",    10, 1.08),
    ("holographic",        "holographic",  8, 1.05),
    ("iridescent",         "holographic",  8, 1.05),
    ("shimmer",            "shimmer",      6, 1.03),
    ("pearl",              "shimmer",      5, 1.02),
    ("matte",              "matte",        2, 0.98),
    ("satin",              "matte",        2, 0.99),
    ("glossy",             "glossy",       4, 1.02),
    ("gloss",              "glossy",       4, 1.02),
    ("textured",           "textured",     8, 1.05),
]

ADHESIVE_KEYWORDS = [
    "adhesive", "sticker", "self-adhesive", "self adhesive",
    "peel & stick", "peel and stick", "(adhesive)", "backed",
]

# ─── Inference functions ───────────────────────────────────────────────────────

def _lb_to_gsm(lb: float) -> float:
    """Convert US pound basis weight to g/m² (for 8.5×11 bond/text sheet basis)."""
    # Cardstock basis: 1 lb text ≈ 1.48 gsm; bond: 1 lb ≈ 3.76 gsm
    # We use cardstock (cover stock basis) conversion: ~2.71× is common in literature
    return lb * 3.76  # conservative: most Cricut cardstock uses bond basis


def _oz_to_mm(name: str) -> float | None:
    m = re.search(r"(\d+)[-–]?(\d+)?\s*oz", name, re.IGNORECASE)
    if m:
        oz = (float(m.group(1)) + float(m.group(2) or m.group(1))) / 2
        return oz * 0.40  # 1 oz leather ≈ 0.40 mm
    return None


def _mm_from_name(name: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s*mm", name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def infer_gsm(name: str, category: str) -> float:
    n = name.lower()
    # Explicit gsm in name
    m = re.search(r"(\d+\.?\d*)\s*g/?m[²2]?", n)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*gsm", n)
    if m:
        return float(m.group(1))
    # lb/oz weight
    m = re.search(r"(\d+)\s*lb", n)
    if m:
        return _lb_to_gsm(float(m.group(1)))
    # For board/cardboard and leather: estimate from thickness × density
    if category in ("Board/Cardboard", "Leather"):
        mm = _mm_from_name(name)
        if mm is None and category == "Leather":
            mm = _oz_to_mm(name)
        mm = mm or (1.0 if category == "Board/Cardboard" else 1.6)
        density = DENSITY_DEFAULTS.get(category, 850)
        return round(mm * density, 1)  # mm × kg/m³ → g/m²  (density×thickness_m×1000 = density×thickness_mm)
    return float(GSM_DEFAULTS.get(category, 100))


def infer_surface_texture(name: str) -> tuple[str, int, float]:
    """Return (texture_label, shore_bonus, density_multiplier)."""
    n = name.lower()
    for kw, label, shore_bonus, dens_mult in SURFACE_KEYWORDS:
        if kw in n:
            return label, shore_bonus, dens_mult
    return "plain", 0, 1.0


def infer_has_adhesive(name: str) -> int:
    n = name.lower()
    for kw in ADHESIVE_KEYWORDS:
        if kw in n:
            return 1
    return 0


def infer_density(name: str, category: str, texture_label: str, density_mult: float) -> float:
    base = float(DENSITY_DEFAULTS.get(category, 500))
    # Override for specific sub-types detectable from name
    n = name.lower()
    if "basswood" in n:
        base = 400.0
    elif "balsa" in n:
        base = 150.0
    elif "wood veneer" in n:
        base = 350.0
    elif "rubber" in n:
        base = 1200.0
    elif "magnetic" in n:
        base = 3500.0
    elif "aluminum" in n or "aluminium" in n:
        base = 2700.0
    elif "acetate" in n or "transparency" in n:
        base = 1300.0
    elif "eva foam" in n or "craft foam" in n or "foam" in n:
        base = 100.0
    elif "felt" in n:
        base = 300.0
    elif "kraft" in n and category == "Board/Cardboard":
        base = 900.0
    return round(base * density_mult, 1)


def infer_shore_hardness(name: str, category: str, shore_bonus: int) -> float:
    base = float(SHORE_DEFAULTS.get(category, 40))
    n = name.lower()
    # Specific material overrides
    if "basswood" in n:
        base = 97.0
    elif "balsa" in n:
        base = 60.0
    elif "wood veneer" in n:
        base = 85.0
    elif "magnetic" in n:
        base = 90.0
    elif "rubber" in n:
        base = 50.0
    elif "matboard" in n or "mat board" in n:
        base = 80.0
    elif "chipboard" in n:
        base = 72.0
    elif "corrugated" in n:
        base = 55.0
    elif "acetate" in n or "transparency" in n:
        base = 75.0
    elif "eva foam" in n or "craft foam" in n or "foam" in n:
        base = 42.0
    elif "felt" in n:
        base = 20.0
    elif "washi" in n:
        base = 10.0
    elif "vellum" in n:
        base = 12.0
    elif "tissue" in n:
        base = 8.0
    # Thickness-based adjustment for leather
    if category == "Leather":
        mm = _mm_from_name(name)
        if mm is None:
            mm = _oz_to_mm(name)
        if mm is not None:
            base = min(85.0, 45.0 + mm * 10.0)  # thin=55, thick=75
    # Thickness-based adjustment for board
    if category == "Board/Cardboard":
        mm = _mm_from_name(name)
        if mm is not None:
            base = min(90.0, 50.0 + mm * 15.0)
    return round(min(100.0, base + shore_bonus), 1)


# ─── Main enrichment ──────────────────────────────────────────────────────────

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop existing new columns if re-running
    for col in NEW_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    gsm_vals      = []
    texture_vals  = []
    adhesive_vals = []
    density_vals  = []
    shore_vals    = []

    for _, row in df.iterrows():
        name     = str(row.get("Material Name (EN)", ""))
        category = str(row.get("Category", "Others"))

        texture_label, shore_bonus, dens_mult = infer_surface_texture(name)

        gsm_vals.append(infer_gsm(name, category))
        texture_vals.append(texture_label)
        adhesive_vals.append(infer_has_adhesive(name))
        density_vals.append(infer_density(name, category, texture_label, dens_mult))
        shore_vals.append(infer_shore_hardness(name, category, shore_bonus))

    df["GSM"]              = gsm_vals
    df["Surface Texture"]  = texture_vals
    df["Has Adhesive"]     = adhesive_vals
    df["Density (kg/m3)"]  = density_vals
    df["Shore Hardness A"] = shore_vals

    return df


def process_csv(path: str, has_machine_col: bool = True) -> None:
    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)
    df_enriched = enrich_df(df)
    df_enriched.to_csv(path, index=False, encoding="utf-8-sig")

    # Summary
    print(f"\n{os.path.basename(path)} — {n_before} rows enriched")
    sample = df_enriched[["Material Name (EN)", "GSM", "Surface Texture",
                           "Has Adhesive", "Density (kg/m3)", "Shore Hardness A"]].head(8)
    print(sample.to_string(index=False))

    # Stats per category
    print("\n  GSM range by category:")
    for cat, grp in df_enriched.groupby("Category"):
        print(f"    {cat:<22}  GSM {grp['GSM'].min():.0f}–{grp['GSM'].max():.0f}"
              f"  density {grp['Density (kg/m3)'].min():.0f}–{grp['Density (kg/m3)'].max():.0f}"
              f"  shore {grp['Shore Hardness A'].min():.0f}–{grp['Shore Hardness A'].max():.0f}")

    # Surface texture counts
    print("\n  Surface texture distribution:")
    for tex, cnt in df_enriched["Surface Texture"].value_counts().items():
        print(f"    {tex:<15} {cnt:>3}")
    print(f"  Has adhesive: {df_enriched['Has Adhesive'].sum()} / {len(df_enriched)}")


if __name__ == "__main__":
    print("=== Enriching Combined CSV ===")
    process_csv(COMBINED_CSV, has_machine_col=True)

    for path in MACHINE_CSVS:
        print(f"\n=== Enriching {os.path.basename(path)} ===")
        process_csv(path, has_machine_col=False)

    print("\n✓ All CSVs enriched with 5 new columns.")