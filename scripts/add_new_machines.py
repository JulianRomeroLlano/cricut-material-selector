"""
Parse manually entered cut settings for Cricut Joy 2 and Joy Xtra,
assign categories, create individual CSVs, and update the combined CSV.
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "data")
COMBINED_CSV = os.path.join(DATA_DIR, "Material List (Combined).csv")

# ── Raw data ──────────────────────────────────────────────────────────────────

JOY2_DATA = [
    ("3D Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Acetate", 250, "2x", "Fine-Point Blade"),
    ("Adhesive Foil", 90, "Off", "Fine-Point Blade"),
    ("Calibration Paper", 150, "Off", "Fine-Point Blade"),
    ("Chalkboard Vinyl", 180, "Off", "Fine-Point Blade"),
    ("Chameleon Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Color Pop Iron-On", 175, "Off", "Fine-Point Blade"),
    ("Construction Paper", 168, "Off", "Fine-Point Blade"),   # fixed typo "contruction"
    ("Copy Paper", 95, "Off", "Fine-Point Blade"),
    ("Cutaway Card + Backer", 150, "2x", "Fine-Point Blade"),
    ("Dry Erase Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Everyday Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Faux Leather (Paper Thin)", 120, "2x", "Fine-Point Blade"),
    ("Faux Suede", 90, "3x", "Fine-Point Blade"),
    ("Felt", 160, "2x", "Fine-Point Blade"),
    ("Flocked Iron-On", 120, "Off", "Fine-Point Blade"),
    ("Foil Iron-On", 90, "Off", "Fine-Point Blade"),
    ("Glitter Cardstock", 270, "2x", "Fine-Point Blade"),
    ("Glitter Iron-On", 180, "Off", "Fine-Point Blade"),
    ("Glow in the Dark Iron-On", 220, "Off", "Fine-Point Blade"),
    ("Glow in the Dark Vinyl", 145, "2x", "Fine-Point Blade"),
    ("Heat/Cold Activated Color Changing Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Heavy Cardstock - 100 lb (270 gsm)", 280, "2x", "Fine-Point Blade"),
    ("Heavy Watercolor Paper - 140 lbs (300 gsm)", 220, "2x", "Fine-Point Blade"),
    ("Holographic Iron-On", 120, "Off", "Fine-Point Blade"),
    ("Holographic Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Infusible Ink Transfer Sheet", 210, "Off", "Fine-Point Blade"),
    ("Insert Card - Cardstock", 150, "2x", "Fine-Point Blade"),
    ("Iron-On Patches", 160, "3x", "Fine-Point Blade"),
    ("Laminated Printable Sticker Paper, White", 250, "2x", "Fine-Point Blade"),
    ("Laser Copy Paper", 110, "Off", "Fine-Point Blade"),
    ("Medium Cardstock - 80 lb (216 gsm)", 210, "2x", "Fine-Point Blade"),
    ("Medium Cardstock - 80 lb (Basic Cuts, Single Pass)", 270, "Off", "Fine-Point Blade"),
    ("Metallic Puff Iron-On", 140, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Frosted Opaque", 110, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Holographic Sparkle", 100, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Holographic Threads", 100, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Pearl", 120, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Permanent", 100, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Removable", 100, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Shimmer", 110, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Textured Metallic", 110, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - True Brushed", 110, "Off", "Fine-Point Blade"),
    ("Printable Iron-On Glitter Kit", 230, "2x", "Fine-Point Blade"),
    ("Printable Iron-On Glow in the Dark Kit", 200, "2x", "Fine-Point Blade"),
    ("Printable Iron-On Holographic Kit", 200, "2x", "Fine-Point Blade"),
    ("Printable Iron-On, Dark", 120, "Off", "Fine-Point Blade"),
    ("Printable Iron-On, Light", 95, "Off", "Fine-Point Blade"),
    ("Printable Iron-On, Sublimation", 100, "Off", "Fine-Point Blade"),
    ("Printable Magnetic Sheet", 300, "Off", "Fine-Point Blade"),
    ("Printable Sticker Paper, White", 100, "2x", "Fine-Point Blade"),
    ("Printable Temporary Tattoo Paper", 350, "2x", "Fine-Point Blade"),
    ("Printable Vinyl, Specialty (Gold/Silver)", 120, "Off", "Fine-Point Blade"),
    ("Printable Vinyl, Transparent", 110, "Off", "Fine-Point Blade"),
    ("Printable Vinyl, White (Green Liner Printing)", 90, "Off", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - Transparent", 250, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - Transparent Holographic", 250, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - White", 180, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - White Holographic", 180, "2x", "Fine-Point Blade"),
    ("Puff Iron-On", 120, "Off", "Fine-Point Blade"),
    ("Reflective Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Smart Iron-On Glitter Matless Heat Transfer Vinyl", 180, "Off", "Fine-Point Blade"),
    ("Smart Iron-On Holographic Matless Heat Transfer Vinyl", 130, "Off", "Fine-Point Blade"),
    ("Smart Iron-On Matless Heat Transfer Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Smart Label Dissolvable Matless Writable Paper", 100, "Off", "Fine-Point Blade"),
    ("Smart Label Matless Writable Paper", 100, "Off", "Fine-Point Blade"),
    ("Smart Label Matless Writable Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Smart Paper Matless Sticker Cardstock", 220, "2x", "Fine-Point Blade"),
    ("Smart Stencil Matless Flexible Stencil Film", 100, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matless Permanent Vinyl", 130, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matless Removable Vinyl", 130, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matte Metallic Matless Permanent Vinyl", 110, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Shimmer Matless Permanent Vinyl", 145, "Off", "Fine-Point Blade"),
    ("SportFlex Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Sublimation Paper", 110, "Off", "Fine-Point Blade"),
    ("UV Activated Color Changing Iron-On", 100, "Off", "Fine-Point Blade"),
    ("UV-Activated Color-Changing Printable Iron-On Kit", 200, "2x", "Fine-Point Blade"),
    ("Value Cardstock - 65 lb (176gsm)", 130, "2x", "Fine-Point Blade"),
    ("Value Cardstock - 65 lb (Basic Cuts, Single Pass)", 200, "Off", "Fine-Point Blade"),
    ("Value Glitter Iron-On", 130, "Off", "Fine-Point Blade"),
    ("Value Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Value Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Vellum", 150, "Off", "Fine-Point Blade"),
    ("Watercolor Cards", 180, "2x", "Fine-Point Blade"),
]

XTRA_DATA = [
    ("3D Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Acetate", 290, "2x", "Fine-Point Blade"),
    ("Adhesive Foil", 90, "Off", "Fine-Point Blade"),
    ("Adhesive Foil, Matte", 90, "Off", "Fine-Point Blade"),
    ("Aluminum Foil", 70, "2x", "Fine-Point Blade"),
    ("Butcher Paper", 100, "Off", "Fine-Point Blade"),
    ("Calibration Paper", 90, "Off", "Fine-Point Blade"),
    ("Cardstock (H)", 182, "2x", "Fine-Point Blade"),
    ("Cardstock, Adhesive-Backed", 200, "2x", "Fine-Point Blade"),
    ("Chalkboard Vinyl", 160, "Off", "Fine-Point Blade"),
    ("Chameleon Iron-On", 130, "Off", "Fine-Point Blade"),
    ("Color Pop Iron-On", 250, "Off", "Fine-Point Blade"),
    ("Construction Paper", 150, "Off", "Fine-Point Blade"),
    ("Copy Paper", 90, "Off", "Fine-Point Blade"),
    ("Copy Paper - 24 lb (90 gsm)", 90, "Off", "Fine-Point Blade"),
    ("Copy Paper - 32 lb (120 gsm)", 90, "Off", "Fine-Point Blade"),
    ("Corrugated Cardboard", 320, "Off", "Deep-Point Blade"),
    ("Craft Foam", 80, "10x", "Deep-Point Blade"),
    ("Cutaway Card + Backer", 165, "2x", "Fine-Point Blade"),
    ("Denim, Bonded", 320, "3x", "Fine-Point Blade"),
    ("Dry Erase Vinyl", 120, "Off", "Fine-Point Blade"),
    ("EVA Foam", 150, "Off", "Deep-Point Blade"),        # normalized "EVA foam" → "EVA Foam"
    ("Everyday Iron-On", 90, "Off", "Fine-Point Blade"),
    ("Faux Leather (Paper Thin)", 150, "2x", "Fine-Point Blade"),
    ("Faux Suede", 90, "3x", "Fine-Point Blade"),
    ("Felt", 160, "2x", "Fine-Point Blade"),
    ("Felt, Wool Bonded", 90, "3x", "Fine-Point Blade"),
    ("Felt, Wool Fabric", 320, "6x", "Fine-Point Blade"),
    ("Flat Cardboard", 320, "Off", "Fine-Point Blade"),
    ("Flocked Iron-On", 120, "Off", "Fine-Point Blade"),
    ("Flocked Paper", 250, "3x", "Fine-Point Blade"),
    ("Foil Iron-On", 90, "Off", "Fine-Point Blade"),
    ("Foil Poster Board", 330, "2x", "Fine-Point Blade"),
    ("Glitter Cardstock", 280, "2x", "Fine-Point Blade"),
    ("Glitter Iron-On", 180, "Off", "Fine-Point Blade"),
    ("Glow in the Dark Iron-On", 90, "Off", "Fine-Point Blade"),
    ("Glow in the Dark Vinyl", 110, "2x", "Fine-Point Blade"),
    ("Heat/Cold Activated Color Changing Vinyl", 130, "Off", "Fine-Point Blade"),
    ("Heavy Cardstock - 100 lb (270 gsm)", 270, "2x", "Fine-Point Blade"),
    ("Heavy Watercolor Paper - 140 lbs (300 gsm)", 220, "2x", "Fine-Point Blade"),
    ("Holographic Iron-On", 90, "Off", "Fine-Point Blade"),
    ("Holographic Vinyl", 90, "Off", "Fine-Point Blade"),
    ("Infusible Ink Transfer Sheet", 230, "Off", "Fine-Point Blade"),
    ("Insert Card - Cardstock", 180, "2x", "Fine-Point Blade"),
    ("Iron-On Patches", 130, "3x", "Fine-Point Blade"),
    ("Kraft Board", 330, "3x", "Fine-Point Blade"),
    ("Kraft Cardstock", 300, "Off", "Fine-Point Blade"),
    ("Laminated Printable Sticker Paper, White", 250, "2x", "Fine-Point Blade"),
    ("Laser Copy Paper", 90, "Off", "Fine-Point Blade"),
    ("Light Cardstock - 65 lb (176 gsm)", 140, "2x", "Fine-Point Blade"),
    ("Light Chipboard - 0.37 mm", 320, "2x", "Fine-Point Blade"),
    ("Magnetic Sheet - 0.5 mm", 400, "5x", "Fine-Point Blade"),
    ("Magnetic Sheet - 1.0 mm", 350, "8x", "Deep-Point Blade"),
    ("Matboard (1.5 mm)", 350, "6x", "Deep-Point Blade"),
    ("Medium Cardstock - 80 lb (216 gsm)", 180, "2x", "Fine-Point Blade"),
    ("Medium Cardstock - 80 lb (Basic Cuts, Single Pass)", 270, "Off", "Fine-Point Blade"),
    ("Medium Fabrics (like Cotton), Bonded", 90, "3x", "Fine-Point Blade"),
    ("Metallic Puff Iron-On", 120, "Off", "Fine-Point Blade"),
    ("Natural Wood Veneer", 310, "4x", "Deep-Point Blade"),
    ("Photo Paper - 48 lb", 200, "Off", "Fine-Point Blade"),
    ("Premium Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Frosted Opaque", 130, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - Textured Metallic", 150, "Off", "Fine-Point Blade"),
    ("Premium Vinyl - True Brushed", 240, "Off", "Fine-Point Blade"),
    ("Printable Iron-On Glitter Kit", 240, "2x", "Fine-Point Blade"),
    ("Printable Iron-On Glow in the Dark Kit", 160, "2x", "Fine-Point Blade"),
    ("Printable Iron-On Holographic Kit", 160, "2x", "Fine-Point Blade"),
    ("Printable Iron-On, Dark", 100, "Off", "Fine-Point Blade"),
    ("Printable Iron-On, Light", 70, "Off", "Fine-Point Blade"),
    ("Printable Iron-On, Sublimation", 100, "Off", "Fine-Point Blade"),
    ("Printable Magnetic Sheet", 300, "Off", "Fine-Point Blade"),
    ("Printable Sticker Paper, White", 80, "2x", "Fine-Point Blade"),
    ("Printable Temporary Tattoo Paper", 280, "2x", "Fine-Point Blade"),
    ("Printable Vinyl, Specialty (Gold/Silver)", 110, "Off", "Fine-Point Blade"),
    ("Printable Vinyl, Transparent", 100, "Off", "Fine-Point Blade"),
    ("Printable Vinyl, White (Green Liner Printing)", 60, "Off", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - Transparent", 230, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - Transparent Holographic", 230, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - White", 200, "2x", "Fine-Point Blade"),
    ("Printable Waterproof Sticker Set - White Holographic", 200, "2x", "Fine-Point Blade"),
    ("Puff Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Reflective Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Rubber (1.0 mm)", 180, "2x", "Deep-Point Blade"),
    ("Smart Iron-On Glitter Matless Heat Transfer Vinyl", 170, "Off", "Fine-Point Blade"),
    ("Smart Iron-On Holographic Matless Heat Transfer Vinyl", 180, "Off", "Fine-Point Blade"),
    ("Smart Iron-On Matless Heat Transfer Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Smart Label Dissolvable Matless Writable Paper", 100, "Off", "Fine-Point Blade"),
    ("Smart Label Matless Writable Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Smart Paper Matless Sticker Cardstock", 180, "Off", "Fine-Point Blade"),
    ("Smart Stencil Matless Flexible Stencil Film", 100, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matless Permanent Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matless Removable Vinyl", 120, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Matte Metallic Matless Permanent Vinyl", 110, "Off", "Fine-Point Blade"),
    ("Smart Vinyl Shimmer Matless Permanent Vinyl", 150, "Off", "Fine-Point Blade"),
    ("SportFlex Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Sublimation Paper", 100, "Off", "Fine-Point Blade"),
    ("Transparency", 230, "Off", "Fine-Point Blade"),
    ("UV Activated Color Changing Iron-On", 120, "Off", "Fine-Point Blade"),
    ("UV-Activated Color-Changing Printable Iron-On Kit", 180, "2x", "Fine-Point Blade"),
    ("Value Cardstock - 65 lb (176gsm)", 160, "2x", "Fine-Point Blade"),
    ("Value Cardstock - 65 lb (Basic Cuts, Single Pass)", 240, "Off", "Fine-Point Blade"),
    ("Value Glitter Iron-On", 150, "Off", "Fine-Point Blade"),
    ("Value Iron-On", 100, "Off", "Fine-Point Blade"),
    ("Value Vinyl", 100, "Off", "Fine-Point Blade"),
    ("Vellum", 130, "Off", "Fine-Point Blade"),
    ("Washi Tape - 0.06 mm", 50, "Off", "Fine-Point Blade"),
    ("Watercolor Cards", 200, "2x", "Fine-Point Blade"),
    ("Wax Paper", 160, "Off", "Fine-Point Blade"),
    ("Wrapping Paper", 100, "Off", "Fine-Point Blade"),
]

# ── Category assignment ───────────────────────────────────────────────────────

# Explicit overrides take priority over pattern matching
CATEGORY_OVERRIDES = {
    "acetate":                            "Others",
    "aluminum foil":                      "Others",
    "butcher paper":                      "Paper",
    "calibration paper":                  "Paper",
    "craft foam":                         "Others",
    "denim, bonded":                      "Fabric",
    "eva foam":                           "Others",
    "faux leather (paper thin)":          "Leather",
    "faux suede":                         "Leather",
    "felt":                               "Others",
    "felt, wool bonded":                  "Fabric",
    "felt, wool fabric":                  "Others",
    "flat cardboard":                     "Board/Cardboard",
    "foil poster board":                  "Board/Cardboard",
    "heavy watercolor paper - 140 lbs (300 gsm)": "Paper",
    "kraft board":                        "Board/Cardboard",
    "kraft cardstock":                    "Cardstock",
    "light chipboard - 0.37 mm":          "Board/Cardboard",
    "magnetic sheet - 0.5 mm":           "Others",
    "magnetic sheet - 1.0 mm":           "Others",
    "matboard (1.5 mm)":                  "Board/Cardboard",
    "medium fabrics (like cotton), bonded": "Fabric",
    "natural wood veneer":                "Others",
    "photo paper - 48 lb":               "Paper",
    "printable magnetic sheet":           "Printable Materials",
    "rubber (1.0 mm)":                    "Others",
    "sticker new":                        "Others",   # custom user entry
    "sticker sub one":                    "Others",   # custom user entry
    "stnew":                              "Others",   # custom user entry
    "sublimation paper":                  "Paper",
    "transparency":                       "Others",
    "vellum":                             "Paper",
    "washi tape - 0.06 mm":              "Others",
    "watercolor cards":                   "Cardstock",
    "wax paper":                          "Paper",
    "wrapping paper":                     "Paper",
    "corrugated cardboard":               "Board/Cardboard",
    "flocked paper":                      "Paper",
    "cardstock (h)":                      "Cardstock",
    "cardstock, adhesive-backed":         "Cardstock",
    "adhesive foil":                      "Vinyl",
    "adhesive foil, matte":               "Vinyl",
}


def assign_category(name: str) -> str:
    key = name.strip().lower()
    if key in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[key]

    # Smart Materials (must come before Iron-On / Vinyl checks)
    if key.startswith("smart "):
        return "Smart Materials"

    # Iron-On (including "Printable Iron-On" variants)
    if "iron-on" in key or "iron on" in key:
        return "Iron-On"

    # Infusible Ink
    if "infusible ink" in key:
        return "Infusible Ink"

    # Printable Materials (before Vinyl / Paper checks; excludes Iron-On handled above)
    if "printable" in key:
        return "Printable Materials"

    # Vinyl — check before Board/Cardboard so "Chalkboard Vinyl" → Vinyl, not Board
    if "vinyl" in key:
        return "Vinyl"

    # Board / Cardboard
    if any(w in key for w in ("chipboard", "cardboard", "matboard", "board")):
        return "Board/Cardboard"

    # Cardstock
    if any(w in key for w in ("cardstock", "insert card", "cutaway card")):
        return "Cardstock"

    # Leather
    if any(w in key for w in ("leather", "suede")):
        return "Leather"

    # Fabric (bonded)
    if "bonded" in key:
        return "Fabric"

    # Paper
    if any(w in key for w in ("paper", "vellum")):
        return "Paper"

    return "Others"


# ── Format converters ─────────────────────────────────────────────────────────

MULTICUT_MAP = {
    "Off":  "-",
    "2x":   "2倍",
    "3x":   "3倍",
    "4x":   "4倍",
    "5x":   "5倍",
    "6x":   "6倍",
    "7x":   "7倍",
    "8x":   "8倍",
    "10x":  "10倍",
    "12x":  "12倍",
    "24x":  "24倍",
}

BLADE_MAP = {
    "Fine-Point Blade":  "ファインポイントブレード",
    "Deep-Point Blade":  "ディープポイントブレード",
    "Knife Blade":       "ナイフの刃",
    "Rotary Blade":      "ロータリーブレード",
    "Bonded Fabric Blade": "ボンデッドファブリックブレード",
}


def to_csv_row(machine, name, pressure, multicut_raw, blade_en, jp_lookup):
    category = assign_category(name)
    multicut = MULTICUT_MAP.get(multicut_raw, multicut_raw)
    blade_jp = BLADE_MAP.get(blade_en, blade_en)
    name_jp = jp_lookup.get(name.lower(), "")
    return {
        "Machine": machine,
        "Category": category,
        "Material Name (JP)": name_jp,
        "Material Name (EN)": name,
        "Cutting Pressure": pressure,
        "Multi-Cut": multicut,
        "Blade Type": blade_jp,
    }


# ── Build JP name lookup from existing combined CSV ───────────────────────────

combined = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")
jp_lookup = dict(
    zip(
        combined["Material Name (EN)"].str.strip().str.lower(),
        combined["Material Name (JP)"].fillna(""),
    )
)

# ── Build DataFrames ──────────────────────────────────────────────────────────

joy2_rows  = [to_csv_row("Cricut Joy 2",   n, p, m, b, jp_lookup) for n, p, m, b in JOY2_DATA]
xtra_rows  = [to_csv_row("Cricut Joy Xtra", n, p, m, b, jp_lookup) for n, p, m, b in XTRA_DATA]

joy2_df  = pd.DataFrame(joy2_rows)
xtra_df  = pd.DataFrame(xtra_rows)

# ── Save individual CSVs ──────────────────────────────────────────────────────

joy2_path = os.path.join(DATA_DIR, "Material List (Settings) - Cricut Joy 2.csv")
xtra_path = os.path.join(DATA_DIR, "Material List (Settings) - Cricut Joy Xtra.csv")

joy2_df.drop(columns=["Machine"]).to_csv(joy2_path, index=False, encoding="utf-8-sig")
xtra_df.drop(columns=["Machine"]).to_csv(xtra_path, index=False, encoding="utf-8-sig")

print(f"Saved {len(joy2_df)} rows → {os.path.basename(joy2_path)}")
print(f"Saved {len(xtra_df)} rows → {os.path.basename(xtra_path)}")

# ── Append to combined CSV (remove existing Joy 2 / Joy Xtra rows first) ─────

NEW_MACHINES = {"Cricut Joy 2", "Cricut Joy Xtra"}
base = combined[~combined["Machine"].isin(NEW_MACHINES)]
new_rows = pd.concat([joy2_df, xtra_df], ignore_index=True)
updated  = pd.concat([base, new_rows], ignore_index=True)

updated.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
print(f"\nCombined CSV updated: {len(combined)} → {len(updated)} rows")
print(f"  Added: {len(new_rows)} rows ({len(joy2_df)} Joy 2 + {len(xtra_df)} Joy Xtra)")

# ── Category distribution report ─────────────────────────────────────────────

print("\nCategory distribution — new machines:")
for machine, df in [("Cricut Joy 2", joy2_df), ("Cricut Joy Xtra", xtra_df)]:
    print(f"\n  {machine}:")
    for cat, count in df["Category"].value_counts().items():
        print(f"    {cat}: {count}")

