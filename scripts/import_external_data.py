"""
import_external_data.py — Import material settings from external community database.

Source: https://tamayuzucraft-source.github.io/cricut/jcutatu.html
Cached HTML: provided as argument or auto-detected from transcript cache.

Usage:
  source venv/bin/activate
  python scripts/import_external_data.py <path_to_cached_html>
"""
import re, json, sys, os
import pandas as pd
from collections import defaultdict

ROOT     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "assets", "data")
COMBINED = os.path.join(DATA_DIR, "Material List (Combined).csv")

CACHED_HTML = (
    sys.argv[1] if len(sys.argv) > 1
    else "/home/julian/.claude/projects/-home-julian-Projects-cricut-material-selector/"
         "a9f4938f-9e12-4435-89f7-cb358a0ba43f/tool-results/bwcvji0m2.txt"
)

# ─── Mapping tables ────────────────────────────────────────────────────────────

MACHINE_MAP = {
    "Joy": "Cricut Joy",
    "Explorer3": "Explore 3",
    "Maker3": "Maker 3",
}

CATEGORY_MAP = {
    "アートボード":       "Board/Cardboard",
    "アイロン接着タイプ": "Iron-On",
    "カードストック":     "Cardstock",
    "その他":            "Others",
    "ビニール":          "Vinyl",
    "プラスチック":       "Others",   # re-checked below for vinyl sub-types
    "ホイル・金属":       "Others",
    "紙":               "Paper",
    "布":               "Fabric",
    "フェルト":          "Others",   # re-checked below for rotary blade
    "フォーム":          "Others",
    "革":               "Leather",
    "木材":             "Others",
}

BLADE_MAP = {
    "-":                    "ファインポイントブレード",  # Joy: Fine-Point only
    "ファインポイントブレード": "ファインポイントブレード",
    "ディープポイントブレード": "ディープポイントブレード",
    "ナイフブレード":          "ナイフの刃",
    "ロータリーブレード":       "ロータリーブレード",
    "該当なし":               None,  # skip — no valid cut setting
}

MULTICUT_MAP = {
    "Off": "-", "1x": "-",
    "2x": "2倍", "3x": "3倍", "3ｘ": "3倍",
    "4x": "4倍", "5x": "5倍", "6x": "6倍", "7x": "7倍", "8x": "8倍",
    "12x": "12倍", "14x": "14倍", "16x": "16倍",
    "17x": "17倍", "18x": "18倍", "24x": "24倍",
    "該当なし": None,  # skip
}

# ─── Name extraction helpers ───────────────────────────────────────────────────

def _is_jp_char(c: str) -> bool:
    cp = ord(c)
    return (0x3000 <= cp <= 0x9FFF) or (0xFF00 <= cp <= 0xFFEF)


def extract_en(name: str) -> str:
    """Extract the English portion from a combined JP/EN name field."""
    if " / " in name:
        return name.split(" / ")[-1].strip()
    # Find the last CJK/full-width character; everything after is English
    last_jp = -1
    for i, c in enumerate(name):
        if _is_jp_char(c):
            last_jp = i
    return name[last_jp + 1:].strip() if last_jp >= 0 else name.strip()


def extract_jp(name: str) -> str:
    """Extract the Japanese portion from a combined JP/EN name field."""
    if " / " in name:
        return name.split(" / ")[0].strip()
    last_jp = -1
    for i, c in enumerate(name):
        if _is_jp_char(c):
            last_jp = i
    return name[:last_jp + 1].strip() if last_jp >= 0 else ""


# ─── Parse external DATA array ────────────────────────────────────────────────

def load_external_data(html_path: str) -> list[dict]:
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const DATA = (\[.*?\]);", html, re.DOTALL)
    if not m:
        raise ValueError("Could not find 'const DATA = [...]' in HTML.")
    return json.loads(m.group(1))


# ─── Main import ──────────────────────────────────────────────────────────────

def main():
    print("Loading external HTML data...")
    raw = load_external_data(CACHED_HTML)
    print(f"  {len(raw)} raw entries found.")

    print("Loading existing combined CSV...")
    df_exist = pd.read_csv(COMBINED, encoding="utf-8-sig")
    n_before = len(df_exist)
    print(f"  {n_before} existing rows.")

    # Build dedup key set: (machine, lower-EN-name)
    existing_keys = set()
    for _, row in df_exist.iterrows():
        key = (str(row["Machine"]).strip(), str(row["Material Name (EN)"]).strip().lower())
        existing_keys.add(key)

    new_rows = []
    skipped = defaultdict(int)

    for d in raw:
        machine = MACHINE_MAP.get(d["machine"])
        if not machine:
            skipped["unknown_machine"] += 1
            continue

        blade = BLADE_MAP.get(d["blade"])
        if blade is None:
            skipped["no_blade"] += 1
            continue

        mcut = MULTICUT_MAP.get(d["multicut"])
        if mcut is None:
            skipped["no_multicut"] += 1
            continue

        try:
            pressure = float(d["pressure"])
        except (ValueError, TypeError):
            skipped["no_pressure"] += 1
            continue

        en_name = extract_en(d["name"])
        jp_name = extract_jp(d["name"])
        if not en_name:
            skipped["no_en_name"] += 1
            continue

        # Category: initial mapping
        cat = CATEGORY_MAP.get(d["category"], "Others")

        # プラスチック sub-type: vinyl keywords → Vinyl
        if d["category"] == "プラスチック" and "vinyl" in en_name.lower():
            cat = "Vinyl"

        # フェルト with ロータリーブレード → Fabric (bonded/wool felt cut with rotary)
        if d["category"] == "フェルト" and blade == "ロータリーブレード":
            cat = "Fabric"

        key = (machine, en_name.lower())
        if key in existing_keys:
            skipped["duplicate"] += 1
            continue

        new_rows.append({
            "Machine":           machine,
            "Category":          cat,
            "Material Name (JP)": jp_name,
            "Material Name (EN)": en_name,
            "Cutting Pressure":  pressure,
            "Multi-Cut":         mcut,
            "Blade Type":        blade,
        })
        existing_keys.add(key)  # prevent within-batch duplicates

    print(f"\nImport summary:")
    print(f"  New rows to add:  {len(new_rows)}")
    print(f"  Skipped (dup):    {skipped['duplicate']}")
    print(f"  Skipped (blade):  {skipped['no_blade']}")
    for k, v in skipped.items():
        if k not in ("duplicate", "no_blade"):
            print(f"  Skipped ({k}): {v}")

    if not new_rows:
        print("\nNothing to add.")
        return

    # Machine breakdown
    from collections import Counter
    mc = Counter(r["Machine"] for r in new_rows)
    print(f"\n  By machine:   {dict(mc)}")
    cc = Counter(r["Category"] for r in new_rows)
    print(f"  By category:  {dict(cc)}")

    # Build new DataFrame — use same column order as existing (no enriched cols yet)
    base_cols = ["Machine", "Category", "Material Name (JP)", "Material Name (EN)",
                 "Cutting Pressure", "Multi-Cut", "Blade Type"]
    df_new = pd.DataFrame(new_rows, columns=base_cols)

    # Append to existing CSV (keep enriched columns from existing rows; new rows get
    # NaN for those — enrich_features.py will fill them in next step)
    df_combined = pd.concat([df_exist, df_new], ignore_index=True)
    df_combined.to_csv(COMBINED, index=False, encoding="utf-8-sig")

    n_after = len(df_combined)
    print(f"\nCombined CSV: {n_before} → {n_after} rows (+{n_after - n_before})")
    print(f"Saved to: {COMBINED}")
    print("\nNext: run  python scripts/enrich_features.py  to populate new rows' columns.")


if __name__ == "__main__":
    main()