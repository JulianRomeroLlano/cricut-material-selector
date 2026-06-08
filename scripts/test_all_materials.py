"""
test_all_materials.py — comprehensive accuracy test against ground truth CSV.

Compares model predictions (blade, multi-cut, pressure) against every row in
Material List (Combined).csv.  Reports:
  A. Wrong blade
  B. Wrong multi-cut  (split by root cause)
  C. Pressure error > 15%
"""

import json, math, re
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).parent.parent
MODEL = ROOT / "assets/model"
DATA  = ROOT / "assets/data"

MACHINE_SLUG = {
    "Maker 3":         "maker3_v2",
    "Explore 3":       "explore3_v2",
    "Cricut Joy":      "cricut_joy_v2",
    "Cricut Joy 2":    "cricut_joy2_v2",
    "Cricut Joy Xtra": "cricut_joy_xtra_v2",
}

# ── load preprocessor ────────────────────────────────────────────────────────
with open(MODEL / "preprocessor.json") as f:
    PP = json.load(f)
assert PP.get("version") == "v2"

BLADE_TYPES_EN = PP["blade_types_en"]   # ['Deep-Point', 'Knife', 'Fine-Point', 'Bonded Fabric', 'Rotary']
BLADE_TYPES_JP = PP["blade_types_jp"]

JP_BLADE_TO_EN = dict(zip(BLADE_TYPES_JP, BLADE_TYPES_EN))

# Multi-cut: model outputs bucket index.  Bucket map must match train_model_v2.py.
# BUCKET_LABELS from training: [0, 2, 3, 4, 6, 10]  → indices 0-5
MC_BUCKET_LABELS = [0, 2, 3, 4, 6, 10]

def bucket_mc(n: int) -> int:
    """Same bucketing as train_model_v2.py"""
    if n == 0:       return 0
    if n == 2:       return 1
    if n == 3:       return 2
    if 4 <= n <= 5:  return 3
    if 6 <= n <= 8:  return 4
    return 5

# Parse Japanese multi-cut values ("2倍"→2, "-"→0)
JP_MC_RE = re.compile(r'^(\d+)倍$')
def parse_mc(val) -> int:
    s = str(val).strip()
    if s == "-" or s == "nan" or s == "1": return 0
    m = JP_MC_RE.match(s)
    if m: return int(m.group(1))
    try:  return int(s)
    except ValueError: return 0

def mc_bucket_was_trained_as_zero(raw_val: str) -> bool:
    """After the fix, Japanese MC values ARE in MULTICUT_MAP.
    Flag is now only True for values STILL not in the map (none expected)."""
    MULTICUT_MAP = {
        "-": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
        "2倍": 2, "3倍": 3, "4倍": 4, "5倍": 5, "6倍": 6, "7倍": 7, "8倍": 8,
        "10倍": 10, "12倍": 12, "14倍": 14, "16倍": 16, "17倍": 17, "18倍": 18, "24倍": 24,
    }
    s = str(raw_val).strip()
    return s not in MULTICUT_MAP and s not in ("nan",)

# category defaults (mirror predict.js)
GSM_DEF     = {"Paper":80,"Cardstock":176,"Iron-On":100,"Vinyl":120,"Smart Materials":120,
               "Infusible Ink":75,"Printable Materials":100,"Board/Cardboard":750,
               "Leather":900,"Fabric":150,"Plastic":150,"Others":200}
DENSITY_DEF = {"Paper":750,"Cardstock":800,"Iron-On":1050,"Vinyl":1300,"Smart Materials":1300,
               "Infusible Ink":800,"Printable Materials":900,"Board/Cardboard":850,
               "Leather":900,"Fabric":280,"Plastic":1350,"Others":500}
SHORE_DEF   = {"Paper":15,"Cardstock":30,"Iron-On":45,"Vinyl":65,"Smart Materials":65,
               "Infusible Ink":20,"Printable Materials":25,"Board/Cardboard":65,
               "Leather":55,"Fabric":10,"Plastic":70,"Others":40}
THICK_DEF   = {"Paper":0.08,"Cardstock":0.22,"Iron-On":0.10,"Vinyl":0.08,
               "Smart Materials":0.10,"Infusible Ink":0.10,"Printable Materials":0.12,
               "Board/Cardboard":1.0,"Leather":1.6,"Fabric":0.50,"Plastic":0.10,"Others":2.0}

def clamp01(x): return max(0.0, min(1.0, x))

# Only strip measurement-only parentheticals — matches train_model_v2.py
_MEAS_PAREN = re.compile(
    r'''\s*\(\s*\d[\d\s./\-]*(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?
        (?:\s*/\s*[\d\s./\-]+(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?)?
        \s*\)\s*$''',
    re.IGNORECASE | re.VERBOSE,
)
def normalize_name(name: str) -> str:
    cleaned = _MEAS_PAREN.sub('', name.strip()).strip()
    return cleaned or name.strip()

def build_features(material_name, category, thickness_mm):
    base = normalize_name(material_name)
    # embedding
    emb = PP["name_embeddings"].get(base)
    if emb is None:
        lower = base.lower()
        for k, v in PP["name_embeddings"].items():
            if k.lower().startswith(lower) or lower.startswith(k.lower()):
                emb = v; break
    if emb is None:
        emb = PP["category_avg_embeddings"].get(category, [0.0] * PP["emb_dim"])

    lk      = PP["material_lookup"].get(base)
    cat     = (lk["category"] if lk else None) or category or "Others"
    gsm     = lk["gsm"]                    if lk else GSM_DEF.get(cat, 100)
    density = lk["density"]                if lk else DENSITY_DEF.get(cat, 500)
    shore   = lk["shore"]                  if lk else SHORE_DEF.get(cat, 40)
    texture = lk.get("texture", 0)         if lk else 0
    adhesive= lk.get("has_adhesive", 0)    if lk else 0
    bonded  = lk.get("is_bonded", 0)       if lk else 0
    thick   = thickness_mm if (thickness_mm and thickness_mm > 0) \
              else (lk["thickness_mm"] if lk else THICK_DEF.get(cat, 0.5))

    feat = np.zeros(PP["feature_dim"], dtype=np.float32)
    for i, v in enumerate(emb): feat[i] = float(v)
    glog = math.log1p(gsm)
    feat[16] = clamp01((glog - PP["gsm_log_min"]) / (PP["gsm_log_max"] - PP["gsm_log_min"] + 1e-9))
    tlog = math.log1p(thick)
    feat[17] = clamp01((tlog - PP["thickness_log_min"]) / (PP["thickness_log_max"] - PP["thickness_log_min"] + 1e-9))
    feat[18] = float(bonded)
    feat[19] = float(texture)
    feat[20] = float(adhesive)
    dlog = math.log1p(density)
    feat[21] = clamp01((dlog - PP["density_log_min"]) / (PP["density_log_max"] - PP["density_log_min"] + 1e-9))
    feat[22] = clamp01(shore / 100.0)
    return feat

def decode_pressure(norm):
    return math.exp(float(norm) * PP["pressure_log_std"] + PP["pressure_log_mean"])

# ── load ONNX sessions ───────────────────────────────────────────────────────
SESS = {}
for mach, slug in MACHINE_SLUG.items():
    SESS[mach] = ort.InferenceSession(
        str(MODEL / f"material_predictor_{slug}.onnx"),
        providers=["CPUExecutionProvider"]
    )

# ── load CSV ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / "Material List (Combined).csv", encoding="utf-8-sig")
df = df[df["Cutting Pressure"].notna()].copy()     # drop pen/tool rows
df = df[df["Category"] != "Pens & Markers"].copy()
df["Cutting Pressure"] = df["Cutting Pressure"].astype(float)
df["Blade Type"]       = df["Blade Type"].fillna("").str.strip()
df["Multi-Cut"]        = df["Multi-Cut"].fillna("-").astype(str).str.strip()

# ── run all predictions ───────────────────────────────────────────────────────
errors_blade    = []   # wrong blade
errors_mc_bug   = []   # MC wrong DUE TO training bug (JP notation untrained)
errors_mc_real  = []   # MC wrong despite having proper Arabic notation in training
errors_press    = []   # pressure >15% (blade and MC correct OR pressure-only issue)

total = 0
for _, row in df.iterrows():
    machine  = str(row["Machine"]).strip()
    if machine not in SESS: continue

    name      = str(row["Material Name (EN)"]).strip()
    category  = str(row["Category"]).strip()
    gt_press  = float(row["Cutting Pressure"])
    gt_blade_jp = str(row["Blade Type"]).strip()
    gt_blade  = JP_BLADE_TO_EN.get(gt_blade_jp, gt_blade_jp)  # normalise to EN
    gt_mc_raw = str(row["Multi-Cut"]).strip()
    gt_mc_n   = parse_mc(gt_mc_raw)
    gt_mc_bucket = bucket_mc(gt_mc_n)

    lk       = PP["material_lookup"].get(normalize_name(name))
    thick_mm = lk["thickness_mm"] if lk else THICK_DEF.get(category, 0.5)

    feat  = build_features(name, category, thick_mm).reshape(1, -1)
    outs  = SESS[machine].run(None, {"features": feat})
    p_norm   = outs[0][0]
    b_logits = outs[1][0]
    mc_logits= outs[2][0]

    pred_press      = decode_pressure(p_norm)
    pred_blade_idx  = int(np.argmax(b_logits))
    pred_blade      = BLADE_TYPES_EN[pred_blade_idx]
    pred_mc_bucket  = int(np.argmax(mc_logits))

    pape      = abs(pred_press - gt_press) / max(gt_press, 1) * 100
    blade_ok  = (pred_blade == gt_blade)
    mc_ok     = (pred_mc_bucket == gt_mc_bucket)
    press_ok  = (pape <= 15.0)
    name_stripped = (normalize_name(name) != name)   # had parenthetical

    total += 1

    if not blade_ok:
        errors_blade.append({
            "machine": machine, "name": name, "category": category,
            "gt": gt_blade, "pred": pred_blade,
        })

    if not mc_ok:
        entry = {
            "machine": machine, "name": name,
            "gt_raw": gt_mc_raw, "gt_bucket": gt_mc_bucket,
            "pred_bucket": pred_mc_bucket,
            "gt_n": gt_mc_n,
        }
        if mc_bucket_was_trained_as_zero(gt_mc_raw):
            errors_mc_bug.append(entry)
        else:
            errors_mc_real.append(entry)

    if not press_ok:
        errors_press.append({
            "machine": machine, "name": name, "category": category,
            "gt": gt_press, "pred": round(pred_press, 1),
            "pape": round(pape, 1),
            "stripped": name_stripped,
            "blade_ok": blade_ok, "mc_ok": mc_ok,
        })

# ── report ────────────────────────────────────────────────────────────────────
W = 82
print(f"\n{'═'*W}")
print(f"  PREDICTION TEST  — {total} materials across 5 machines")
print(f"{'═'*W}")
blade_wrong  = len(errors_blade)
mc_bug       = len(errors_mc_bug)
mc_real      = len(errors_mc_real)
press_wrong  = len(errors_press)
p = lambda n: f"{n/total*100:.1f}%"

print(f"  Wrong blade:          {blade_wrong:4d}/{total}  ({p(blade_wrong)})")
print(f"  Wrong multi-cut:")
print(f"    Training bug*:      {mc_bug:4d}/{total}  ({p(mc_bug)})  ← JP notation silently zeroed")
print(f"    Genuine MC error:   {mc_real:4d}/{total}  ({p(mc_real)})")
print(f"  Pressure >15%%:       {press_wrong:4d}/{total}  ({p(press_wrong)})")
correct = total - blade_wrong - mc_bug - mc_real - press_wrong + \
          sum(1 for e in errors_press if not e["blade_ok"]) + \
          sum(1 for e in errors_press if not e["mc_ok"])
print()
print("  * Training bug: MULTICUT_MAP only handled Arabic numerals, not Japanese")
print("    'N倍' notation → all silently mapped to 0 (no multi-cut) during training.")
print("    Model learned to never predict multi-cut. Requires fix + retrain.")

# ── blade errors ──────────────────────────────────────────────────────────────
if errors_blade:
    print(f"\n{'─'*W}")
    print(f"  BLADE ERRORS ({blade_wrong})")
    print(f"{'─'*W}")
    print(f"  {'Machine':<16} {'Material':<44} {'GT':<22} {'Pred'}")
    for e in sorted(errors_blade, key=lambda x: x["machine"]):
        print(f"  {e['machine']:<16} {e['name'][:43]:<44} {e['gt']:<22} {e['pred']}")

# ── genuine MC errors ─────────────────────────────────────────────────────────
if errors_mc_real:
    print(f"\n{'─'*W}")
    print(f"  GENUINE MULTI-CUT ERRORS ({mc_real})")
    print(f"{'─'*W}")
    label = {0:"-", 1:"2x", 2:"3x", 3:"4-5x", 4:"6-8x", 5:"≥10x"}
    print(f"  {'Machine':<16} {'Material':<44} {'GT':<10} {'Pred'}")
    for e in sorted(errors_mc_real, key=lambda x: (x["machine"], x["name"])):
        print(f"  {e['machine']:<16} {e['name'][:43]:<44} {label[e['gt_bucket']]:<10} {label[e['pred_bucket']]}")

# ── pressure errors ───────────────────────────────────────────────────────────
if errors_press:
    print(f"\n{'─'*W}")
    print(f"  PRESSURE ERRORS > 15%  ({press_wrong})")
    print(f"  S = name was stripped of parenthetical (known limitation)")
    print(f"{'─'*W}")
    strip_count = sum(1 for e in errors_press if e["stripped"])
    other_count = press_wrong - strip_count
    print(f"  Stripped variant:    {strip_count}  (normalization artifact)")
    print(f"  Genuine press error: {other_count}")
    print()
    print(f"  {'Machine':<16} {'S'} {'Material':<42} {'GT':>5}→{'Pred':>5}  {'Err%':>6}")
    print(f"  {'':─<16} {'─'} {'':─<42} {'':─>5} {'':─>5}  {'':─>6}")
    for e in sorted(errors_press, key=lambda x: -x["pape"]):
        s = "S" if e["stripped"] else " "
        mat = e["name"][:41]
        print(f"  {e['machine']:<16} {s} {mat:<42} {e['gt']:>5.0f} {e['pred']:>5.0f}  {e['pape']:>5.1f}%")

# ── per-machine summary ───────────────────────────────────────────────────────
print(f"\n{'─'*W}")
print("  PER-MACHINE SUMMARY")
print(f"{'─'*W}")
print(f"  {'Machine':<16} {'Total':>5}  {'Blade':>5}  {'MC-bug':>6}  {'MC-real':>7}  {'Press':>5}")
for mach in MACHINE_SLUG:
    n   = sum(1 for _, r in df.iterrows() if str(r["Machine"]).strip() == mach)
    bl  = sum(1 for e in errors_blade   if e["machine"] == mach)
    mb  = sum(1 for e in errors_mc_bug  if e["machine"] == mach)
    mr  = sum(1 for e in errors_mc_real if e["machine"] == mach)
    pr  = sum(1 for e in errors_press   if e["machine"] == mach)
    print(f"  {mach:<16} {n:>5}  {bl:>5}  {mb:>6}  {mr:>7}  {pr:>5}")
print()
