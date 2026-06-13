"""
Live-site test: loads the GitHub Pages deployment in a headless browser,
calls window.CricutPredict.predict() for every material in the CSV,
and reports errors vs expected values.
"""
import csv, re, math, sys, json, asyncio, pathlib
from playwright.async_api import async_playwright

LIVE_URL   = "https://julianromerollano.github.io/cricut-material-selector/"
CSV_PATH   = pathlib.Path(__file__).parent.parent / "assets/data/Material List (Combined).csv"
PRESSURE_ERR_PCT = 15.0

# Map CSV "Machine" → JS machine key (must match pp.machines keys in predict.js)
MACHINE_MAP = {
    "Cricut Joy":       "Cricut Joy",
    "Cricut Joy 2":     "Cricut Joy 2",
    "Cricut Joy Xtra":  "Cricut Joy Xtra",
    "Explore 3":        "Explore 3",
    "Maker 3":          "Maker 3",
}

# JP multi-cut regex
JP_MC_RE = re.compile(r'^(\d+)倍$')

BLADE_JP_TO_EN = {
    "ファインポイントブレード":      "Fine-Point Blade",
    "ディープポイントブレード":      "Deep-Point Blade",
    "ナイフの刃":                    "Knife Blade",
    "ロータリーブレード":            "Rotary Blade",
    "ボンデッドファブリックブレード": "Bonded Fabric Blade",
    # pass-through for already-English values
    "Fine-Point Blade":      "Fine-Point Blade",
    "Deep-Point Blade":      "Deep-Point Blade",
    "Knife Blade":           "Knife Blade",
    "Rotary Blade":          "Rotary Blade",
    "Bonded Fabric Blade":   "Bonded Fabric Blade",
}

# MC labels must exactly match predict.js: ["1×","2×","3×","4–5×","6–8×","10+×"]
# Bucket 0 = single pass (raw "−" or "1") → site label "1×"
MC_SITE_LABELS = ["1×", "2×", "3×", "4–5×", "6–8×", "10+×"]

def parse_mc(raw: str) -> int:
    s = str(raw).strip()
    if s in ("-", "", "nan", "1"):
        return 0   # single pass = bucket 0
    m = JP_MC_RE.match(s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except ValueError:
        return 0

def bucket_mc(n: int) -> int:
    """Map MC integer value to bucket index — must match train_model_v2.py."""
    if n <= 1:         return 0
    if n == 2:         return 1
    if n == 3:         return 2
    if 4 <= n <= 5:    return 3
    if 6 <= n <= 8:    return 4
    return 5

def mc_label(n: int) -> str:
    return MC_SITE_LABELS[bucket_mc(n)]

def load_csv():
    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            machine = r["Machine"].strip()
            if machine not in MACHINE_MAP:
                continue
            # Smart materials are excluded from the model (Cricut-only presets)
            if r["Material Name (EN)"].strip().startswith("Smart"):
                continue
            mc_raw = r.get("Multi-Cut", "-").strip()
            mc_n   = parse_mc(mc_raw)
            if mc_n is None:
                continue
            pressure = float(r["Cutting Pressure"] or 0)
            if pressure <= 0:
                continue
            # Use explicit thickness_mm column when present (avoids infer_thickness
            # giving the wrong default for same-name thickness variants).
            th_explicit = r.get("thickness_mm", "").strip()
            thickness = float(th_explicit) if th_explicit else 0  # 0 = let site resolve via material_lookup
            rows.append({
                "machine":   machine,
                "name":      r["Material Name (EN)"].strip(),
                "category":  r["Category"].strip(),
                "thickness": thickness,
                "pressure":  pressure,
                "blade":     BLADE_JP_TO_EN.get(r["Blade Type"].strip(), r["Blade Type"].strip()),
                "mc_label":  mc_label(mc_n),
                "mc_raw":    mc_raw,
            })
    return rows

async def run_test():
    rows = load_csv()
    print(f"Loaded {len(rows)} materials from CSV")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page    = await context.new_page()

        # Force fresh fetch for model assets (bypass Chromium disk cache)
        async def no_cache(route):
            await route.continue_(headers={**route.request.headers,
                                           "Cache-Control": "no-cache", "Pragma": "no-cache"})
        await context.route("**/*.onnx", no_cache)
        await context.route("**/*.json", no_cache)

        # Suppress console noise
        page.on("console", lambda msg: None)

        print(f"Loading {LIVE_URL} …", flush=True)
        await page.goto(LIVE_URL, wait_until="networkidle", timeout=60_000)

        # Wait until CricutPredict and ONNX are ready
        await page.wait_for_function(
            "window.CricutPredict && typeof window.CricutPredict.predict === 'function'",
            timeout=30_000,
        )

        # Warm up: force preprocessor + all 5 models to load
        print("Warming up ONNX sessions …", flush=True)
        machines = list(MACHINE_MAP.keys())
        for mach in machines:
            await page.evaluate(f"""async () => {{
                await window.CricutPredict.predict({{
                    machine: {json.dumps(mach)},
                    materialName: "Cardstock",
                    category: "Cardstock",
                    thicknessMm: 0.2
                }});
            }}""")
        print("Warm-up done.\n", flush=True)

        # --- run all predictions ---
        blade_errors   = []
        mc_errors      = []
        press_errors   = []
        js_errors      = []
        ok = 0

        total = len(rows)
        for i, row in enumerate(rows):
            if i % 100 == 0:
                print(f"  {i}/{total} …", flush=True)

            try:
                result = await page.evaluate("""async ({ machine, name, category, thickness }) => {
                    const r = await window.CricutPredict.predict({
                        machine:      machine,
                        materialName: name,
                        category:     category,
                        thicknessMm:  thickness,
                    });
                    return { pressure: r.pressure, blade: r.bladeEn, mc: r.multicut };
                }""", {"machine": row["machine"], "name": row["name"],
                       "category": row["category"], "thickness": row["thickness"]})
            except Exception as e:
                js_errors.append({"row": row, "error": str(e)})
                continue

            pred_p  = result["pressure"]
            pred_b  = result["blade"]
            pred_mc = result["mc"]

            # Blade check
            if pred_b != row["blade"]:
                blade_errors.append({**row, "pred_blade": pred_b})

            # MC check
            if pred_mc != row["mc_label"]:
                mc_errors.append({**row, "pred_mc": pred_mc})

            # Pressure check
            gt = row["pressure"]
            pct = abs(pred_p - gt) / gt * 100 if gt else 0
            if pct > PRESSURE_ERR_PCT:
                press_errors.append({**row, "pred_p": pred_p, "err_pct": pct})
            else:
                ok += 1

        await context.close()
        await browser.close()

    # ── report ──────────────────────────────────────────────────────────────
    n = len(rows)
    print("\n" + "═"*82)
    print(f"  LIVE SITE TEST  — {n} materials across {len(machines)} machines")
    print("═"*82)
    print(f"  Wrong blade:          {len(blade_errors):3}/{n}  ({len(blade_errors)/n*100:.1f}%)")
    print(f"  Wrong multi-cut:      {len(mc_errors):3}/{n}  ({len(mc_errors)/n*100:.1f}%)")
    print(f"  Pressure >15%%:      {len(press_errors):3}/{n}  ({len(press_errors)/n*100:.1f}%)")
    if js_errors:
        print(f"  JS errors:           {len(js_errors):3}/{n}")

    if blade_errors:
        print("\n── BLADE ERRORS ──────────────────────────────────────────────────────────────")
        print(f"  {'Machine':<16} {'Material':<44} {'GT':<22} {'Pred'}")
        for e in blade_errors:
            print(f"  {e['machine']:<16} {e['name'][:43]:<44} {e['blade']:<22} {e['pred_blade']}")

    if mc_errors:
        print("\n── MULTI-CUT ERRORS ──────────────────────────────────────────────────────────")
        print(f"  {'Machine':<16} {'Material':<44} {'GT':<10} {'Pred'}")
        for e in mc_errors:
            print(f"  {e['machine']:<16} {e['name'][:43]:<44} {e['mc_label']:<10} {e['pred_mc']}")

    if press_errors:
        # Flag names that were normalized (measurement parenthetical stripped)
        _MEAS_PAREN = re.compile(
            r'''\s*\(\s*\d[\d\s./\-]*(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?
                (?:\s*/\s*[\d\s./\-]+(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?)?
                \s*\)\s*''',
            re.IGNORECASE | re.VERBOSE,
        )
        def is_stripped(name):
            cleaned = _MEAS_PAREN.sub('', name.strip()).strip()
            return bool(cleaned) and cleaned != name.strip()

        stripped = [e for e in press_errors if is_stripped(e["name"])]
        genuine  = [e for e in press_errors if not is_stripped(e["name"])]
        print(f"\n── PRESSURE ERRORS > 15%%  ({len(press_errors)})")
        print(f"  S = name was stripped of measurement parenthetical (known limitation)")
        print(f"  Stripped variant: {len(stripped)}  |  Genuine: {len(genuine)}")
        print(f"\n  {'Machine':<16} {'S'} {'Material':<44} {'GT':>5} {'Pred':>5}  {'Err%':>6}")
        print(f"  {'─'*16} {'─'} {'─'*44} {'─'*5} {'─'*5}  {'─'*6}")
        for e in sorted(press_errors, key=lambda x: -x["err_pct"]):
            s = "S" if is_stripped(e["name"]) else " "
            print(f"  {e['machine']:<16} {s} {e['name'][:43]:<44} {int(e['pressure']):>5} "
                  f"{int(e['pred_p']):>5}  {e['err_pct']:>5.1f}%")

    if js_errors:
        print("\n── JS ERRORS ─────────────────────────────────────────────────────────────────")
        for e in js_errors:
            print(f"  {e['row']['machine']} / {e['row']['name']}: {e['error']}")

    # Per-machine summary
    from collections import defaultdict
    mach_blade   = defaultdict(int)
    mach_mc      = defaultdict(int)
    mach_press   = defaultdict(int)
    mach_total   = defaultdict(int)
    for r in rows:
        mach_total[r["machine"]] += 1
    for e in blade_errors:  mach_blade[e["machine"]]  += 1
    for e in mc_errors:     mach_mc[e["machine"]]     += 1
    for e in press_errors:  mach_press[e["machine"]]  += 1

    print(f"\n── PER-MACHINE SUMMARY ───────────────────────────────────────────────────────")
    print(f"  {'Machine':<16} {'Total':>5}  {'Blade':>5}  {'MC':>5}  {'Press':>5}")
    for m in machines:
        print(f"  {m:<16} {mach_total[m]:>5}  {mach_blade[m]:>5}  "
              f"{mach_mc[m]:>5}  {mach_press[m]:>5}")
    print("═"*82 + "\n")

if __name__ == "__main__":
    asyncio.run(run_test())
