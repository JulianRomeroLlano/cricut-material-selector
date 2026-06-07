# Manual Test Checklist — Cricut Material Selector
**App URL:** http://localhost:8765  (run: `python3 -m http.server 8765`)  
**Automated tests:** `node tests/check_data.js` · `tests/unit.html` · `tests/model_test.html`

---

## S — Browse / Search

| ID | Test | Pass |
|----|------|------|
| S-01 | Page load shows all Joy materials, no filters active | ☐ |
| S-02 | Click each machine tab → material list + count updates | ☐ |
| S-03 | All 5 machine tabs visible and clickable | ☐ |
| S-04 | Type "iron" → case-insensitive EN name match shown | ☐ |
| S-05 | In JP mode, type "アイロン" → matching rows shown | ☐ |
| S-06 | Type "xyznotexist" → empty state message, count = 0 | ☐ |
| S-07 | Category dropdown "Plastic" → only Plastic badge rows | ☐ |
| S-08 | Category=Vinyl + search="smart" → rows match both | ☐ |
| S-09 | Reset category dropdown → original list restored | ☐ |
| S-10 | Blade filter "Rotary Blade" → only rotary rows (Maker 3) | ☐ |
| S-11 | Sort ↑ Pressure → rows re-order, lowest first | ☐ |
| S-12 | Sort ↓ Pressure → rows re-order, highest first | ☐ |
| S-13 | Each row shows: name, category badge, pressure, blade, multi-cut | ☐ |
| S-14 | Pressure shown as integer (e.g. "280" not "280.0") | ☐ |
| S-15 | Multi-cut "1×" shown (not raw "-") | ☐ |
| S-16 | Result count label updates after every filter change | ☐ |
| S-17 | Maker 3 (471 rows, no filter) renders without visible jank | ☐ |

---

## P — AI Predict

| ID | Test | Pass |
|----|------|------|
| P-01 | AI Predict tab shows: machine, category, thickness, predict button | ☐ |
| P-02 | Machine dropdown has all 5 machines | ☐ |
| P-03 | Category dropdown has all 12 categories (incl. Plastic) | ☐ |
| P-04 | Select "Paper" → thickness auto-fills 0.08 | ☐ |
| P-06 | Explore 3 + Cardstock + 0.25mm → shows pressure/blade/multicut | ☐ |
| P-07 | Button shows "Predicting…" and disables while running | ☐ |
| P-08 | Pressure result looks sane (60–4000) | ☐ |
| P-09 | Blade shown as full name e.g. "Fine-Point Blade" | ☐ |
| P-10 | Multi-cut shown as "1×","2×" etc. — not raw index | ☐ |
| P-11 | Disclaimer "AI estimate… verify with a test cut" visible | ☐ |
| P-12 | Cricut Joy + any category → blade always "Fine-Point Blade" | ☐ |
| P-17 | Thickness = 0 → validation error shown, model NOT called | ☐ |
| P-18 | Thickness = -1 → validation error shown | ☐ |
| P-19 | Thickness = "abc" → validation error shown | ☐ |

---

## L — Localization

| ID | Test | Pass |
|----|------|------|
| L-01 | Default load (no stored lang) → UI in English | ☐ |
| L-02 | Click JP toggle → UI labels switch to Japanese | ☐ |
| L-03 | JP mode → material name_jp shown as primary | ☐ |
| L-04 | EN mode → material name_en shown as primary | ☐ |
| L-05 | JP mode → category dropdown shows "ビニール" etc. | ☐ |
| L-06 | JP mode → predict result blade in Japanese | ☐ |
| L-08 | JP mode + invalid thickness → error in Japanese | ☐ |
| L-10 | No raw i18n key strings visible anywhere in either mode | ☐ |
| L-11 | Toggle EN→JP→EN without page reload; lang preserved in localStorage | ☐ |

---

## R — Responsive Layout

| ID | Test | Device | Pass |
|----|------|--------|------|
| R-01 | No horizontal scroll; all controls usable | 375px (iPhone SE) | ☐ |
| R-02 | All 5 machine tabs accessible (scroll if needed) | 375px | ☐ |
| R-05 | Content centered with max-width; not edge-to-edge | 1280px | ☐ |

---

## X — Error & Edge Cases

| ID | Test | Pass |
|----|------|------|
| X-01 | (automated) ONNX 404 → graceful error message | ☐ |
| X-03 | (automated) materials.json 404 → error state, no white screen | ☐ |
| X-05 | Very long material name → text wraps or truncates; layout intact | ☐ |
| X-08 | Materials with pressure > 1000 (e.g. Canvas 2100) → displayed correctly | ☐ |

---

## Automated tests (run before shipping)

```bash
node tests/check_data.js           # D-01..D-08, M-01..M-03a: 12/12
# open in browser:
# tests/unit.html                  # E, D, M, P validation: 24/24
# tests/model_test.html            # M, P, Regression: 21/21
```

*Last automated run: 2026-06-07 — 57 tests, 0 failures*
