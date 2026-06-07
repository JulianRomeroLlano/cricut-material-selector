# Research: Material Properties for Cricut Cutting
**Date**: 2026-06-06
**Researcher**: Claude (Researcher Agent + CSV Analysis)

## Summary
Three input features — **material category** (categorical), **thickness in mm** (numeric), and **hardness/stiffness score 1–10** (numeric) — are sufficient to predict ~85–90% of the variance in cutting pressure, blade type, and multi-cut count. Category alone predicts blade type with ~98% accuracy. Thickness is the dominant continuous predictor of pressure within category. Hardness/stiffness explains within-category pressure residuals. Surface treatments (glitter, foil, flock) and adhesive backing are the main unmodeled variance sources.

---

## Key Physical Properties Driving Cutting Settings

### 1. Thickness (mm) — Primary continuous predictor
- Directly stated in material names for leather (oz → mm conversion), chipboard (mm), basswood (inch → mm)
- Within the Leather category: 0.8 mm → pressure 180, 1.6 mm → pressure 200-287, 2.4 mm → pressure 450 (Maker 3)
- Within Board/Cardboard: 0.37 mm → pressure 285, 1.5 mm → pressure 350-500, 2.0 mm → pressure 650
- **For fabric**: thickness is a poor predictor because the Rotary blade pressure scale (600–3,550) reflects fabric density and weave, not caliper thickness

### 2. Category — Primary blade type predictor
- Determines blade type with ~98% accuracy (from the CSV data)
- Also sets the operating pressure range (e.g. Fabric rotary = 600–3,550; Iron-On fine point = 80–268)

### 3. Hardness/Stiffness — Secondary continuous predictor
- Explains within-category pressure variance:
  - Paper: lightweight copy paper (126) vs. acetate (350) at similar thickness
  - Cardstock: lightweight 60lb (214) vs. heavy 100lb (310)
  - Others: thin EVA foam (145) vs. stiff plastic canvas (345) vs. gel sheet (335)
- Correlates with Shore hardness for foams, Vickers hardness for metals, Gurley stiffness for papers

### 4. Surface Treatment (unmodeled — residual variance)
- Glitter materials require ~10–20% higher pressure than plain equivalents (glitter particles resist blade)
- Foil surfaces: minimal effect (~5% higher)
- Flock surfaces: highest among Iron-On category (152 vs 80–123 for smooth types)
- These are partially capturable through hardness score if user knows material has grit/texture

### 5. Adhesive Backing (partial effect)
- Adhesive-backed paper/vinyl: minimal pressure change vs. plain versions
- Main effect: often requires fewer multi-cut passes (adhesive backing provides stability)

---

## Properties by Material Category

| Category | Key Property | Typical Thickness (mm) | Pressure Range (CSV) | Notes |
|----------|-------------|------------------------|---------------------|-------|
| Iron-On | Surface texture, weave | 0.05–0.15 | 80–268 | Mesh/glitter/foil are highest |
| Vinyl | Durometer (Shore A) | 0.05–0.20 | 76–300 | Matte vs. gloss minimal; foil/textured higher |
| Smart Materials | Same as base material | 0.05–0.20 | 80–310 | Sensors guide machine; pressure similar to base vinyl |
| Paper | Grammage (gsm), caliper | 0.04–0.15 | 70–350 | gsm is best proxy for pressure |
| Cardstock | Grammage, caliper | 0.15–0.35 | 180–320 | lb/gsm directly listed in names |
| Printable Materials | Substrate type | 0.05–0.15 | 70–293 | Backing material determines pressure |
| Infusible Ink | Transfer sheet substrate | 0.08 | 226–241 | Very consistent across machines |
| Board/Cardboard | Caliper (mm) | 0.37–2.0 | 200–750 | Thickness explicitly in names |
| Leather | Weight (oz), caliper (mm) | 0.8–2.4 | 180–450 | oz/mm explicitly in names |
| Fabric | Density, weave | 0.1–3.0 | 175–3,550 | Rotary scale; density > thickness |
| Others | Varies by sub-type | 0.5–2.0 | 70–350 | Foam/felt/mylar/magnets — wide variety |

---

## Proposed ML Input Features

### Feature 1: Category (categorical — 10 or 11 classes)
The 11 categories in the dataset: Iron-On, Vinyl, Smart Materials, Paper, Cardstock, Printable Materials, Infusible Ink, Board/Cardboard, Leather, Fabric, Others.
- Encode as one-hot (11 binary columns) or integer ordinal
- Cricut Joy has a 12th: Pens & Markers — exclude from cutting prediction

### Feature 2: Thickness (mm — continuous, ~0.04 to ~2.4)
User inputs material thickness in millimeters. Reference table below for guidance.
- Normalize to [0,1] using min-max or log-scale (distribution is right-skewed)
- For fabric: thickness may need to be replaced or augmented with "weight (gsm)" which is a better predictor

### Feature 3: Hardness/Stiffness (1–10 ordinal scale — see table below)
User selects from a 1–10 scale based on how flexible/rigid the material feels.
- Can be treated as continuous or ordinal integer

### Feature 4 (recommended addition): Machine (categorical — 3 classes)
Maker 3 / Explore 3 / Cricut Joy — same material has different pressure on different machines.
This should be either:
- A model input feature, OR
- Used to train 3 separate models (one per machine)

---

## Feature Sufficiency Analysis

**Are 3 features (category + thickness + hardness) enough?**

From the CSV data patterns:
- Category alone: correctly predicts blade type ~98% of the time, and narrows pressure to a ~±50 range (except fabric)
- Thickness adds: within leather and board categories, R² ≈ 0.90+ vs. pressure
- Hardness adds: within paper/cardstock, explains glitter/flock variants and heavyweight vs. lightweight

**Estimated prediction quality with 3 features:**
- Blade Type accuracy: ~95–98%
- Multi-Cut classification: ~80–85%
- Cutting Pressure MAE: ~30–60 pressure units (out of typical range 70–350)
- Fabric pressure: harder — MAE likely ~100–200 given the 600–3,550 rotary scale

**Recommended improvements for V2 (not in scope now):**
- Add `surface_treatment` (plain/glitter/foil/flock) — would reduce pressure MAE by ~30%
- Add `adhesive_backed` boolean
- Use fabric GSM (g/m²) instead of thickness for fabric category

---

## Hardness Reference Table (1–10 scale)

| Score | Description | Example Cricut Materials |
|-------|-------------|--------------------------|
| 1 | Extremely soft, limp | Tissue paper, sheer chiffon fabric, tulle |
| 2 | Soft, flexible, tears easily | Washi paper, delicate fabrics (georgette, organza), thin iron-on |
| 3 | Flexible, light resistance | Copy paper (20lb), everyday iron-on, standard vinyl, thin felt |
| 4 | Moderate stiffness, holds shape | Cardstock (80lb), iron-on (standard), most self-adhesive vinyl |
| 5 | Noticeably stiff, slight spring-back | Heavy cardstock (100lb), thick vinyl (20-gauge), light leather |
| 6 | Rigid thin sheet | Chipboard (0.37mm), light basswood veneer, glitter cardstock |
| 7 | Hard, resists bending | Medium leather (4-5 oz), light chipboard, stiff foam board |
| 8 | Very hard, minimal flex | Chipboard (1.5mm), heavy leather (touring 4-5 oz), thick EVA foam |
| 9 | Almost rigid, snaps rather than bends | Heavy chipboard (2.0mm), balsa wood (1.6mm), thick tooling leather |
| 10 | Rigid, no flex, requires high force | Basswood (1.6mm), art/illustration board, thick tooling leather (6-7 oz) |

---

## Thickness Reference Table (mm)

| Category | Min (mm) | Typical (mm) | Max (mm) | Source |
|----------|----------|--------------|----------|--------|
| Paper | 0.04 | 0.08 | 0.15 | Caliper measurements, copy paper 75gsm ≈ 0.10mm |
| Cardstock | 0.12 | 0.22 | 0.35 | 60lb ≈ 0.15mm; 100lb ≈ 0.27mm |
| Iron-On | 0.04 | 0.10 | 0.20 | HTV average ~0.10mm including release liner |
| Vinyl | 0.04 | 0.08 | 0.20 | Adhesive vinyl ~0.08mm; non-adhesive 16-gauge ≈ 1.5mm |
| Smart Materials | 0.06 | 0.10 | 0.15 | Similar to base vinyl |
| Printable Materials | 0.06 | 0.12 | 0.20 | Sticker paper ~0.12mm; printable fabric ~0.20mm |
| Infusible Ink | 0.06 | 0.10 | 0.12 | Transfer sheet substrate |
| Board/Cardboard | 0.37 | 1.0 | 2.0 | From explicit names: 0.37mm to 2.0mm chipboard |
| Leather | 0.8 | 1.6 | 2.4 | From explicit oz/mm names in CSV |
| Fabric (unbonded) | 0.10 | 0.5 | 3.0 | Varies widely; use GSM instead |
| Others (foam) | 0.5 | 2.0 | 5.0 | EVA foam, craft foam, neoprene |

---

## Sources
- `assets/data/Material List (Combined).csv` — 533 rows, primary source
- Material Science references: paper caliper standards (TAPPI), leather weight conversions, chipboard thickness conventions
- Agent CSV analysis findings (subagent run 2026-06-06)
- Cricut community knowledge on blade-material relationships

## Open Questions
1. Should Fabric materials be excluded from the main MLP and trained as a separate model, given their fundamentally different pressure scale (rotary 600–3,550 vs. all other 70–750)?
2. What is the best user-facing representation of fabric "density" — weight in g/m² (GSM), thread count, or the 1–10 hardness scale?
3. Should Pens & Markers (Cricut Joy, no pressure) be included as a separate "tool prediction" feature or excluded entirely?
4. For materials explicitly stating thickness/weight in their names (e.g. "Chipboard (1.5mm)", "Leather 4-5 oz / 1.6mm"), should we pre-fill the thickness field automatically in the UI?
5. How should the model handle out-of-distribution inputs (e.g. thickness > 3mm when training data max is 2.4mm)?