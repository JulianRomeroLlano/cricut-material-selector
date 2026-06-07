# Research: Material Physical Properties for Cutting Prediction
**Date**: 2026-06-06
**Researcher**: Claude (WebSearch — academic + industry sources)
**Purpose**: Identify physical material properties to add as ML features in the Cricut cutting settings predictor, addressing the unacceptable val metrics (MAPE 38.5%, blade acc 81.2%, multi-cut acc 66%).

---

## Executive Summary

The current ML model uses only 4 derived features: `category` (one-hot), `thickness_mm` (log-normalized), `hardness_score` (1–10 heuristic), and `is_bonded_fabric` (binary). These features lose critical physical information because they are either coarse (category), inferred by regex from material names (thickness, hardness), or only relevant to one blade type (is_bonded_fabric).

Academic literature on blade cutting mechanics, knife die-cutting, and kiss-cutting identifies **6 additional physical properties** that are directly correlated with blade force requirement, cut depth, and multi-pass strategy:

| Property | Relevance to Cutting |
|----------|---------------------|
| Density (kg/m³) | Heavier = more mass per cut stroke = higher resistance |
| Grammage / GSM (g/m²) | Directly maps to paper/fabric pressure in CSV data |
| Shore hardness (A or D) | Quantifies deformation resistance — strongly correlated with blade force |
| Elastic modulus E (MPa) | Stiffness dictates spring-back and necessary cut pressure |
| Surface texture / finish | Glitter, foil, flock add 10–25% pressure above smooth baseline |
| Adhesive backing | Affects material stability under blade; slightly reduces multi-cut need |

The two highest-impact additions for the current 726-row dataset are **GSM** (grammage) and **surface texture**, as both are inferable from existing material names without external lookup, and both have clear signal in the pressure CSV data.

---

## 1. Blade Cutting Mechanics — Academic Basis

### Kiss Cutting and Die Cutting Models

From industrial kiss-cutting literature (IQS Directory, Strouse, ResearchGate — "AE Analysis on Blade Cutting Pressure Adjustment in Dynamic Cutting of Paperboard"):

- A blade must penetrate **75–90% of material depth**, with the remaining thin web bursting under shearing + compressive forces. This is the "kiss-cut" principle, directly analogous to Cricut cutting: the machine calibrates force to cut through the top layer without scoring the mat.
- Initial blade force estimate = f(substrate thickness + top layer thickness). Cricut's "cutting pressure" directly maps to this force calibration.
- Acoustic emission (AE) studies confirm that each material has a characteristic crack-initiation signature — equivalent to the "breakthrough" event the blade triggers.

**Implication for ML**: Thickness alone is insufficient for kiss-cut force prediction. Layer count, material modulus, and fracture toughness are all independent contributors to the force needed.

### Orthotropic Behavior (Paper and Wood)

Paper and wood are orthotropic (different properties in machine direction vs. cross direction). From ScienceDirect ("Directional dependence of mechanical properties of aged paper"):
- Paper Young's modulus in thickness direction is **1–2 orders of magnitude smaller** than in-plane modulus
- Cutting is primarily an in-plane operation, so in-plane stiffness drives blade force
- Machine-direction (MD) vs. cross-direction (CD) stiffness ratio: typically 1.5–2.0× for copy paper, up to 3× for cardstock

---

## 2. Physical Properties by Material Category

### 2.1 Paper and Cardstock

**Reference values (from TAPPI standards, materials science literature):**

| Property | Copy Paper (75 gsm) | Cardstock (80 lb / 216 gsm) | Thick Cardstock (100 lb / 271 gsm) |
|----------|--------------------|-----------------------------|-------------------------------------|
| GSM (g/m²) | 60–100 | 150–250 | 250–350 |
| Caliper (mm) | 0.08–0.12 | 0.20–0.30 | 0.27–0.40 |
| Density (kg/m³) | 700–900 | 700–850 | 700–850 |
| Elastic modulus E (MPa) | 2,000–4,000 (in-plane) | 3,000–6,000 | 5,000–9,000 |
| Poisson's ratio ν | 0.25–0.35 | 0.25–0.35 | 0.25–0.35 |
| Tensile strength (MPa) | 20–60 | 40–100 | 60–130 |
| Gurley stiffness correlation | Low | Medium | High |

Key finding: **GSM is the single best continuous predictor of paper/cardstock cutting pressure** because:
- CSV data shows: thin paper (70–110 gsm) → pressure 70–130; mid cardstock (216 gsm) → 180–220; heavy cardstock (271 gsm) → 260–320
- Many material names already contain weight (e.g., "Cardstock (216gsm)", "Paper 90gsm"), enabling direct extraction
- Fracture toughness increases >50% when caliper doubles; GSM and caliper are linearly correlated for same material grade

### 2.2 Vinyl (Adhesive and Non-Adhesive)

**Reference values (PVC/PU vinyl materials, ASTM D412, D2240):**

| Property | Standard Adhesive Vinyl | Shimmer/Metallic Vinyl | Glitter Vinyl | Iron-On HTV (PU) |
|----------|------------------------|------------------------|---------------|------------------|
| Shore hardness | A 50–75 (flexible PVC) | A 60–80 | A 40–65 (surface grit) | A 30–55 |
| Density (kg/m³) | 1,200–1,400 (PVC) | 1,200–1,400 | 1,300–1,600 | 900–1,200 (PU) |
| Thickness total (mm) | 0.07–0.12 | 0.07–0.15 | 0.10–0.20 | 0.08–0.15 |
| Elastic modulus E (MPa) | 10–50 | 15–60 | 20–80 (with grit) | 5–30 |
| Elongation at break (%) | 150–400 | 100–250 | 50–200 | 200–500 |

From ResearchGate and Stahls' HTV Reference Guide:
- Rigid PVC: Shore D ~80, tensile strength 40–60 MPa — this is "thick vinyl" / window cling territory
- Flexible PVC: Shore A 50–100, modulus 5–50 MPa — standard craft vinyl
- PU (polyurethane) HTV: softer than PVC, stretches with fabric, Shore A 30–55
- **Glitter vinyl requires 10–25% higher pressure** because the glass/polyester glitter particles resist blade deformation — accounted for by surface texture feature
- Plasticizer content controls Shore A in PVC — matte vinyl has slightly more plasticizer (softer) than glossy, but pressure difference is minimal (<5%)

### 2.3 Fabric (Unbonded — Rotary Blade)

**Reference values (textile standards, fabric science):**

| Property | Lightweight Fabric (e.g., Muslin) | Medium Fabric (Canvas) | Heavy Fabric (Denim) |
|----------|----------------------------------|------------------------|----------------------|
| GSM (g/m²) | 60–120 | 200–350 | 350–600 |
| Density (kg/m³) | 200–400 | 300–500 | 400–600 |
| Caliper/thickness (mm) | 0.1–0.5 | 0.4–1.5 | 0.8–3.0 |
| Weave tightness (threads/cm) | 10–20 | 15–25 | 20–40 |
| Tensile strength (MPa) | 5–50 | 30–150 | 80–300 |

Key finding: For rotary blade (pressure range 600–3,550 gf), **GSM is the dominant predictor**. The CSV shows:
- Lightweight fabrics (muslin, cotton lawn ~100–130 gsm): pressure ~600–900 gf
- Medium fabrics (quilting cotton ~130–170 gsm): pressure ~1,100–1,800 gf
- Heavy fabrics (canvas, denim ~350+ gsm): pressure ~2,000–3,550 gf

Plain weave vs. twill weave at the same GSM: plain weave typically requires ~5–10% higher pressure (more interlocking threads per cut stroke). However, this effect is small compared to GSM variation.

### 2.4 Leather

**Reference values (from MDPI "Mechanical Parameters of Leather..." 2022, JALCA "Prediction of Leather Mechanical Properties" 2017):**

| Property | Thin Leather (2 oz / 0.8 mm) | Medium Leather (4 oz / 1.6 mm) | Thick Leather (6 oz / 2.4 mm) |
|----------|------------------------------|--------------------------------|-------------------------------|
| Thickness (mm) | 0.6–1.0 | 1.4–1.8 | 2.0–2.8 |
| Density (kg/m³) | 860–950 | 860–950 | 860–950 |
| Young's modulus E (MPa) | 10–40 (full-grain) | 15–60 | 25–80 |
| Poisson's ratio ν | 0.25–0.45 | 0.25–0.45 | 0.25–0.45 |
| Tensile strength (MPa) | 10–30 | 15–40 | 20–55 |
| Elongation at break (%) | 20–60 | 15–50 | 10–40 |

From MDPI research:
- Leather exhibits **non-elastic, elastic, and plastic behavior** typical of a multilayer anisotropic structure (grain layer vs. corium/dermal layer)
- Young's modulus from tensile testing: higher moisture content → softer (moisture acts as plasticizer)
- Full-grain leather (grain layer intact) cuts harder than suede/split leather at the same thickness
- **Processing reduces modulus by ~80%** for stitched/tanned leather vs. raw hide
- Key insight: **Thickness (oz/mm) is the single best predictor within the Leather category** — CSV confirms this (0.8mm → 180, 1.6mm → 200–287, 2.4mm → 450 on Maker 3)

### 2.5 EVA Foam and Craft Foam

**Reference values (Damao Tech EVA Foam Guide, ASTM D3575, JIS K6767):**

| Property | Soft EVA Foam | Medium EVA Foam | Hard EVA Foam | Craft Foam (thin) |
|----------|--------------|-----------------|---------------|-------------------|
| Shore A hardness | 25–38 | 38–55 | 55–75 | 38–45 |
| Density (kg/m³) | 30–80 | 80–200 | 200–400 | 80–120 |
| Elastic modulus E (MPa) | 0.5–3 | 3–15 | 15–50 | 2–8 |
| Tensile strength (MPa) | 0.5–2 | 1–5 | 3–10 | 0.5–2 |
| Elongation at break (%) | 400–600 | 200–400 | 100–200 | 200–600 |
| Typical thickness (mm) | 2–10 | 1–6 | 0.5–3 | 0.5–2 |

Key finding: **Shore hardness and density are independent in EVA foam** (high-density foam can be soft, low-density can be hard depending on cell structure). Both are needed to characterize foam cutting resistance.

Craft foam (typically used in Cricut: 0.5–2mm thick): Shore A 38–45, density ~85–120 kg/m³. Cutting pressure in CSV: ~145–200 gf (fine-point, Joy-family). This is consistent with thin soft foam requiring only moderate pressure.

### 2.6 Board/Cardboard and Chipboard

**Reference values (corrugated board mechanics literature, chipboard manufacturing specs):**

| Property | Chipboard (0.37mm) | Chipboard (1.5mm) | Chipboard (2.0mm) | Corrugated |
|----------|-------------------|-------------------|-------------------|------------|
| GSM (g/m²) | 350–400 | 1,100–1,400 | 1,500–1,900 | 300–600 |
| Caliper (mm) | 0.35–0.40 | 1.3–1.6 | 1.8–2.2 | 2.5–8.0 |
| Density (kg/m³) | 700–1,000 | 800–1,000 | 800–950 | 100–400 |
| Elastic modulus E (MPa) | 2,000–5,000 | 3,000–7,000 | 4,000–8,000 | 500–3,000 |
| Bending stiffness (N·m) | 0.001–0.005 | 0.05–0.2 | 0.1–0.5 | 0.5–5 |

Key finding: Chipboard is orthotropic; bending stiffness increases roughly as caliper³ × material modulus. For cutting prediction: **thickness (mm) is directly extractable from material names** and is the dominant predictor. GSM adds secondary information.

### 2.7 Wood (Balsa, Basswood Veneer)

**Reference values (Wood Database, True Geometry, ScienceDirect):**

| Property | Balsa Wood | Basswood |
|----------|-----------|----------|
| Janka hardness (lbf) | 67 | 410 |
| Elastic modulus E (GPa) | 2–3.7 | 10.0 |
| Modulus of rupture (MPa) | 19.6 | 60.0 |
| Density (kg/m³) | 120–200 | 300–500 |
| Typical Cricut thickness (mm) | 0.5–2.0 | 0.5–1.6 |

Key finding: Balsa vs. basswood have 6× difference in Janka hardness and 5× difference in elastic modulus. Both are in the "Others" or special categories in CSV. Cricut Maker 3 can cut both with Knife Blade. **Hardness score + thickness together predict wood cutting pressure well.**

---

## 3. Poisson's Ratio Summary Table

Poisson's ratio (ν) measures how much a material contracts laterally when stretched longitudinally. For cutting operations, lower Poisson's ratio = material holds its shape under blade side-load = clean cuts.

| Material | Poisson's Ratio ν | Notes |
|---------|------------------|-------|
| Paper (in-plane) | 0.25–0.35 | Orthotropic; MD and CD differ |
| Cardstock | 0.25–0.35 | Similar to paper |
| PVC vinyl (flexible) | 0.38–0.45 | Near rubber-like (incompressible) |
| PU (HTV/iron-on) | 0.35–0.45 | Highly elastic |
| Leather | 0.25–0.45 | Wide range due to anisotropy |
| EVA foam | 0.10–0.25 | Compressible foam; lower ν |
| Chipboard | 0.20–0.35 | Paper-based composite |
| Balsa wood | 0.22–0.30 | Orthotropic wood |
| Basswood | 0.25–0.35 | Orthotropic wood |
| Fabric (knit) | 0.30–0.60 | Highly anisotropic |
| Fabric (woven) | 0.20–0.50 | Direction-dependent |

**Assessment**: Poisson's ratio has limited practical utility as an ML feature because:
1. It is very difficult for users to estimate without lab equipment
2. It does not vary enough within categories to explain pressure differences
3. Its effect on blade cutting force is secondary to stiffness (E) and thickness

**Recommendation**: Do NOT add Poisson's ratio as a user-facing input feature. Include it in this research document for completeness.

---

## 4. Proposed New Features for ML Model

Based on the research above and analysis of existing CSV data, here are the proposed new columns to add to the material CSVs and the ML feature set:

### Priority 1 — High Signal, Inferable from Names (add to CSV immediately)

#### 4.1 GSM — Grams per Square Meter (g/m²)
- **ML relevance**: Strongest continuous predictor for paper, cardstock, and fabric pressure
- **Source**: Many material names contain gsm directly ("Paper (90gsm)", "Cardstock (216gsm)")
- **Inference method**: Regex extraction from name; fallback to category+thickness default
- **Reference defaults by category**:
  | Category | Default GSM |
  |----------|------------|
  | Paper | 75 gsm |
  | Cardstock | 216 gsm |
  | Infusible Ink | 75 gsm |
  | Iron-On / HTV | 100 gsm |
  | Vinyl | 120 gsm (includes adhesive+film) |
  | Fabric (lightweight) | 110 gsm |
  | Fabric (medium) | 200 gsm |
  | Fabric (heavy) | 400 gsm |
  | Board/Cardboard | computed from thickness × 950 kg/m³ |
  | Leather | computed from thickness × 900 kg/m³ |
  | Others (foam) | 200 gsm (EVA 2mm default) |

#### 4.2 Surface Texture (categorical)
- **ML relevance**: Glitter/foil/flock consistently add 10–25% cutting pressure above smooth baseline
- **Source**: Inferable from material name keywords with high accuracy
- **Categories**: `plain`, `glitter`, `foil`, `flock`, `shimmer`, `matte`, `glossy`, `holographic`, `textured`
- **Simplified encoding for ML**: `surface_hardness_modifier` (0.0 = plain smooth, 0.1 = matte, 0.2 = glossy/shimmer, 0.3 = foil/metallic, 0.5 = glitter/flock/textured)

#### 4.3 Has Adhesive Backing (boolean)
- **ML relevance**: Adhesive-backed materials are slightly more stable under blade; minor effect on multi-cut
- **Source**: Inferable from name keywords ("adhesive", "sticker", "self-adhesive", "peel")
- **Encoding**: 0 or 1

### Priority 2 — Medium Signal, Requires Category Defaults (add with defaults)

#### 4.4 Density (kg/m³)
- **ML relevance**: Denser materials require more blade force at equal thickness
- **Source**: Not in material names; use category+material type defaults
- **Reference defaults**:
  | Material Type | Density (kg/m³) |
  |--------------|----------------|
  | Paper | 750 |
  | Cardstock | 800 |
  | PVC vinyl | 1,300 |
  | PU iron-on (HTV) | 1,050 |
  | Glitter vinyl | 1,500 (with grit filler) |
  | Leather | 900 |
  | EVA foam (craft) | 100 |
  | EVA foam (thick) | 200 |
  | Chipboard | 850 |
  | Balsa | 150 |
  | Basswood | 400 |
  | Woven fabric | 300 |
  | Knit fabric | 200 |
  | Infusible ink | 800 |

#### 4.5 Shore Hardness (normalized 0–1 scale)
- **ML relevance**: Direct measure of deformation resistance; correlated with blade force requirement
- **Current model uses**: 1–10 heuristic hardness score (keyword-based inference)
- **Improvement**: Replace heuristic with calibrated Shore A equivalent value
- **Reference defaults by category**:
  | Material | Shore A equivalent |
  |---------|------------------|
  | Paper (copy) | 10–20 (very soft when cut in-plane) |
  | Cardstock (heavy) | 25–35 |
  | Chipboard | 50–80 |
  | Standard vinyl (PVC) | 55–70 |
  | Glitter vinyl | 60–75 |
  | Iron-on HTV (PU) | 35–55 |
  | Thin leather | 40–55 |
  | Thick leather | 60–80 |
  | Craft foam (EVA thin) | 38–45 |
  | Thick EVA foam | 50–70 |
  | Balsa wood | 85–95 |
  | Basswood | 95–100+ |
  | Fabric | 5–15 (very low — fibers compress) |

### Priority 3 — Lower Signal, Research-Only (do NOT add as features yet)

#### 4.6 Elastic Modulus E (MPa)
- Wide range (0.5 MPa for foam → 10,000 MPa for chipboard) requires careful normalization
- Log-scale normalization would be needed (similar to pressure)
- Not easily estimated by end users
- Correlated with Shore hardness (which we already capture) — adding both risks feature collinearity
- **Recommendation**: Monitor whether GSM + Shore hardness improve accuracy sufficiently before adding E

#### 4.7 Poisson's Ratio
- See Section 3 — low practical utility as user input; limited within-category variation
- **Recommendation**: Do NOT add

#### 4.8 Fracture Toughness (KIc)
- Academic literature shows strong correlation with cutting force at crack initiation
- Not estimable by crafters; requires lab testing
- Partially captured by material stiffness + thickness features
- **Recommendation**: Do NOT add (use as validation concept, not input feature)

---

## 5. Impact Assessment on ML Model

### Why Current Model Underperforms

The best val epoch was epoch 15 out of 800 — the model stopped improving very early because the 19 current features do not capture enough independent variance. Adding features that are:
- **Independent** from current features (surface texture ≠ category, GSM ≠ thickness)
- **Directly correlated** with the targets (GSM → pressure for paper/fabric; surface texture → pressure for vinyl/iron-on)
- **Inferable from existing data** (no external lookup needed)

...should increase the variance explained by features and allow the model to train longer before overfitting.

### Projected Feature Improvement

| Feature Addition | Expected Pressure MAE Reduction | Expected Multi-cut Acc Gain |
|-----------------|--------------------------------|----------------------------|
| GSM (paper/cardstock) | ~15–20% | ~5% |
| Surface texture modifier | ~8–12% | ~2% |
| Has adhesive backing | ~3–5% | ~8% (multi-cut stability) |
| Density (category defaults) | ~5–8% | ~2% |
| Shore hardness (calibrated) | ~5–10% | ~3% |
| **All combined** | **~30–45% total** | **~15–20%** |

These are estimates. Actual improvement depends on whether the new features have correlated signal in the 726-row training set.

### New Proposed Feature Vector (30 total)

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1–5 | Machine one-hot (5) | Binary | User input |
| 6–16 | Category one-hot (11) | Binary | User input |
| 17 | Thickness log-normalized | Float | Regex from name / user input |
| 18 | Hardness score normalized | Float | Keyword inference / user input |
| 19 | is_bonded_fabric | Binary | Name / blade type inference |
| **20** | **GSM log-normalized** | **Float** | **Regex from name / category default** |
| **21** | **Surface hardness modifier** | **Float** | **Keyword inference from name** |
| **22** | **has_adhesive_backing** | **Binary** | **Keyword inference from name** |
| **23** | **density_norm** | **Float** | **Category default lookup** |
| **24** | **shore_hardness_norm** | **Float** | **Category default + surface override** |
| **25–30** | *(reserve for future: layer_count, foil_type, weave_type)* | — | — |

Active new features: 6 additional → 25 total feature dimensions.

---

## 6. Inference Logic for New Features (Python Pseudocode)

```python
GSM_KEYWORDS = {
    # Pattern: regex → gsm value
    r'(\d+)\s*gsm': lambda m: float(m.group(1)),
    r'(\d+)\s*g/m': lambda m: float(m.group(1)),
    r'(\d+)\s*lb\b': lambda m: float(m.group(1)) * 3.76,  # lb to gsm (cardstock approx)
}

GSM_DEFAULTS = {
    'Paper': 75, 'Cardstock': 216, 'Iron-On': 100, 'Vinyl': 120,
    'Smart Materials': 120, 'Infusible Ink': 75, 'Printable Materials': 100,
    'Board/Cardboard': None,  # compute from thickness
    'Leather': None,  # compute from thickness
    'Fabric': 150, 'Others': 200,
}

SURFACE_TEXTURE_KEYWORDS = {
    'glitter': 0.5, 'flock': 0.5, 'textured': 0.4,
    'foil': 0.3, 'metallic': 0.3, 'chrome': 0.3, 'mirror': 0.3,
    'holographic': 0.3, 'iridescent': 0.25, 'shimmer': 0.2,
    'glossy': 0.15, 'glossy': 0.15,
    'matte': 0.1, 'satin': 0.1,
    # default: 0.0 (plain/smooth)
}

ADHESIVE_KEYWORDS = ['adhesive', 'sticker', 'self-adhesive', 'peel', 'peel & stick',
                     'self adhesive', '(adhesive)', 'backed']

DENSITY_DEFAULTS = {
    'Paper': 750, 'Cardstock': 800, 'Iron-On': 1050,
    'Vinyl': 1300, 'Smart Materials': 1300, 'Infusible Ink': 800,
    'Printable Materials': 900, 'Board/Cardboard': 850,
    'Leather': 900, 'Fabric': 280, 'Others': 150,
}

SHORE_DEFAULTS = {  # Shore A equivalent, normalized to 0-1 range (max=100)
    'Paper': 0.15, 'Cardstock': 0.30, 'Iron-On': 0.45, 'Vinyl': 0.65,
    'Smart Materials': 0.65, 'Infusible Ink': 0.20, 'Printable Materials': 0.25,
    'Board/Cardboard': 0.65, 'Leather': 0.55, 'Fabric': 0.10, 'Others': 0.40,
}
```

---

## 7. New CSV Columns to Add

The following columns should be added to `assets/data/Material List (Combined).csv` and the per-machine CSVs:

| Column Name | Type | Values | Population Method |
|-------------|------|--------|------------------|
| `GSM` | Float | 40–600 g/m² | Regex extraction from EN name; fallback to category default |
| `Surface Texture` | String | plain/matte/glossy/shimmer/foil/metallic/holographic/glitter/flock/textured | Keyword inference from EN name |
| `Has Adhesive` | Integer | 0 / 1 | Keyword inference from EN name |
| `Density (kg/m3)` | Float | 100–1,500 | Category default lookup with keyword overrides |
| `Shore Hardness (A)` | Float | 5–100 | Category default with surface texture/thickness override |

**Total new columns**: 5 (bringing CSV width from 7 to 12 columns)

---

## 8. Limitations and Caveats

1. **Category-level defaults are approximations**: Assigning "Vinyl → Shore A 65" treats all 80+ vinyl variants identically. The inter-material variance within categories is exactly what we want the model to learn — but within-category variance of the *new features* (if populated only from defaults) may not add signal beyond the category one-hot.

2. **GSM extraction coverage**: Scanning 726 material names, approximately:
   - 30–40 materials explicitly state gsm in name → extracted accurately
   - 80–100 materials state lb/oz → convertible
   - Remaining 580+ use category default → adds noise, not signal

3. **Surface texture coverage**: Approximately 60–80 materials in the combined CSV have explicit surface texture keywords (glitter, foil, flock, shimmer) → these are the materials where this feature adds clear signal. The rest default to 0.0 (plain).

4. **Model size**: Adding 6 features (25 total) allows a slightly larger model without overfitting. With 726 rows, a model up to ~20,000 parameters should be safe.

5. **The fundamental data size limit**: 726 rows for 5 machines × 11 categories × 5 blade types is a sparse problem. The most impactful single improvement would be adding more training rows from real Cricut Design Space data for missing material/machine combinations.

---

## Sources

- IQS Directory, Strouse: die cutting and kiss cutting mechanics
- ResearchGate — "AE Analysis on Blade Cutting Pressure Adjustment in Dynamic Cutting of Paperboard"
- ScienceDirect — "Directional dependence of mechanical properties of aged paper"
- ScienceDirect — "Mechanical properties of a balsa wood veneer structural sandwich core material"
- MDPI — "Mechanical Parameters of Leather in Relation to Technological Processing of the Footwear Uppers" (2022)
- JALCA — "Prediction of Leather Mechanical Properties" (2017)
- Damao Tech — EVA Foam Density & Hardness Guide
- HTVRONT — Heat Transfer Vinyl composition and structure
- Stahls' HTV Reference Guide (PDF)
- Wood Database (wood-database.com) — Balsa, Basswood Janka hardness and elastic modulus
- True Geometry — Balsa Wood Material Properties
- TAPPI standards (paper grammage and caliper)
- ASTM D2240 (Shore durometer), ASTM D3575 (EVA foam), JIS K6767 (foam)