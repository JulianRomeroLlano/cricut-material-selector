# Research: Cricut Machine Specifications
**Date**: 2026-06-06
**Researcher**: Claude (Researcher Agent + CSV Analysis)

## Summary
Three machines are in scope: Maker 3 (professional/full-featured), Explore 3 (mid-range), and Cricut Joy (compact/portable). Their key differences are cutting force range, supported blade types, and mat size — which directly affect which materials each machine can handle. The same material requires different pressure settings on different machines, so **machine must be included as a feature in the ML model**.

---

## Cricut Maker 3

### Cutting Force / Pressure
- **CSV range**: 70 – 3,550 (grams-force equivalent)
- Maximum advertised cutting force: ~4,000 g (~4 kg)
- The pressure scale in the CSV is approximately 1:1 with grams-force
- Fabric materials (Rotary blade) reach up to 3,550 — a fundamentally different operating range than all other blade types (max ~750)

### Supported Blades (from CSV data)
| Japanese | English | Use Case |
|----------|---------|----------|
| ファインポイントブレード | Fine Point Blade | Paper, vinyl, iron-on, cardstock, thin materials |
| ディープポイントブレード | Deep Point Blade | Dense/thick materials: chipboard, leather, foam |
| ナイフの刃 | Knife Blade | Very thick materials: basswood, heavy chipboard, balsa |
| ロータリーブレード | Rotary Blade | Unbonded fabric (cuts without backing mat) |
| ボンデッドファブリックブレード | Bonded Fabric Blade | Fabric bonded to iron-on backing |

### Mat Sizes
- 12×12 inches (30.5×30.5 cm) — standard
- 12×24 inches (30.5×61 cm) — extended
- Smart Materials: matless cutting up to 12 ft (366 cm) long

### Material Categories (11 total)
Board/Cardboard, Cardstock, Fabric, Infusible Ink, Iron-On, Leather, Others, Paper, Printable Materials, Smart Materials, Vinyl

### Pressure by Category
| Category | Min | Max | Count |
|----------|-----|-----|-------|
| Board/Cardboard | 200 | 750 | 18 |
| Cardstock | 200 | 320 | 10 |
| Fabric | 175 | 3,550 | 132 |
| Infusible Ink | 226 | 226 | 1 |
| Iron-On | 80 | 268 | 16 |
| Leather | 180 | 450 | 10 |
| Others | 70 | 350 | 39 |
| Paper | 70 | 350 | 32 |
| Printable Materials | 70 | 293 | 17 |
| Smart Materials | 106 | 310 | 8 |
| Vinyl | 85 | 300 | 30 |

### Multi-Cut
- Maximum: 24 passes
- Distribution: 82% single-pass, 18% require multi-cut

### Total Materials: 313

---

## Cricut Explore 3

### Cutting Force / Pressure
- **CSV range**: 70 – 350 (grams-force equivalent)
- Maximum advertised cutting force: ~2,000 g (~2 kg)
- Does NOT support Knife Blade or Rotary Blade → no thick boards or unbonded fabric

### Supported Blades (from CSV data)
| Japanese | English | Use Case |
|----------|---------|----------|
| ファインポイントブレード | Fine Point Blade | Paper, vinyl, iron-on, cardstock |
| ディープポイントブレード | Deep Point Blade | Denser materials: light leather, thick paper |
| ボンデッドファブリックブレード | Bonded Fabric Blade | Fabric with iron-on backing |

### Mat Sizes
- 12×12 inches (30.5×30.5 cm)
- 12×24 inches (30.5×61 cm)
- Smart Materials: matless cutting supported

### Material Categories (11 total, same as Maker 3)
Board/Cardboard, Cardstock, Fabric, Infusible Ink, Iron-On, Leather, Others, Paper, Printable Materials, Smart Materials, Vinyl

### Pressure by Category
| Category | Min | Max | Count |
|----------|-----|-----|-------|
| Board/Cardboard | 300 | 350 | 8 |
| Cardstock | 180 | 320 | 9 |
| Fabric | 285 | 321 | 4 |
| Infusible Ink | 226 | 226 | 1 |
| Iron-On | 80 | 268 | 15 |
| Leather | 215 | 325 | 5 |
| Others | 70 | 350 | 22 |
| Paper | 70 | 345 | 28 |
| Printable Materials | 70 | 277 | 15 |
| Smart Materials | 121 | 300 | 8 |
| Vinyl | 76 | 190 | 25 |

### Multi-Cut
- Maximum: 8 passes
- Distribution: 79% single-pass, 21% require multi-cut

### Total Materials: 140

---

## Cricut Joy

### Cutting Force / Pressure
- **CSV range**: 80 – 330 (grams-force equivalent)
- Compact machine, lower maximum force than Maker 3 / Explore 3
- Fine Point blade only — most limited blade support

### Supported Blades (from CSV data)
| Japanese | English | Use Case |
|----------|---------|----------|
| ファインポイント | Fine Point | Paper, vinyl, iron-on, cardstock, thin materials |

### Mat Sizes
- 4.5×6.5 inches (11.4×16.5 cm) — standard Joy mat
- 4.5×12 inches (11.4×30.5 cm) — extended Joy mat
- Smart Materials: matless cutting is the primary use case for Joy

### Material Categories (10 total — no Fabric category)
Board/Cardboard, Cardstock, Infusible Ink, Iron-On, Leather (limited), Others, Paper, Pens & Markers, Smart Materials, Vinyl

### Pressure by Category
| Category | Min | Max | Count |
|----------|-----|-----|-------|
| Board/Cardboard | 226 | 286 | 3 |
| Cardstock | 223 | 315 | 6 |
| Infusible Ink | 241 | 241 | 1 |
| Iron-On | 95 | 246 | 13 |
| Leather | 251 | 251 | 1 |
| Others | 120 | 269 | 3 |
| Paper | 100 | 330 | 11 |
| Pens & Markers | N/A | N/A | 6 |
| Smart Materials | 80 | 285 | 14 |
| Vinyl | 121 | 206 | 22 |

### Multi-Cut
- Maximum: 3 passes
- Distribution: 78% single-pass, 22% require multi-cut

### Total Materials: 80

---

## Machine Comparison Table

| Feature | Maker 3 | Explore 3 | Cricut Joy |
|---------|---------|-----------|------------|
| Cutting Force (max) | ~4,000 g | ~2,000 g | ~1,000 g |
| Pressure CSV range | 70–3,550 | 70–350 | 80–330 |
| Fine Point Blade | ✓ | ✓ | ✓ |
| Deep Point Blade | ✓ | ✓ | ✗ |
| Knife Blade | ✓ | ✗ | ✗ |
| Rotary Blade | ✓ | ✗ | ✗ |
| Bonded Fabric Blade | ✓ | ✓ | ✗ |
| Standard mat | 12×12 in (30.5×30.5 cm) | 12×12 in (30.5×30.5 cm) | 4.5×6.5 in (11.4×16.5 cm) |
| Extended mat | 12×24 in (30.5×61 cm) | 12×24 in (30.5×61 cm) | 4.5×12 in (11.4×30.5 cm) |
| Smart Materials (matless) | ✓ | ✓ | ✓ (primary mode) |
| Max multi-cut passes | 24× | 8× | 3× |
| Max material thickness | ~2.4 mm (basswood) | ~1.5 mm (light board) | ~0.5 mm (thin card) |
| Fabric support | Full (bonded + unbonded) | Bonded only | None |
| Material count in data | 313 | 140 | 80 |
| Total unique categories | 11 | 11 | 10 (no Fabric) |

---

## Cutting Pressure Scale Notes

The CSV "Cutting Pressure" column values map approximately 1:1 to grams-force:
- Maker 3 CSV max = 3,550 g; advertised max ≈ 4,000 g → scale factor ≈ 1.0
- Values are **machine-specific**: the same material (e.g. Everyday Iron-On) shows pressure 113 on Maker 3, 113 on Explore 3, and 95 on Cricut Joy
- **Critical ML implication**: Machine must be an input feature OR separate models per machine must be trained
- Rotary blade fabric pressures (600–3,550) operate on a functionally different scale — consider treating as a separate output target or normalizing per blade type
- Pens & Markers (Cricut Joy only) have no cutting pressure — these rows should be excluded from ML training

---

## Sources
- `assets/data/Material List (Combined).csv` — 533 rows, analyzed 2026-06-06
- `assets/data/Material List (Settings) - Maker 3.csv`
- `assets/data/Material List (Settings) - explore 3.csv`
- `assets/data/Material List (Settings) - cricut joy.csv`
- Agent findings (subagent run 2026-06-06)

## Open Questions
1. Do Explore 3 fabric pressures (285–321) use a special blade adapter, or does it use the Bonded Fabric blade in a non-rotary fashion?
2. The Cricut Joy shows a "Pens & Markers" category with no pressure values — these are draw-only, not cut. Should this become a separate "tool type" feature?
3. Are Joy's lower pressure values (vs. Maker 3 / Explore 3 for same material) due to machine hardware or calibration differences?