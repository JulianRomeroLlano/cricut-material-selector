# Research: Cricut Joy 2 Specifications
**Date**: 2026-06-06
**Researcher**: Claude (WebSearch + WebFetch from cricut.com, help.cricut.com, thecountrychiccottage.net, heyletsmakestuff.com)

## Summary
The Cricut Joy 2 is the 2026 successor to the original Cricut Joy. It retains the same compact form factor and Fine Point-only blade system but adds **Print Then Cut** and **Scoring** capabilities absent from the original. Material compatibility is slightly expanded (~75+ vs 80 in original Joy CSV). No per-material cut setting data is publicly available — settings exist only in Design Space software. **For ML purposes, Joy 2 closely mirrors original Joy data and can be treated as a Joy-family machine pending actual CSV data.**

---

## Machine Specifications

| Feature | Value |
|---------|-------|
| Release | 2026 |
| Dimensions | 9.1 × 5.4 × 2.8 in (23.1 × 13.7 × 7.1 cm) |
| Weight | ~0.91 kg (just over 2 lbs) |
| Cut Speed | Up to 5.3 ips (inches per second) |
| Cutting Force | Not publicly disclosed (estimated ~600–800 gf based on Joy-family hardware) |
| Materials Count | 75+ |
| Bluetooth | Yes |

---

## Supported Blades & Tools

| Tool | Notes |
|------|-------|
| Premium Fine-Point Blade (integrated) | New integrated blade+housing — replace entire unit when worn |
| Scoring Tool | **New vs original Joy** |
| Foil Transfer Tool | ✓ |
| Pens & Markers | ✓ |

**No Deep Point Blade, No Rotary Blade, No Knife Blade, No Bonded Fabric Blade.**

---

## Mat / Cutting Dimensions

| Mode | Width | Length |
|------|-------|--------|
| With Standard Grip Mat (small) | 4.25 in (10.8 cm) | 6.25 in (15.9 cm) |
| With Standard/Light Grip Mat (long) | 4.25 in (10.8 cm) | 11.75 in (29.8 cm) |
| Smart Materials (individual shape) | 4.5 in (11.4 cm) | Up to 4 ft (1.2 m) |
| Smart Materials (repeated cuts) | 4.5 in (11.4 cm) | Up to ~20 ft (6 m) |
| Card Mat | Supported | — |

---

## Key Differences vs Original Cricut Joy

| Feature | Original Joy | Joy 2 |
|---------|-------------|-------|
| Print Then Cut | ✗ | **✓ (new)** |
| Scoring | ✗ | **✓ (new)** |
| Integrated blade housing | ✗ | **✓ (new design)** |
| Blade types | Fine Point | Fine Point |
| Mat sizes | Same | Same |
| Material count | 80 (from CSV) | ~75+ |
| Cutting force | ~600–800 gf (est.) | Similar (not disclosed) |

---

## Material Categories (Estimated)
Based on original Joy pattern + blade constraints (Fine Point only):
- Cardstock, Paper, Iron-On, Vinyl, Smart Materials, Infusible Ink, Board/Cardboard (light), Leather (thin), Others, Pens & Markers
- **No Fabric category** (no Rotary blade)
- Printable Materials category likely added due to Print Then Cut support

---

## ML / Data Implications

### Cut Settings Availability
Cricut does **not** publish per-material cut settings on their website. Settings for Joy 2 exist only in Design Space software.

### Recommended ML Strategy
1. **Proxy approach**: Joy 2 is architecturally identical to original Joy (same blade, similar force). Use Joy CSV data as a proxy for Joy 2 predictions, with a documented assumption note.
2. **If Joy 2 CSV available**: Merge into training data with `Machine = "Cricut Joy 2"` column value.
3. **Machine grouping option**: Treat "Cricut Joy" and "Cricut Joy 2" as a single "Joy-family" class to pool data.

### Pressure Range Estimate (from Joy CSV proxy)
| Category | Min | Max |
|----------|-----|-----|
| Cardstock | 223 | 315 |
| Iron-On | 95 | 246 |
| Paper | 100 | 330 |
| Smart Materials | 80 | 285 |
| Vinyl | 121 | 206 |
| Board/Cardboard | 226 | 286 |

*These are proxied from original Joy CSV. Actual Joy 2 values may differ slightly.*

---

## Sources
- https://cricut.com/en-us/cutting-machines/cricut-joy/cricut-joy-2/cricut-joy-2/2012138.html (product page)
- https://cricut.com/en-us/machine-comparison/machines-compare.html (official comparison, 2026-06-06)
- https://help.cricut.com/hc/en-us/articles/36545520180887-Cricut-Joy-2-Quick-Start-Guide (403 — not accessible)
- https://www.thecountrychiccottage.net/cricut-joy-2/ (review)
- https://heyletsmakestuff.com/cricut-joy-2/ (review)

## Open Questions
1. Do Joy 2 per-material cut settings differ from original Joy, or are they identical? (Only answerable via Design Space)
2. Does Print Then Cut support add a new material category (e.g., Printable Sticker Paper) not in original Joy?
3. Is the integrated blade design a universal change, or are legacy Joy blades also compatible?
