# Research: Cricut Explore 5 Specifications
**Date**: 2026-06-06
**Researcher**: Claude (WebSearch + WebFetch from cricut.com, cuttabl.co, jennifermaker.com)

## Summary
The Cricut Explore 5 is the 2026 successor to the Explore 3. It is **30% more compact** while keeping the same 12-inch cutting width and Smart Materials (12 ft) capability. Key upgrades include a new integrated blade design, a **Wavy Blade** (decorative edge cutting), and a Scoring Tool. The Bonded Fabric Blade remains compatible. Cutting force is cited at ~1,000 gf in one source vs. Explore 3's ~2,000 gf — this reduction aligns with the compact redesign. No per-material cut settings are publicly available. **For ML purposes, Explore 5 is a direct successor to Explore 3 with the same blade set plus Wavy Blade; pressure values likely differ slightly from Explore 3 due to force reduction.**

---

## Machine Specifications

| Feature | Value |
|---------|-------|
| Release | 2026 |
| Form Factor | 30% more compact than Explore 3/Air 2 |
| Cutting Width | 12 inches (30.5 cm) — unchanged |
| Cutting Force | ~1,000 gf (cited; not confirmed officially) |
| Explore 3 Force (reference) | ~2,000 gf |
| Max Material Thickness | Up to 1.5 mm (with Deep Cutting Tool) |
| Materials Count | 100+ |
| Compatible Tools | 6 total |
| Bluetooth | Yes |

*Note: The 1,000 gf force spec is from a third-party source comparing it to Maker 4 (4,000 gf). Cricut does not officially publish force specs. Explore 3 was documented at ~2,000 gf. Explore 5 being lower is plausible for the compact redesign, but should be verified against actual Design Space settings when available.*

---

## Supported Blades & Tools

| Tool | Notes |
|------|-------|
| Premium Fine-Point Blade (integrated) | New integrated blade+housing design |
| Deep Point Blade | ✓ — cuts up to 1.5 mm thick |
| Bonded Fabric Blade | ✓ — confirmed compatible (same as Explore 3) |
| Scoring Tool | ✓ |
| **Wavy Blade** | **New in Explore 5** — decorative edge cuts |
| Foil Transfer Tool | ✓ |

**No Rotary Blade, No Knife Blade** (Maker-exclusive tools)

Total: 6 compatible tools (Fine Point, Deep Point, Bonded Fabric, Scoring, Wavy, Foil Transfer)

---

## Mat / Cutting Dimensions

| Mode | Width | Length |
|------|-------|--------|
| Standard Grip Mat | 12 in (30.5 cm) | 12 in (30.5 cm) |
| Extended Mat | 12 in (30.5 cm) | 24 in (61 cm) |
| Smart Materials (matless) | 12 in (30.5 cm) | Up to 12 ft (3.6 m) |

---

## Key Differences vs Explore 3

| Feature | Explore 3 | Explore 5 |
|---------|-----------|-----------|
| Cutting force | ~2,000 gf | ~1,000 gf (cited) |
| Form factor | Larger | **30% more compact** |
| Integrated blade | No | **Yes** |
| Wavy Blade | No | **Yes (new)** |
| Deep Point Blade | ✓ | ✓ |
| Bonded Fabric Blade | ✓ | ✓ |
| Smart Materials | 12 ft | 12 ft |
| Material count | 140 (CSV) | 100+ |
| Max thickness | ~1.5 mm | 1.5 mm |
| Print Then Cut | ✓ | ✓ |
| Scoring | ✓ | ✓ |

---

## Material Categories (Estimated)
Same 11 categories as Explore 3 (Bonded Fabric Blade confirmed):
Board/Cardboard, Cardstock, Fabric (bonded only), Infusible Ink, Iron-On, Leather, Others, Paper, Printable Materials, Smart Materials, Vinyl

**No unbonded Fabric** (Rotary blade is Maker-only)

---

## ML / Data Implications

### Cut Settings Availability
Cricut does **not** publish per-material cut settings on their website. Settings for Explore 5 exist only in Design Space software.

### Critical Consideration: Reduced Cutting Force
If Explore 5's force is truly ~1,000 gf (half of Explore 3's ~2,000 gf), pressure values in Design Space **will likely be different** from Explore 3 CSV data. The machine compensates by applying higher internal pressure per unit, but the effective cut settings the user sees may be recalibrated.

### Recommended ML Strategy
1. **Proxy approach**: Explore 5 has the same blade set as Explore 3. Use Explore 3 CSV data as proxy, but document the force reduction caveat.
2. **If Explore 5 CSV available**: Merge into training data with `Machine = "Cricut Explore 5"` column value. Expect pressure values to differ from Explore 3 by a calibration offset.
3. **Machine grouping option**: Treat "Explore 3" and "Explore 5" as "Explore-family" class if force/pressure difference proves to be a simple linear scaling.

### Pressure Range Estimate (from Explore 3 proxy)
| Category | Min | Max |
|----------|-----|-----|
| Board/Cardboard | 300 | 350 |
| Cardstock | 180 | 320 |
| Fabric (bonded) | 285 | 321 |
| Infusible Ink | 226 | 226 |
| Iron-On | 80 | 268 |
| Leather | 215 | 325 |
| Others | 70 | 350 |
| Paper | 70 | 345 |
| Printable Materials | 70 | 277 |
| Smart Materials | 121 | 300 |
| Vinyl | 76 | 190 |

*Proxied from Explore 3 CSV. Actual Explore 5 values may differ due to reduced cutting force.*

---

## Sources
- https://cricut.com/en-us/cutting-machines/cricut-explore/cricut-explore-5/cricut-explore-5/2012400.html (product page)
- https://cricut.com/en-us/machine-comparison/machines-compare.html (official comparison, 2026-06-06)
- https://cricut.com/blog/introducing-cricut-explore-5/ (Cricut blog announcement)
- https://cuttabl.co/blog/cricut-explore-5-review (review — 1,000 gf force cited)
- https://help.cricut.com/hc/en-us/articles/36546544830615-Cricut-Explore-5-Quick-Start-Guide (403 — not accessible)
- https://jennifermaker.com/cricut-explore-5-cricut-joy-2/ (403 — not accessible)

## Open Questions
1. Is the ~1,000 gf force spec accurate, or is it a mis-citation? What are the actual Design Space pressure values for common materials (vinyl, cardstock, iron-on) on Explore 5 vs. Explore 3?
2. Does the Wavy Blade have separate material settings in Design Space, or does it reuse Fine Point settings?
3. Does the 30% compact redesign affect mat compatibility (standard Explore 3 mats should still fit)?
4. Material count decreased from 140 (Explore 3 CSV) to "100+" — were materials removed from Explore 5 support, or is "100+" an approximate marketing number?
