# Research: Cricut Joy Xtra Specifications
**Date**: 2026-06-06
**Researcher**: Claude (WebSearch + WebFetch from cricut.com, abbikirstencollections.com, extraordinarychaos.com, craft-e-corner.com)

## Summary
The Cricut Joy Xtra (released 2023) is a larger sibling of the Cricut Joy, positioned between Joy and Explore 3. Its defining differentiator is a **wider 8.5-inch cutting width** (vs. Joy's 4.5 in), making it the widest of the Joy family. It retains Fine Point-only blade support. Cutting force is cited at 350 gf in one source, lower than the Explore line (~2,000 gf). No per-material cut settings are publicly available. **For ML purposes, Joy Xtra is most similar to original Joy but may handle slightly thicker/denser materials due to its wider build and pressure spec.**

---

## Machine Specifications

| Feature | Value |
|---------|-------|
| Release | 2023 |
| Dimensions | 12.5 × 6 × 5.5 in (31.8 × 15.2 × 14 cm) |
| Weight | 6 lbs (2.7 kg) |
| Cutting Force | ~350 gf (cited in one technical review; not confirmed by Cricut officially) |
| Materials Count | 75+ |
| Bluetooth | Yes |

---

## Supported Blades & Tools

| Tool | Notes |
|------|-------|
| Premium Fine-Point Blade + Housing | Only cutting blade |
| Foil Transfer Tool | ✓ |
| Pens & Markers | ✓ |

**No Deep Point Blade, No Rotary Blade, No Knife Blade, No Bonded Fabric Blade.**

Note: The 350 gf force spec means Fine Point can cut slightly more than original Joy but nowhere near Explore 3 (~2,000 gf) capability.

---

## Mat / Cutting Dimensions

| Mode | Width | Length |
|------|-------|--------|
| With 8.5×12 mat | 8.25 in (21.0 cm) | 11.5 in (29.2 cm) |
| Smart Materials (individual shape) | 8.5 in (21.6 cm) | Up to 4 ft (1.2 m) |
| Smart Materials (repeated cuts) | 8.5 in (21.6 cm) | Extended length |

**The 8.5-inch width is the key advantage over Joy (4.5 in) and Joy 2 (4.5 in). Same 4 ft Smart Material length as other Joy models.**

---

## Print Then Cut
**Conflicting information across sources:**
- Official Cricut comparison table (2026-06-06): **Not listed as supported**
- Third-party reviews: Joy Xtra has an "Easy Printables sensor" for cutting printable sticker paper
- Cricut Design Space Print Then Cut (full feature): **Likely not supported**

**Conclusion**: Joy Xtra has a limited Easy Printables feature (sticker paper cutting) but does NOT support full Design Space Print Then Cut with registration marks. The Joy 2 (2026) added full Print Then Cut.

---

## Key Differences vs Joy and Explore 3

| Feature | Cricut Joy | Joy Xtra | Explore 3 |
|---------|-----------|----------|-----------|
| Cut width | 4.5 in | **8.5 in** | 12 in |
| Smart Material length | 4 ft | 4 ft | 12 ft |
| Cutting force | ~600 gf (est.) | ~350 gf | ~2,000 gf |
| Blade types | Fine Point | Fine Point | Fine + Deep + Bonded Fabric |
| Fabric support | None | None | Bonded only |
| Print Then Cut | No | Partial (Easy Printables) | Yes |
| Mat size | 4.5×12 | 8.5×12 | 12×12 / 12×24 |

*Note: Joy Xtra 350 gf is lower than Joy's estimated 600 gf — possibly a different measurement methodology (static vs. dynamic force). Practical cutting capability may be similar.*

---

## Material Categories (Estimated)
Based on Joy-family constraints (Fine Point only, no Fabric):
- Cardstock, Paper, Iron-On, Vinyl, Smart Materials, Infusible Ink, Board/Cardboard (very light only), Others
- **No Fabric category**
- Limited leather support (thin only)

---

## ML / Data Implications

### Cut Settings Availability
Cricut does **not** publish per-material cut settings on their website. Settings for Joy Xtra exist only in Design Space software.

### Recommended ML Strategy
1. **Proxy approach**: Joy Xtra has similar blade support to Joy. Use Joy CSV data as a proxy, potentially adjusting pressure predictions slightly upward for thicker materials.
2. **If Joy Xtra CSV available**: Merge into training data with `Machine = "Cricut Joy Xtra"` column value.
3. **Machine grouping option**: Treat Joy, Joy 2, and Joy Xtra as "Joy-family" class, acknowledging the wider cut width difference doesn't affect pressure values.

### Pressure Range Estimate (from Joy CSV proxy)
Same as Cricut Joy CSV ranges — see `research_cricut_machines_2026-06-06.md` for original Joy pressure table.

---

## Sources
- https://cricut.com/en-us/cutting-machines/cricut-joy-xtra/cricut-joy-xtra (product page)
- https://cricut.com/en-us/machine-comparison/machines-compare.html (official comparison, 2026-06-06)
- https://www.abbikirstencollections.com/cricut-joy-xtra/ (review — 350 gf spec cited)
- https://extraordinarychaos.com/the-cricut-joy-vs-cricut-joy-xtra-which-cricut-machine-is-best.html (comparison review)
- https://www.craft-e-corner.com/blogs/project-inspiration/cricut-cutting-machine-comparison-joy-joy-xtra-explore-3 (comparison review)
- https://teacupsandthings.com/the-cricut-joy-vs-joy-xtra-which-one-should-i-get-machine-comparison/ (comparison review)

## Open Questions
1. Does 350 gf (Joy Xtra) use the same measurement scale as Joy CSV pressure values (80–330)? If so, Joy Xtra max CSV pressure would be around 350 — slightly higher than Joy's 330 max.
2. Does the wider 8.5 in cutting path affect any material settings (e.g., alignment compensation)?
3. Is there a Joy Xtra–specific material list in Design Space distinct from Joy's list?
4. What exactly does "Easy Printables sensor" enable vs. full Print Then Cut?