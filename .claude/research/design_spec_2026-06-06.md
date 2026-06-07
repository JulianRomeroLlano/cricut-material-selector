# Design Specification — Cricut Material Selector Web App
**Version:** 1.0  
**Date:** 2026-06-06  
**Style references:** Cricut international brand (cricut.com), community layout inspiration (tamayuzucraft)

---

## 1. Design Philosophy

The layout takes structural cues from the tamayuzucraft reference (machine tabs → filters → cards) because that pattern works extremely well on mobile for this use case. Everything visual — colors, typography, weight, spacing — is replaced with Cricut's actual brand system.

**Three principles:**
1. **Brand-correct** — Cricut Green (#1eb487) as the signature accent; no cream/teal palette from the reference site.
2. **Bilingual-first** — EN and JP feel equally native; not a translation bolted on.
3. **Two modes, one app** — Browse (known materials) and Predict (ML) are distinct but flow from the same surface without a navigation jump.

---

## 2. Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--green` | `#1eb487` | Primary accent: active tabs, selected chips, CTAs, card left-border |
| `--green-accessible` | `#1a8163` | Text on white that must meet WCAG AA (button labels, links) |
| `--green-light` | `#ecf9f1` | Shamrock — tinted row backgrounds, subtle input fills, stats bar |
| `--text` | `#121212` | Body text |
| `--text-secondary` | `#6b6b6b` | Labels, metadata, placeholder text |
| `--border` | `#e4e4e4` | Dividers, chip borders, card outlines |
| `--surface` | `#fafafa` | Alabaster — page background |
| `--card` | `#ffffff` | Card and panel backgrounds |
| `--error` | `#e03229` | Validation error text |
| `--shadow` | `rgba(30,180,135,0.10)` | Green-tinted box shadows |

> Cricut brand purples, blues, and salmons are available as category-color accents on cards — one hue per category chip, pulled from the brand palette.

---

## 3. Typography

```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
             "Noto Sans JP", "Helvetica Neue", Arial, sans-serif;
```

`"Noto Sans JP"` loaded from Google Fonts at `weights: 400, 500, 700` — loaded only if `lang === "ja"` is detected, otherwise system stack only (performance).

| Element | Weight | Size (mobile) | Size (desktop) | Notes |
|---------|--------|--------------|----------------|-------|
| App title | 500 | 15px | 18px | Cricut medium-weight convention |
| Card material name | 700 | 13px | 14px | Bold for scannability |
| Spec value (pressure, etc.) | 700 | 14px | 15px | Green `--green-accessible` color |
| Spec label | 500 | 10px | 10px | Uppercase, `--text-secondary` |
| Filter chip | 600 | 11px | 12px | |
| Body / search results count | 400 | 13px | 13px | |
| Form labels | 500 | 13px | 13px | |

---

## 4. Page Layout (Mobile-First)

### 4.1 Overall Structure

```
┌─────────────────────────────────────────┐
│  HEADER (sticky)                        │
│  [≡ logo] Cricut Material Selector  [JP/EN] │
├─────────────────────────────────────────┤
│  MODE TABS                              │
│  [ Browse Materials ]  [ ✦ Predict ]   │
├─────────────────────────────────────────┤
│                                         │
│  ── BROWSE mode ──────────────────────  │
│  MACHINE TABS                           │
│  [ Joy ][ Joy 2 ][ Joy Xtra ]…         │
│                                         │
│  STATS BAR  "1 039 materials · 5 machines" │
│                                         │
│  SEARCH BAR  🔍 [Search materials…    ] │
│                                         │
│  CATEGORY CHIPS                         │
│  [All] [Vinyl] [Iron-On] [Cardstock]…  │
│                                         │
│  BLADE CHIPS                            │
│  [All] [Fine-Point] [Rotary]…          │
│                                         │
│  SORT  ↑ Pressure  ↓ Pressure          │
│                                         │
│  RESULT COUNT  "243 results"            │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ▌ Adhesive Vinyl                │    │
│  │ ▌ Vinyl  ·  Fine-Point Blade   │    │
│  │ ▌                               │    │
│  │ [Pressure] [Multi-Cut] [Blade]  │    │
│  │  170        Off        Fine-Pt  │    │
│  └─────────────────────────────────┘    │
│  … more cards …                         │
│                                         │
│  ── PREDICT mode ─────────────────────  │
│  MACHINE: [Explore 3 ▾]                │
│  CATEGORY: [Vinyl ▾]                   │
│  THICKNESS: [0.08] mm                  │
│  HARDNESS: [──●────] 6                 │
│  [  Predict Cut Settings  ]            │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ✦ Predicted Settings            │    │
│  │ Pressure  Blade       Multi-Cut │    │
│  │    170    Fine-Point    Off     │    │
│  │ ⚠ AI estimate — verify with    │    │
│  │   a test cut before production  │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
[ ↑ scroll-to-top FAB, bottom-right ]
```

### 4.2 Header (sticky, `position: sticky; top: 0; z-index: 100`)

```
┌───────────────────────────────────────────┐
│  [Cricut ✦ logo mark]  Material Selector  │   [EN / JP]  │
└───────────────────────────────────────────┘
```

- Background: `--green` `#1eb487` with a subtle `linear-gradient(135deg, #1eb487 0%, #1a8163 100%)`
- Height: 52px mobile, 60px desktop
- Title text: white, 15px / 500 weight
- `[EN / JP]` toggle: pill button, white outline, toggles `lang` attribute on `<html>` and re-renders all i18n strings
- Cricut logo mark: simple `✦` or an SVG spark (the Cricut "spark" icon motif) at 22px, white

### 4.3 Mode Tabs

Two tabs directly below the header:

```
[ Browse Materials ]   [ ✦ Predict ]
```

- Selected tab: white text on `--green` pill, `border-radius: 20px`
- Inactive tab: `--green-accessible` text on white, green border
- Tab container: padding 12px 16px, background white, bottom `1px solid --border`

### 4.4 Machine Tabs (Browse mode only)

Horizontally scrollable tab strip:

```
[Joy] [Joy 2] [Joy Xtra] [Explore 3] [Maker 3] [All]
```

- Active: `--green` background, white text, `border-radius: 10px`
- Inactive: transparent background, `--text-secondary` color
- Container: `background: --green-light`, padding 6px 16px, `border-radius: 12px`, margin 12px 16px
- Overflow: `scroll` with `-webkit-overflow-scrolling: touch`, no visible scrollbar

### 4.5 Stats Bar

```
1 039 materials  ·  5 machines  ·  11 categories
```

- Background: `--green-light` (`#ecf9f1`)
- Font: 11px, `--text-secondary`, centered, padding 6px 16px

### 4.6 Search Bar

```
🔍  [Search materials…                          ]
```

- Full width, `border-radius: 24px`
- Border: `2px solid --border`, focus: `2px solid --green`
- Background: white
- Placeholder: `--text-secondary`
- Icon: search emoji or SVG, non-interactive, positioned absolutely left

### 4.7 Filter Chips

Two rows of horizontally scrollable chips:

**Row 1 — Category:**
```
[All] [Vinyl] [Iron-On] [Cardstock] [Paper] [Fabric] [Others] …
```

**Row 2 — Blade:**
```
[All blades] [Fine-Point] [Deep-Point] [Rotary] [Bonded Fabric] [Knife]
```

Chip style:
- Default: white background, `1.5px solid --border`, `--text-secondary`, `border-radius: 16px`, `padding: 5px 12px`
- Active: `--green` background, white text, `border-color: --green`
- Hover: border `--green`, text `--green-accessible`

Category chips get a small colored dot indicator using brand palette hues (Vinyl = Ocean blue, Iron-On = Orchid, Leather = Salmon, Fabric = Aqua, Paper = Corn, Cardstock = Stone, Board = Peach, Others = Fog).

### 4.8 Sort Controls

Single row, right-aligned:

```
Sort: [ Default ] [ ↑ Pressure ] [ ↓ Pressure ]
```

- Same chip style as filters but smaller (10px font, `padding: 4px 10px`)
- Active sort: filled with `--green-accessible` (darker, to distinguish from category chips)

### 4.9 Material Card

```
┌──────────────────────────────────────────┐
│▌ Adhesive Vinyl                  [Vinyl] │  ← card-name + category badge
│  ファインポイントブレード                    │  ← JP name (if JP mode) or blade note
│                                          │
│  [  170  ] [ Fine-Point ] [   Off   ]    │  ← spec pills
│   Pressure   Blade          Multi-Cut    │
└──────────────────────────────────────────┘
```

- Background: `--card` white
- Left border: `4px solid --green` (exact width), `border-radius: 0 12px 12px 0` on the right side
- Box shadow: `0 2px 8px var(--shadow)`
- Hover: `transform: translateY(-1px)`, shadow increases
- Category badge: small pill, right of title, colored per category (see color map below)
- Spec pills: `background: --green-light`, `border-radius: 8px`, `padding: 6px 10px`
  - Label: 9px uppercase `--text-secondary`
  - Value: 14px bold `--green-accessible`

**Category badge colors:**

| Category | Background | Text |
|----------|-----------|------|
| Vinyl | `#e5effd` | `#245ccc` |
| Iron-On | `#fde4ff` | `#a237b4` |
| Cardstock | `#f0f0f0` | `#6b6b6b` |
| Paper | `#fff3cd` (Peach tint) | `#a06322` |
| Fabric | `#e1f6f6` | `#0c8487` |
| Leather | `#fee9e9` | `#e03229` |
| Board/Cardboard | `#ffeddc` | `#c75001` |
| Others | `#f0f0f0` | `#6b6b6b` |
| Infusible Ink | `#ecf9f1` | `#1a8163` |
| Smart Materials | `#ecf9f1` | `#1a8163` |
| Printable Materials | `#ecf9f1` | `#1a8163` |

### 4.10 Prediction Form

Full-width card panel below the mode tab area:

```
┌──────────────────────────────────────┐
│ Machine           [Explore 3      ▾] │
│ Material category [Vinyl          ▾] │
│ Thickness         [0.08] mm          │
│ Hardness          [──●────────] 6    │
│                                      │
│  [    Predict Cut Settings    ]      │
└──────────────────────────────────────┘
```

- Background: white card, same shadow as material cards
- Labels: 13px, 500 weight, `--text`
- Selects/inputs: full width, `border-radius: 8px`, `border: 1.5px solid --border`, focus: `--green`
- Thickness input: `<input type="number" min="0.01" max="60" step="0.01">` + `mm` unit label inline
- Hardness: `<input type="range" min="0" max="10">` with live value display, plus optional number input alongside
- Submit button: full width, `background: --green`, white text, `border-radius: 24px`, `font-weight: 700`, 48px height, hover: `--green-accessible`
- Loading state: button shows spinner, disabled

**Prediction Result Card:**

```
┌──────────────────────────────────────┐
│  ✦  Predicted Cut Settings           │
│                                      │
│  [ 170  ] [ Fine-Point ]  [  Off  ] │
│  Pressure   Blade           Multi-Cut│
│                                      │
│  ⚠ AI estimate — always verify with │
│    a test cut before production.     │
└──────────────────────────────────────┘
```

- Left border: `4px solid --green` (same as material cards)
- `✦` icon in Cricut green
- Same spec pill layout as material cards
- Disclaimer: `--text-secondary`, italic, 11px, icon `⚠` in `--green-accessible`
- Appears below the form with a fade-in animation (`opacity 0→1, translateY 8px→0, 200ms`)

### 4.11 Empty State

```
      ✦
  No materials found
  Try a different search or filter.
```

- Centered, padding 60px 20px
- Spark icon: `--green`, 36px
- Title: 16px, `--text`
- Sub: 13px, `--text-secondary`

### 4.12 Scroll-to-Top FAB

- Fixed bottom-right: `position: fixed; bottom: 24px; right: 20px`
- 48×48px circle, `background: --green`, white `↑`
- Appears only when scrolled > 300px
- `box-shadow: 0 4px 12px rgba(30,180,135,0.35)`
- Hover: `transform: scale(1.08)`

---

## 5. Responsive Breakpoints

| Breakpoint | Width | Changes |
|-----------|-------|---------|
| Mobile | 0–599px | Single column, all chips horizontal-scroll |
| Tablet | 600–1023px | Machine tabs may show all without scroll; form + results side-by-side on Predict tab |
| Desktop | 1024px+ | Max content width 900px, centered; cards may go 2-column grid |

---

## 6. Localization UI

- `<html lang="en">` default; set to `"ja"` when `navigator.language.startsWith("ja")`
- All user-visible strings defined in a `const I18N = { en: {…}, ja: {…} }` object in `app.js`
- Header `[EN / JP]` button sets a `localStorage.lang` override and re-calls `applyI18n()`
- Japanese: material name primary = `name_jp`, EN name shown as `font-size: 10px` subtitle
- English: material name primary = `name_en`, JP name omitted

### Key i18n string keys

```
app_title, mode_browse, mode_predict, machine_all,
search_placeholder, filter_all_categories, filter_all_blades,
sort_default, sort_asc, sort_desc,
result_count (singular/plural),
form_machine, form_category, form_thickness, form_hardness,
btn_predict, btn_predicting,
result_heading, disclaimer,
err_thickness_required, err_thickness_range, err_model_load,
empty_state_title, empty_state_sub,
blade_fine, blade_deep, blade_rotary, blade_bonded, blade_knife,
multicut_off, multicut_2x, multicut_3x, multicut_4x, multicut_6x, multicut_10x
```

---

## 7. Animation & Micro-interactions

| Event | Animation |
|-------|-----------|
| Tab switch | Cards fade-out (100ms) → re-render → fade-in (150ms) |
| Chip activate | `background-color` transition 150ms ease |
| Card hover | `box-shadow` + `translateY(-1px)` 150ms ease |
| Prediction result appear | `opacity 0→1` + `translateY(8px→0)` 200ms ease-out |
| Scroll-to-top FAB show/hide | `opacity 0→1` 200ms |

Respect `prefers-reduced-motion`: all transitions skipped if user preference set.

---

## 8. File Structure (Step 5 target)

```
index.html
assets/
  css/
    styles.css          ← all styles, CSS custom properties
  js/
    app.js              ← main app (search, filter, render, i18n)
    predict.js          ← ONNX inference, feature encoding
    i18n.js             ← EN + JP string tables
  data/
    materials.json      ← converted from combined CSV
  model/
    preprocessor.json
    material_predictor_cricut_joy.onnx
    material_predictor_cricut_joy2.onnx
    material_predictor_cricut_joy_xtra.onnx
    material_predictor_explore3.onnx
    material_predictor_maker3.onnx
  fonts/                ← (optional: local Noto Sans JP subset)
```

`materials.json` is generated from the combined CSV by a small Python build script — no runtime CSV parsing in the browser.

---

*Next: Step 5 — App Development (implement this spec)*