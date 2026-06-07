# Research: Cricut Brand Style
**Date**: 2026-06-06
**Researcher**: Claude (CSS extraction from cricut.com)
**Source CSS**: `/on/demandware.static/Sites-cricut-us-Site/-/en_US/v1780718527803/css/global.css`

## Summary
Cricut uses a clean, energetic design system built on Bootstrap 4 with a rich custom color palette. The unmistakable primary brand color is **Cricut Green (#1eb487)** — a vibrant emerald/mint green. The system defines ~40 named color variants with both accessible (dark, WCAG-compliant) and vivid (pastel-to-saturated) versions. Typography is a system sans-serif stack with tight letter-spacing on headings. The Japanese site is visually identical to English — same colors, same layout, translated copy only.

---

## Color Palette

### Primary Brand Color
| Name | Hex | Usage |
|------|-----|-------|
| **Cricut Green** | **#1eb487** | Primary CTA buttons, brand accents, logo — THE Cricut color |

### Accessible CTAs (dark, text-contrast safe on white)
| Name | Hex | Usage |
|------|-----|-------|
| Forrest | #0a3728 | Dark green — "success dark" equivalent |
| Accessible Green | #1a8163 | Accessible version of brand green |
| Accessible Aqua | #0c8487 | Accessible teal/cyan CTA |
| Teal | #015259 | Dark teal variant |
| Accessible Ocean | #245ccc | Accessible blue |
| Accessible Orchid | #a237b4 | Accessible purple |
| Accessible Salmon | #e03229 | Accessible red/error |
| Accessible Orange | #c75001 | Accessible orange/warning |
| Accessible Corn | #a06322 | Accessible yellow/gold |
| Accessible Stone | #6b6b6b | Accessible neutral gray |
| Indigo | #00237d | Dark navy blue |
| Plum | #8c005a | Deep magenta |
| Maroon | #872301 | Deep red |

### Vivid Brand Palette (full saturation)
| Name | Hex | Usage |
|------|-----|-------|
| Cricut Green | #1eb487 | Primary — emerald green |
| Aqua | #5fced1 | Bright teal/cyan |
| Ocean | #57a1ff | Bright blue |
| Orchid | #df9aff | Bright purple |
| Salmon | #ff9a95 | Bright coral/pink |
| Orange | #ff7d4f | Bright orange |
| Corn | #ffd25f | Bright yellow |
| Stone | #bebebe | Medium gray |

### Light / Background Palette
| Name | Hex | Usage |
|------|-----|-------|
| Shamrock | #ecf9f1 | Light green background |
| Mint | #e1f6f6 | Light teal background |
| Sky | #e5effd | Light blue background |
| Lavender | #fde4ff | Light purple background |
| Blush | #fee9e9 | Light red/error background |
| Peach | #ffeddc | Light orange/warning background |
| Fog | #f0f0f0 | Light gray |
| Alabaster | #fafafa | Near-white background |
| Light Gray | #e4e4e4 | Disabled state background |
| White | #ffffff | Page background |

### Semantic Colors
| Role | Hex | Source |
|------|-----|--------|
| Primary text | #121212 | Near-black (from homepage inline styles) |
| Secondary text | #6b6b6b | Disabled/muted text |
| Dividers | #bebebe | Stone / border color |
| Error | #e03229 | Accessible Salmon |
| Warning | #c75001 | Accessible Orange |
| Success | #1a8163 | Accessible Green |
| Background | #ffffff / #fafafa | Page / card backgrounds |

### Bootstrap 4 Base Variables (in `:root`)
These are the default Bootstrap variables — Cricut overrides them with custom btn-- classes:
```
--blue: #007bff   --indigo: #6610f2   --purple: #6f42c1
--pink: #e83e8c   --red: #dc3545      --orange: #fd7e14
--yellow: #ffc107 --green: #28a745    --teal: #20c997
--font-family-sans-serif: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif
```

---

## Typography

### Font Family
The CSS defines Bootstrap's system font stack as the base. No custom @font-face declarations were found in global.css. Cricut likely loads brand fonts via a separate lazy-load stylesheet or CDN.

**Recommended implementation**: Use the system font stack for maximum performance + Japanese compatibility:
```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans JP",
  "Helvetica Neue", Arial, sans-serif;
```

`"Noto Sans JP"` ensures clean Japanese character rendering without separate font loading.

### Heading Styles (from CSS)
| Element | font-weight | font-size | line-height | letter-spacing |
|---------|-------------|-----------|-------------|----------------|
| h1 | 500 | 3.6rem | 4.4rem | -0.07rem |
| h2 | 500 | 3.2rem | 4.2rem | -0.06rem |
| h3 | — | — | — | — |

Key characteristics:
- **Medium weight (500)** — not bold, gives a clean, modern feel
- **Tight letter-spacing (negative)** — sophisticated, editorial look
- **Generous line-height** — spacious and readable

### Body Text
- Base: 1rem (16px)
- Line-height: ~1.5 (Bootstrap default)
- Color: #121212

---

## UI Components

### Buttons
Cricut uses the Bootstrap `.btn` base class extended with `btn--{color-name}` modifiers.

**Primary CTA button** (use for main actions):
```css
background-color: #1eb487;  /* Cricut Green */
color: #ffffff;
border: none;
border-radius: 4px;          /* Bootstrap .25rem */
padding: 0.75rem 1.5rem;    /* estimated from visual */
font-weight: 600;
text-transform: uppercase;   /* common in CTAs */
letter-spacing: 0.05em;
```

**Hover state**: Darken by ~10% → approximately #19a07a

**Secondary button / outline**:
```css
background-color: transparent;
border: 2px solid #1eb487;
color: #1eb487;
```

**Disabled state**:
```css
background-color: #e4e4e4;
color: #6b6b6b;
border: none;
```

**Other notable button variants**:
- `btn--aqua` (#5fced1) — secondary/alternate actions
- `btn--accessible-green` (#1a8163) — accessible primary on light backgrounds
- `btn--forrest` (#0a3728) — dark mode / high-contrast contexts
- `btn--accessible-salmon` (#e03229) — error/destructive actions

### Cards / Panels
- Background: #ffffff or #fafafa
- Border: 1px solid #e4e4e4 or none (shadow-only cards)
- Border-radius: 8px (estimated from visual)
- Padding: 1.25rem (20px) — confirmed from CSS
- Box-shadow: subtle, 0 2px 8px rgba(0,0,0,0.08) (estimated)

### Navigation
- Background: #ffffff (white)
- Text color: #121212
- Active/hover: #1eb487 (Cricut Green underline or highlight)
- Height: ~60px desktop
- Horizontal mega-menu with dropdown panels

### Forms & Inputs
```css
border: 1px solid #bebebe;
border-radius: 4px;
padding: 0.75rem 1rem;
background: #ffffff;
color: #121212;
font-size: 1rem;
```
Focus state: border-color #1eb487, box-shadow 0 0 0 3px rgba(30,180,135,0.2)

### Tabs
- Inactive: #6b6b6b text, no background
- Active: #1eb487 text or underline indicator, weight 600
- Border-bottom: 2px solid #1eb487 on active tab

---

## Layout & Spacing

### Grid System
Bootstrap 4 responsive grid:
| Breakpoint | Width | Variable |
|------------|-------|----------|
| xs | 0px | `--breakpoint-xs` |
| sm | 576px | `--breakpoint-sm` |
| md | 768px | `--breakpoint-md` |
| lg | 992px | `--breakpoint-lg` |
| xl | 1200px | `--breakpoint-xl` |

### Spacing
- Base unit: 1rem (16px)
- Padding standard: 1.25rem (20px) — confirmed in CSS
- Section padding: 2rem–4rem vertical
- Card gap: 1rem–1.5rem in grid layouts

### Max content width
- ~1200px (xl breakpoint) — typical e-commerce layout

---

## Japanese Site Differences

The Japanese site (cricut.com/ja-jp) uses **identical visual design** to the English site:
- Same color palette
- Same button styles and layout
- Same hero images and photography
- Content is fully translated, not adapted — Japanese is a direct translation of English copy

**Design implications for our app:**
- No separate design language is needed for Japanese
- Font size may need slight increase for Japanese body text (14–16px minimum for readability)
- Japanese text is generally ~20% wider per character than Latin equivalents — reserve extra space in UI labels
- Use `"Noto Sans JP"` as the Japanese font fallback for crisp CJK rendering
- Consider `lang="ja"` on the `<html>` tag when Japanese is active for browser-level typography optimization

---

## Brand Voice & Aesthetic
- **Energetic, creative, crafty** — for makers and DIY enthusiasts
- **Clean and accessible** — large typography, clear CTAs, minimal clutter
- **Colorful but not overwhelming** — primary palette (green + neutrals) with accent colors used sparingly
- **Approachable and fun** — rounded UI elements, friendly copy tone
- **International-ready** — same visual system works across locales

---

## Implementation Guidelines for This Project

### CSS Variables to Define
```css
:root {
  /* Brand */
  --color-cricut-green: #1eb487;
  --color-cricut-green-dark: #1a8163;
  --color-cricut-green-light: #ecf9f1;
  --color-cricut-aqua: #5fced1;
  --color-cricut-aqua-accessible: #0c8487;
  
  /* Text */
  --color-text-primary: #121212;
  --color-text-secondary: #6b6b6b;
  --color-text-disabled: #949494;
  
  /* Backgrounds */
  --color-bg-page: #ffffff;
  --color-bg-card: #fafafa;
  --color-bg-hover: #f0f0f0;
  
  /* Borders */
  --color-border: #e4e4e4;
  --color-border-strong: #bebebe;
  
  /* Semantic */
  --color-error: #e03229;
  --color-error-bg: #fee9e9;
  --color-success: #1a8163;
  --color-success-bg: #ecf9f1;
  
  /* Typography */
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans JP",
    "Helvetica Neue", Arial, sans-serif;
  --font-size-base: 1rem;
  --font-weight-heading: 500;
  --letter-spacing-heading: -0.06rem;
  
  /* Spacing */
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.25rem;
  --space-xl: 2rem;
  
  /* Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-full: 9999px;
}
```

---

## Sources
- https://cricut.com/en-us — Homepage (HTML color extraction, 2026-06-06)
- https://cricut.com/on/demandware.static/Sites-cricut-us-Site/-/en_US/v1780718527803/css/global.css — Main CSS file (direct analysis, 2026-06-06)
- https://cricut.com/ja-jp — Japanese homepage (design comparison, 2026-06-06)

## Open Questions
1. Does Cricut use a custom web font (e.g. Brandon Grotesque, Futura) loaded via a JS bundle or lazy CSS not captured here? Visual inspection suggests the headings may use a slightly geometric sans-serif.
2. Are there specific icon assets or SVG components from Cricut's design system we should replicate?
3. Should the app use the vivid Cricut Green (#1eb487) or the accessible version (#1a8163) for primary CTAs? (Accessibility check: #1a8163 on white = 4.7:1 contrast ratio ✓; #1eb487 on white = 2.9:1 ✗ — fails WCAG AA)