# Cricut Material Selector — Project Orchestrator

## Project Overview
A client-side web app that provides Cricut cutting settings for any material — including materials not in the official list — using an MLP neural network exported to ONNX and run entirely in the browser.

## Key Facts
- **Target machines**: Cricut Maker 3, Explore 3, Cricut Joy
- **ML approach**: MLP (PyTorch → ONNX → onnxruntime-web)
- **Model inputs**: material category, thickness (mm), hardness/stiffness score
- **Model outputs**: cutting pressure, blade type, multi-cut setting, compatible machines
- **Deployment**: fully static (GitHub Pages / Netlify)
- **Languages**: English (default) + Japanese (auto-detected from browser language)
- **Style**: Cricut international brand style

## Data
All material data lives in `assets/data/`:
- `Material List (Combined).csv` — merged dataset (533 rows, 7 columns: Machine, Category, Material Name JP/EN, Cutting Pressure, Multi-Cut, Blade Type)
- Individual per-machine CSVs (Maker 3, Explore 3, Cricut Joy)

## Project Methodology
Track all progress in `.claude/project_history.json`.

| Step | Name | Status |
|------|------|--------|
| 1 | Deep Research | Pending |
| 2 | ML Model Training & ONNX Export | Pending |
| 3 | Test Specification | Pending |
| 4 | UI / Landing Page Design | Pending |
| 5 | App Development | Pending |
| 6 | Testing & Fixes | Pending |

## Skills (Custom Slash Commands)
Use these skills in order during the project:

- `/researcher` — Deep research on Cricut branding, machine specs, material science. Always saves output to `.claude/research/`.
- `/programmer` — Implements Python ML pipeline (training, ONNX export) and the web app (HTML/CSS/JS).
- `/designer` — Designs UI following Cricut international style. Must be used only after research is complete and saved.

## Python Environment
Located at `venv/` — PyTorch with CUDA 12.x, ONNX, onnxruntime.

Activate with: `source venv/bin/activate`

## Constraints
- Do not make code changes without user authorization
- Do not generate new data (ML training data, material properties) without authorization
- Do not delete any existing code without authorization
- Always update `.claude/project_history.json` when a step starts or completes
- Always save all research results as Markdown files in `.claude/research/`