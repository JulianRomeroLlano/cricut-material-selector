# ML Training Report — v4 (Sequential Transfer Learning)
**Date**: 2026-06-06
**Method**: Sequential pre-training on other machines → fine-tune on target
**Pre-train order**: Maker 3 → Explore 3 → Cricut Joy Xtra → Cricut Joy 2 → Cricut Joy (largest dataset first)
**Global pressure normalization**: log_mean=5.4964, log_std=0.9319

---

## Results (val = 20% of target machine)

| Machine | Rows | FT Epoch | Params | MAPE | MAE | Blade Acc | MC Acc |
|---------|------|----------|--------|------|-----|-----------|--------|
| Cricut Joy | 132 | 1735 | 1,032 | 30.3% | 48.4 | 1.000 | 0.857 |
| Cricut Joy 2 | 84 | 135 | 1,032 | 24.3% | 37.0 | 1.000 | 0.889 |
| Cricut Joy Xtra | 109 | 410 | 2,628 | 44.5% | 69.0 | 1.000 | 0.727 |
| Explore 3 | 243 | 5 | 4,012 | 30.1% | 50.7 | 1.000 | 0.640 |
| Maker 3 | 471 | 325 | 8,484 | 22.1% | 141.7 | 0.854 | 0.771 |

---

## Feature Vector (19 dims, global normalization)

| Feature | Dim | Normalization |
|---------|-----|--------------|
| Category one-hot | 11 | Binary |
| Thickness (mm) | 1 | log1p → MinMax [0,1] |
| Hardness (1–10) | 1 | (h-1)/9 |
| Is Bonded Fabric | 1 | Binary |
| GSM (g/m²) | 1 | log1p → MinMax [0,1] |
| Surface Texture | 1 | texture_map / 0.5 |
| Has Adhesive | 1 | Binary |
| Density (kg/m³) | 1 | log1p → MinMax [0,1] |
| Shore Hardness A | 1 | / 100 |
| **Total** | **19** | |

---

## Architecture Per Machine

| Machine | Architecture | Dropout | Weight Decay |
|---------|-------------|---------|-------------|
| Cricut Joy | 19→24→12→heads | 0.50 / 0.40 | 1e-3 |
| Cricut Joy 2 | 19→24→12→heads | 0.50 / 0.40 | 1e-3 |
| Cricut Joy Xtra | 19→48→24→heads | 0.45 / 0.35 | 5e-4 |
| Explore 3 | 19→64→32→heads | 0.40 / 0.30 | 5e-4 |
| Maker 3 | 19→96→48→24→heads | 0.35 / 0.30 | 5e-4 |

All models: BatchNorm1d on every hidden layer, AdamW, ReduceLROnPlateau (factor=0.5, patience=30).
Pre-training: LR=1e-3, early stopping patience=60.
Fine-tuning: LR=2e-4, early stopping patience=200.

---

## ONNX Files

All models share the **same input/output interface** (global normalization):
- Input: `features` float32 (batch, 19)
- `pressure_norm` → decode: `exp(x × 0.9319 + 5.4964)`
- `blade_logits` (batch, 5) → argmax → `blade_types_en[i]`
- `multicut_logits` (batch, 6) → argmax → `multicut_bucket_labels[i]`
