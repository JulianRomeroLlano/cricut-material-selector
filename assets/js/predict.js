/* predict.js — ONNX AI inference + feature encoding (no user-facing hardness input) */
(function () {
  /* Category-level defaults — mirrors Python training constants */
  const HARDNESS_DEFAULTS = {
    "Paper": 2, "Cardstock": 4, "Iron-On": 3, "Vinyl": 3,
    "Smart Materials": 3, "Printable Materials": 3, "Infusible Ink": 3,
    "Board/Cardboard": 7, "Leather": 6, "Fabric": 3, "Plastic": 4, "Others": 5,
  };
  const GSM_DEFAULTS = {
    "Paper": 80, "Cardstock": 216, "Iron-On": 100, "Vinyl": 120,
    "Smart Materials": 120, "Infusible Ink": 75, "Printable Materials": 100,
    "Fabric": 150, "Plastic": 150, "Others": 200,
  };
  const DENSITY_DEFAULTS = {
    "Paper": 750, "Cardstock": 800, "Iron-On": 1050, "Vinyl": 1300,
    "Smart Materials": 1300, "Infusible Ink": 800, "Printable Materials": 900,
    "Board/Cardboard": 850, "Leather": 900, "Fabric": 280, "Plastic": 1350, "Others": 150,
  };
  const SHORE_DEFAULTS = {
    "Paper": 15, "Cardstock": 30, "Iron-On": 45, "Vinyl": 65,
    "Smart Materials": 65, "Infusible Ink": 20, "Printable Materials": 25,
    "Board/Cardboard": 65, "Leather": 55, "Fabric": 10, "Plastic": 70, "Others": 40,
  };

  function getGsm(category, mm) {
    if (category === "Board/Cardboard") return mm * (DENSITY_DEFAULTS["Board/Cardboard"] || 850);
    if (category === "Leather")         return mm * (DENSITY_DEFAULTS["Leather"] || 900);
    return GSM_DEFAULTS[category] || 100;
  }

  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  function buildFeatures(pp, { category, thicknessMm }) {
    const features = new Float32Array(pp.feature_dim);

    /* 0–10: category one-hot */
    const ci = pp.categories.indexOf(category);
    if (ci >= 0) features[ci] = 1.0;

    /* 11: thickness log1p MinMax */
    const tLog = Math.log1p(thicknessMm);
    features[11] = clamp01((tLog - pp.thickness_log_min) /
                           (pp.thickness_log_max - pp.thickness_log_min + 1e-9));

    /* 12: hardness (h-1)/9 — inferred from category */
    const h = HARDNESS_DEFAULTS[category] || 5;
    features[12] = clamp01((h - 1) / 9);

    /* 13: is_bonded_fabric — unknown material: 0 */
    features[13] = 0.0;

    /* 14: GSM log1p MinMax */
    const gsm  = getGsm(category, thicknessMm);
    const gLog = Math.log1p(gsm);
    features[14] = clamp01((gLog - pp.gsm_log_min) /
                           (pp.gsm_log_max - pp.gsm_log_min + 1e-9));

    /* 15: surface_texture plain → 0 */
    features[15] = 0.0;

    /* 16: has_adhesive → 0 */
    features[16] = 0.0;

    /* 17: density log1p MinMax */
    const density = DENSITY_DEFAULTS[category] || 500;
    const dLog    = Math.log1p(density);
    features[17] = clamp01((dLog - pp.density_log_min) /
                           (pp.density_log_max - pp.density_log_min + 1e-9));

    /* 18: Shore A / 100 */
    features[18] = clamp01((SHORE_DEFAULTS[category] || 40) / 100);

    return features;
  }

  const _sessions = {};
  let _pp = null;

  async function loadPreprocessor() {
    if (_pp) return _pp;
    const r = await fetch("assets/model/preprocessor.json");
    if (!r.ok) throw new Error("preprocessor.json not found");
    _pp = await r.json();
    return _pp;
  }

  async function getSession(slug) {
    if (_sessions[slug]) return _sessions[slug];
    const sess = await ort.InferenceSession.create(
      "assets/model/material_predictor_" + slug + ".onnx",
      { executionProviders: ["wasm"] }
    );
    _sessions[slug] = sess;
    return sess;
  }

  async function predict({ machine, category, thicknessMm }) {
    const pp   = await loadPreprocessor();
    const info = pp.machines[machine];
    if (!info) throw new Error("Unknown machine: " + machine);

    const sess     = await getSession(info.slug);
    const features = buildFeatures(pp, { category, thicknessMm });
    const tensor   = new ort.Tensor("float32", features, [1, pp.feature_dim]);
    const out      = await sess.run({ features: tensor });

    /* Pressure */
    const pNorm    = out["pressure_norm"].data[0];
    const pressure = Math.round(Math.exp(pNorm * pp.pressure_log_std + pp.pressure_log_mean));

    /* Blade — constrain to machine-compatible blades */
    const bladeLogits = Array.from(out["blade_logits"].data);
    const compatible  = (pp.notes && pp.notes.compatible_blades &&
                         pp.notes.compatible_blades[machine]) || null;
    let bladeIdx = 0;
    let best = -Infinity;
    pp.blade_types_en.forEach((en, i) => {
      if (compatible && !compatible.includes(en)) return;
      if (bladeLogits[i] > best) { best = bladeLogits[i]; bladeIdx = i; }
    });
    const bladeEn = pp.blade_types_en[bladeIdx] || "Fine-Point Blade";
    const bladeJp = pp.blade_types_jp[bladeIdx] || "ファインポイントブレード";

    /* Multi-cut */
    const mcLogits = Array.from(out["multicut_logits"].data);
    const mcIdx    = mcLogits.indexOf(Math.max(...mcLogits));
    const mcLabels = ["1×", "2×", "3×", "4–5×", "6–8×", "10+×"];
    const multicut = mcLabels[mcIdx] || "1×";

    return { pressure, bladeEn, bladeJp, multicut };
  }

  window.CricutPredict = { predict, loadPreprocessor };
})();