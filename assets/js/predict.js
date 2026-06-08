/* predict.js — ONNX AI inference v2: material name embedding + physics features */
(function () {
  /* Category fallback defaults (used when material name is unknown) */
  const GSM_DEFAULTS = {
    "Paper": 80, "Cardstock": 176, "Iron-On": 100, "Vinyl": 120,
    "Smart Materials": 120, "Infusible Ink": 75, "Printable Materials": 100,
    "Board/Cardboard": 750, "Leather": 900, "Fabric": 150, "Plastic": 150, "Others": 200,
  };
  const DENSITY_DEFAULTS = {
    "Paper": 750, "Cardstock": 800, "Iron-On": 1050, "Vinyl": 1300,
    "Smart Materials": 1300, "Infusible Ink": 800, "Printable Materials": 900,
    "Board/Cardboard": 850, "Leather": 900, "Fabric": 280, "Plastic": 1350, "Others": 500,
  };
  const SHORE_DEFAULTS = {
    "Paper": 15, "Cardstock": 30, "Iron-On": 45, "Vinyl": 65,
    "Smart Materials": 65, "Infusible Ink": 20, "Printable Materials": 25,
    "Board/Cardboard": 65, "Leather": 55, "Fabric": 10, "Plastic": 70, "Others": 40,
  };
  const THICKNESS_DEFAULTS = {
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10, "Vinyl": 0.08,
    "Smart Materials": 0.10, "Infusible Ink": 0.10, "Printable Materials": 0.12,
    "Board/Cardboard": 1.0, "Leather": 1.6, "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
  };

  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  /* Normalize material name the same way the Python training script does.
     Only strips measurement-only parentheticals (digits + units).
     Descriptive ones like "(Mosaic)" or "(Green Liner)" are kept. */
  const MEAS_PAREN = /\s*\(\s*\d[\d\s./\-]*(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?(?:\s*\/\s*[\d\s./\-]+(?:gsm|lbs?|oz\.?|mm|cm|inch(?:es)?|gauge)?)?\s*\)\s*$/i;
  function normalizeName(name) {
    const cleaned = name.trim().replace(MEAS_PAREN, '').trim();
    return cleaned || name.trim();
  }

  /* Resolve embedding vector for a material name (or fall back to category average) */
  function resolveEmbedding(pp, materialName, category) {
    const baseName = normalizeName(materialName);
    if (pp.name_embeddings && pp.name_embeddings[baseName]) {
      return pp.name_embeddings[baseName];               // exact match
    }
    /* partial match: find a name that starts with the normalized input */
    if (pp.name_embeddings) {
      const lower = baseName.toLowerCase();
      for (const [k, v] of Object.entries(pp.name_embeddings)) {
        if (k.toLowerCase().startsWith(lower) || lower.startsWith(k.toLowerCase())) {
          return v;
        }
      }
    }
    /* category average fallback */
    if (pp.category_avg_embeddings && category && pp.category_avg_embeddings[category]) {
      return pp.category_avg_embeddings[category];
    }
    return new Array(pp.emb_dim || 16).fill(0);
  }

  /* Resolve physics properties from material lookup or category defaults */
  function resolveProps(pp, materialName, category, thicknessMm) {
    const baseName = normalizeName(materialName);
    const lookup   = pp.material_lookup && pp.material_lookup[baseName];
    const cat      = (lookup && lookup.category) || category || "Others";

    const gsm       = lookup ? lookup.gsm     : (GSM_DEFAULTS[cat] || 100);
    const density   = lookup ? lookup.density : (DENSITY_DEFAULTS[cat] || 500);
    const shore     = lookup ? lookup.shore   : (SHORE_DEFAULTS[cat] || 40);
    const texture   = lookup ? (lookup.texture || 0) : 0;
    const adhesive  = lookup ? (lookup.has_adhesive || 0) : 0;
    const bonded    = lookup ? (lookup.is_bonded || 0) : 0;
    const thickness = thicknessMm !== null && thicknessMm > 0
      ? thicknessMm
      : (lookup ? lookup.thickness_mm : (THICKNESS_DEFAULTS[cat] || 0.5));

    return { gsm, density, shore, texture, adhesive, bonded, thickness, cat };
  }

  /* Build feature vector (v2: 23-dim, v3: 26-dim).
     v2: [emb(16)] + [gsm_lognorm, thickness_lognorm, is_bonded, surface_texture,
                      has_adhesive, density_lognorm, shore_norm]
     v3: same as v2 + [family_joy, family_explore, family_maker] */
  function buildFeatures(pp, { materialName, category, thicknessMm, machine }) {
    const emb   = resolveEmbedding(pp, materialName, category);
    const props = resolveProps(pp, materialName, category, thicknessMm);

    const features = new Float32Array(pp.feature_dim);

    /* 0–15: name embedding */
    for (let i = 0; i < (pp.emb_dim || 16); i++) {
      features[i] = emb[i] || 0;
    }

    /* 16: GSM log1p MinMax */
    const gLog = Math.log1p(props.gsm);
    features[16] = clamp01((gLog - pp.gsm_log_min) / (pp.gsm_log_max - pp.gsm_log_min + 1e-9));

    /* 17: thickness log1p MinMax */
    const tLog = Math.log1p(props.thickness);
    features[17] = clamp01((tLog - pp.thickness_log_min) / (pp.thickness_log_max - pp.thickness_log_min + 1e-9));

    /* 18: is_bonded */
    features[18] = props.bonded;

    /* 19: surface_texture */
    features[19] = props.texture;

    /* 20: has_adhesive */
    features[20] = props.adhesive;

    /* 21: density log1p MinMax */
    const dLog = Math.log1p(props.density);
    features[21] = clamp01((dLog - pp.density_log_min) / (pp.density_log_max - pp.density_log_min + 1e-9));

    /* 22: Shore A / 100 */
    features[22] = clamp01(props.shore / 100);

    /* dims 23+: machine family one-hot (v3 only) */
    if (pp.n_families && machine) {
      const info   = pp.machines[machine];
      const fam    = info !== undefined ? info.family : (pp.n_families - 1);
      const offset = pp.emb_dim + pp.n_physics;
      for (let f = 0; f < pp.n_families; f++) features[offset + f] = fam === f ? 1 : 0;
    }

    return features;
  }

  /* ── v1 fallback buildFeatures (category-based, 19-dim) ─────────────────── */
  const HARDNESS_DEFAULTS_V1 = {
    "Paper": 2, "Cardstock": 4, "Iron-On": 3, "Vinyl": 3,
    "Smart Materials": 3, "Printable Materials": 3, "Infusible Ink": 3,
    "Board/Cardboard": 7, "Leather": 6, "Fabric": 3, "Plastic": 4, "Others": 5,
  };
  function buildFeaturesV1(pp, { category, thicknessMm }) {
    const features = new Float32Array(pp.feature_dim);
    const ci = (pp.categories || []).indexOf(category);
    if (ci >= 0) features[ci] = 1.0;
    const tLog = Math.log1p(thicknessMm);
    features[11] = clamp01((tLog - pp.thickness_log_min) / (pp.thickness_log_max - pp.thickness_log_min + 1e-9));
    const h = HARDNESS_DEFAULTS_V1[category] || 5;
    features[12] = clamp01((h - 1) / 9);
    features[13] = 0.0;
    const gsm  = GSM_DEFAULTS[category] || 100;
    const gLog = Math.log1p(gsm);
    features[14] = clamp01((gLog - pp.gsm_log_min) / (pp.gsm_log_max - pp.gsm_log_min + 1e-9));
    features[15] = 0.0; features[16] = 0.0;
    const density = DENSITY_DEFAULTS[category] || 500;
    const dLog    = Math.log1p(density);
    features[17] = clamp01((dLog - pp.density_log_min) / (pp.density_log_max - pp.density_log_min + 1e-9));
    features[18] = clamp01((SHORE_DEFAULTS[category] || 40) / 100);
    return features;
  }

  const _sessions = {};
  let _pp = null;

  async function loadPreprocessor() {
    if (_pp) return _pp;
    /* Try v2 first, fall back to v1 */
    for (const path of ["assets/model/preprocessor.json", "assets/model/preprocessor_v2.json"]) {
      try {
        const r = await fetch(path);
        if (r.ok) { _pp = await r.json(); return _pp; }
      } catch (_) {}
    }
    throw new Error("preprocessor.json not found");
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

  async function predict({ machine, materialName, category, thicknessMm }) {
    const pp   = await loadPreprocessor();
    const info = pp.machines[machine];
    if (!info) throw new Error("Unknown machine: " + machine);

    const sess = await getSession(info.slug);

    let features;
    if (pp.version === "v3" || pp.version === "v2") {
      features = buildFeatures(pp, { materialName: materialName || "", category, thicknessMm, machine });
    } else {
      features = buildFeaturesV1(pp, { category, thicknessMm });
    }

    const tensor = new ort.Tensor("float32", features, [1, pp.feature_dim]);
    const out    = await sess.run({ features: tensor });

    /* Pressure */
    const pNorm    = out["pressure_norm"].data[0];
    const pressure = Math.round(Math.exp(pNorm * pp.pressure_log_std + pp.pressure_log_mean));

    /* Blade — constrain to machine-compatible blades */
    const bladeLogits = Array.from(out["blade_logits"].data);
    const compatible  = (pp.notes && pp.notes.compatible_blades &&
                         pp.notes.compatible_blades[machine]) || null;
    let bladeIdx = 0, best = -Infinity;
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

    /* Resolved material info for display */
    const props = resolveProps(pp, materialName || "", category, thicknessMm);
    const resolvedCategory = props.cat;

    return { pressure, bladeEn, bladeJp, multicut, resolvedCategory };
  }

  /* Public: resolve auto-fill properties from material name */
  function getMaterialProps(materialName) {
    if (!_pp) return null;
    const baseName = normalizeName(materialName);
    const lookup   = _pp.material_lookup && _pp.material_lookup[baseName];
    return lookup || null;
  }

  function getKnownNames() {
    if (!_pp || !_pp.name_vocab) return [];
    return Object.keys(_pp.name_vocab);
  }

  window.CricutPredict = { predict, loadPreprocessor, getMaterialProps, getKnownNames };
})();
