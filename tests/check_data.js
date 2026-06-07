#!/usr/bin/env node
/**
 * check_data.js — Data integrity tests (D-01 … D-08, updated for 12 categories)
 * Run: node tests/check_data.js
 */

const fs   = require("fs");
const path = require("path");

const ROOT         = path.resolve(__dirname, "..");
const MAT_JSON     = path.join(ROOT, "assets", "data", "materials.json");
const PREP_JSON    = path.join(ROOT, "assets", "model", "preprocessor.json");

const KNOWN_MACHINES  = ["Cricut Joy","Cricut Joy 2","Cricut Joy Xtra","Explore 3","Maker 3"];
const KNOWN_CATEGORIES = [
  "Board/Cardboard","Cardstock","Fabric","Infusible Ink","Iron-On",
  "Leather","Others","Paper","Plastic","Printable Materials","Smart Materials","Vinyl",
];
const KNOWN_BLADES = [
  "Fine-Point Blade","Deep-Point Blade","Rotary Blade","Bonded Fabric Blade","Knife Blade",
];

let passed = 0, failed = 0;

function pass(id, desc) {
  console.log(`  ✓  [${id}] ${desc}`);
  passed++;
}
function fail(id, desc, detail) {
  console.error(`  ✗  [${id}] ${desc}\n       → ${detail}`);
  failed++;
}

// ── Load files ────────────────────────────────────────────────────────────────
console.log("\n=== Data Integrity Tests ===\n");

let mats, prep;
try {
  mats = JSON.parse(fs.readFileSync(MAT_JSON, "utf8"));
  pass("D-01", "materials.json loads without parse error");
} catch(e) {
  fail("D-01", "materials.json loads without parse error", e.message);
  process.exit(1);
}

try {
  prep = JSON.parse(fs.readFileSync(PREP_JSON, "utf8"));
  pass("M-01a", "preprocessor.json loads without parse error");
} catch(e) {
  fail("M-01a", "preprocessor.json loads without parse error", e.message);
}

// D-02
if (mats.length >= 1039) {
  pass("D-02", `Row count ≥ 1039 (got ${mats.length})`);
} else {
  fail("D-02", "Row count ≥ 1039", `got ${mats.length}`);
}

// D-03 — required fields
const REQUIRED = ["machine","category","name_en","name_jp","pressure","multicut","blade_en","blade_jp"];
const missingFields = [];
mats.forEach((row, i) => {
  REQUIRED.forEach(f => {
    if (row[f] === undefined) missingFields.push(`row ${i}: missing '${f}'`);
  });
});
if (missingFields.length === 0) {
  pass("D-03", "All required fields present on every row");
} else {
  fail("D-03", "All required fields present", missingFields.slice(0,5).join("; "));
}

// D-04 — pressure > 0
const badPressure = mats.filter(r => !(r.pressure > 0 && isFinite(r.pressure)));
if (badPressure.length === 0) {
  pass("D-04", "All pressures are positive finite numbers");
} else {
  fail("D-04", "All pressures positive finite", badPressure.slice(0,3).map(r=>`${r.name_en}:${r.pressure}`).join("; "));
}

// D-05 — blade values
const badBlades = mats.filter(r => !KNOWN_BLADES.includes(r.blade_en));
if (badBlades.length === 0) {
  pass("D-05", "All blade_en values are in the known set");
} else {
  fail("D-05", "All blade_en in known set", [...new Set(badBlades.map(r=>r.blade_en))].join(", "));
}

// D-06 — machine values
const badMachines = mats.filter(r => !KNOWN_MACHINES.includes(r.machine));
if (badMachines.length === 0) {
  pass("D-06", "All machine values are in the known set");
} else {
  fail("D-06", "All machine in known set", [...new Set(badMachines.map(r=>r.machine))].join(", "));
}

// D-07 — all 5 machines represented
const foundMachines = new Set(mats.map(r => r.machine));
const missingMachines = KNOWN_MACHINES.filter(m => !foundMachines.has(m));
if (missingMachines.length === 0) {
  pass("D-07", "All 5 machines represented");
} else {
  fail("D-07", "All 5 machines represented", missingMachines.join(", "));
}

// D-08 — all 12 categories represented (updated from 11)
const foundCats = new Set(mats.map(r => r.category));
const missingCats = KNOWN_CATEGORIES.filter(c => !foundCats.has(c));
if (missingCats.length === 0) {
  pass("D-08", "All 12 categories represented (incl. Plastic)");
} else {
  fail("D-08", "All 12 categories represented", missingCats.join(", "));
}

// ── Preprocessor checks ────────────────────────────────────────────────────────
if (prep) {
  console.log();

  const hasVersion   = "version"           in prep;
  const hasCats      = Array.isArray(prep.categories) && prep.categories.length === 12;
  const hasBlades    = Array.isArray(prep.blade_types_en);
  const hasLogMean   = typeof prep.pressure_log_mean === "number";
  const hasLogStd    = typeof prep.pressure_log_std  === "number";
  const hasMachines  = typeof prep.machines === "object";
  if (hasVersion && hasCats && hasBlades && hasLogMean && hasLogStd && hasMachines) {
    pass("M-01", "preprocessor.json has all required fields (12 categories)");
  } else {
    const missing = [];
    if (!hasVersion) missing.push("version");
    if (!hasCats)    missing.push(`categories (got ${prep.categories?.length})`);
    if (!hasBlades)  missing.push("blade_types_en");
    if (!hasLogMean) missing.push("pressure_log_mean");
    if (!hasLogStd)  missing.push("pressure_log_std");
    if (!hasMachines)missing.push("machines");
    fail("M-01", "preprocessor.json has all required fields", missing.join(", "));
  }

  // feature_dim check
  if (prep.feature_dim === 20) {
    pass("M-03a", `feature_dim = 20 (12 categories + 8 numeric features)`);
  } else {
    fail("M-03a", "feature_dim = 20", `got ${prep.feature_dim}`);
  }

  // All 5 machine slugs map to ONNX files
  const expectedSlugs = ["cricut_joy","cricut_joy2","cricut_joy_xtra","explore3","maker3"];
  const onnxDir = path.join(ROOT, "assets", "model");
  const missingOnnx = [];
  expectedSlugs.forEach(slug => {
    const f = path.join(onnxDir, `material_predictor_${slug}.onnx`);
    if (!fs.existsSync(f)) missingOnnx.push(slug);
  });
  if (missingOnnx.length === 0) {
    pass("M-02a", "All 5 ONNX files exist on disk");
  } else {
    fail("M-02a", "All 5 ONNX files exist on disk", missingOnnx.join(", "));
  }
}

// ── Per-machine counts ─────────────────────────────────────────────────────────
console.log("\n  Machine counts:");
KNOWN_MACHINES.forEach(m => {
  const n = mats.filter(r => r.machine === m).length;
  console.log(`    ${m.padEnd(18)} ${n}`);
});
console.log("\n  Category counts:");
KNOWN_CATEGORIES.forEach(c => {
  const n = mats.filter(r => r.category === c).length;
  console.log(`    ${c.padEnd(25)} ${n}`);
});

// ── Summary ────────────────────────────────────────────────────────────────────
console.log(`\n${"─".repeat(50)}`);
console.log(`  ${passed + failed} tests  |  ${passed} passed  |  ${failed} failed`);
console.log(`${"─".repeat(50)}\n`);
process.exit(failed > 0 ? 1 : 0);
