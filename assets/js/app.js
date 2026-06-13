/* app.js — browse, filter, predict */
(function () {
  const { t, tCat, tBlade, currentLang, setLang } = window.i18n;
  const { predict, loadPreprocessor, getMaterialProps, getKnownNames } = window.CricutPredict;

  const MACHINES = ["Cricut Joy", "Cricut Joy 2", "Cricut Joy Xtra", "Explore 3", "Maker 3"];
  const MACHINE_SHORT = {
    "Cricut Joy": "Joy", "Cricut Joy 2": "Joy 2", "Cricut Joy Xtra": "Joy Xtra",
    "Explore 3": "Explore 3", "Maker 3": "Maker 3",
  };
  const CATEGORIES = [
    "Board/Cardboard","Cardstock","Fabric","Infusible Ink","Iron-On",
    "Leather","Others","Paper","Plastic","Printable Materials","Smart Materials","Vinyl",
  ];
  const BLADES_EN = [
    "Fine-Point Blade","Deep-Point Blade","Rotary Blade","Bonded Fabric Blade","Knife Blade",
  ];
  const CAT_BADGE = {
    "Vinyl":"badge-vinyl","Iron-On":"badge-ironon","Cardstock":"badge-cardstock",
    "Paper":"badge-paper","Fabric":"badge-fabric","Leather":"badge-leather",
    "Board/Cardboard":"badge-board","Others":"badge-others",
    "Infusible Ink":"badge-infusible","Smart Materials":"badge-smart",
    "Printable Materials":"badge-printable","Plastic":"badge-plastic",
  };
  const CAT_COLOR = {
    "Vinyl":               "#1eb487",
    "Iron-On":             "#9333ea",
    "Cardstock":           "#4faa3e",
    "Paper":               "#8fbf3f",
    "Fabric":              "#d05464",
    "Leather":             "#b83232",
    "Board/Cardboard":     "#c75001",
    "Plastic":             "#3d6bb5",
    "Infusible Ink":       "#1a8163",
    "Smart Materials":     "#0c8487",
    "Printable Materials": "#4a6fa5",
    "Others":              "#6b7280",
  };
  const THICKNESS_DEFAULTS = {
    "Paper":0.08,"Cardstock":0.22,"Iron-On":0.10,"Vinyl":0.08,
    "Smart Materials":0.10,"Printable Materials":0.12,"Infusible Ink":0.10,
    "Board/Cardboard":1.0,"Leather":1.6,"Fabric":0.50,"Plastic":0.10,"Others":2.0,
  };

  const BLADE_EN_TO_JP = {
    "Fine-Point Blade":    "ファインポイントブレード",
    "Deep-Point Blade":    "ディープポイントブレード",
    "Rotary Blade":        "ロータリーブレード",
    "Bonded Fabric Blade": "ボンデッドファブリックブレード",
    "Knife Blade":         "ナイフの刃",
  };
  const MC_LABELS = ["1×", "2×", "3×", "4–5×", "6–8×", "10+×"];

  /* ── State ──────────────────────────────────────────────────────────────────── */
  let materials      = [];
  let activeMachine  = MACHINES[0];
  let filterCat      = "";
  let filterBlade    = "";
  let sortMode       = "none"; // none | asc | desc
  let searchQuery    = "";
  let searchTimer    = null;
  let viewMode       = "list"; // "list" | "grid"
  let lastPrediction = null;   // context of the most recent AI prediction

  const $ = id => document.getElementById(id);

  /* ── Custom materials (user corrections, persisted in localStorage) ─────────── */
  const CUSTOM_KEY = "cricut_custom_materials";
  function loadCustomMaterials() {
    try { return JSON.parse(localStorage.getItem(CUSTOM_KEY)) || []; }
    catch { return []; }
  }
  function saveCustomMaterials(arr) {
    try { localStorage.setItem(CUSTOM_KEY, JSON.stringify(arr)); } catch {}
  }

  /* ── Init ───────────────────────────────────────────────────────────────────── */
  async function init() {
    applyI18n();
    buildMachineBar();
    populateFilterSelects();
    bindToolbar();
    bindLangToggle();
    bindViewToggle();
    bindHeroButtons();
    bindPredictForm();

    try {
      const r = await fetch("assets/data/materials.json");
      materials = await r.json();
    } catch {
      $("matList").innerHTML = emptyHtml(t("err_model_load"), "");
      return;
    }
    materials = materials.concat(loadCustomMaterials());
    buildSidebar();
    render();

    // Warm preprocessor + first machine model; populate material name datalist
    loadPreprocessor().then(pp => {
      populateMaterialNameList();
      const info = pp.machines[activeMachine];
      if (info && window.ort) {
        ort.InferenceSession.create(
          "assets/model/material_predictor_" + info.slug + ".onnx",
          { executionProviders: ["wasm"] }
        ).catch(() => {});
      }
    }).catch(() => {});
  }

  /* ── i18n ───────────────────────────────────────────────────────────────────── */
  function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    const lang = currentLang();
    $("langEN").classList.toggle("active", lang === "en");
    $("langJP").classList.toggle("active", lang === "ja");
  }

  /* ── Machine bar ────────────────────────────────────────────────────────────── */
  function buildMachineBar() {
    const bar = $("machineBar");
    bar.innerHTML = MACHINES.map(m =>
      '<button class="machine-tab' + (m === activeMachine ? " active" : "") +
      '" data-machine="' + esc(m) + '" role="tab" aria-selected="' +
      (m === activeMachine) + '">' + esc(MACHINE_SHORT[m]) + '</button>'
    ).join("");
    bar.querySelectorAll(".machine-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        activeMachine = btn.dataset.machine;
        filterCat     = "";
        filterBlade   = "";
        sortMode      = "none";
        searchQuery   = "";
        $("searchInput").value = "";
        $("filterCategory").value = "";
        $("filterBlade").value    = "";
        updateSortBtn();
        buildMachineBar();
        buildSidebar();
        render();
      });
    });
  }

  /* ── Filter selects ─────────────────────────────────────────────────────────── */
  function populateFilterSelects() {
    const cSel = $("filterCategory");
    cSel.innerHTML = '<option value="">' + esc(t("filter_all_categories")) + '</option>' +
      CATEGORIES.map(c => '<option value="' + esc(c) + '">' + esc(tCat(c)) + '</option>').join("");

    const bSel = $("filterBlade");
    bSel.innerHTML = '<option value="">' + esc(t("filter_all_blades")) + '</option>' +
      BLADES_EN.map(b => '<option value="' + esc(b) + '">' + esc(tBlade(b)) + '</option>').join("");
  }

  /* ── Toolbar ────────────────────────────────────────────────────────────────── */
  function bindToolbar() {
    $("searchInput").addEventListener("input", e => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        searchQuery = e.target.value.trim().toLowerCase();
        buildSidebar();
        render();
      }, 150);
    });
    $("filterCategory").addEventListener("change", e => {
      filterCat = e.target.value;
      buildSidebar();
      render();
    });
    $("filterBlade").addEventListener("change", e => {
      filterBlade = e.target.value;
      buildSidebar();
      render();
    });
    $("sortToggle").addEventListener("click", () => {
      sortMode = sortMode === "asc" ? "desc" : sortMode === "desc" ? "none" : "asc";
      updateSortBtn();
      render();
    });
  }

  function updateSortBtn() {
    const btn = $("sortToggle");
    btn.textContent = t(
      sortMode === "asc" ? "sort_asc" : sortMode === "desc" ? "sort_desc" : "sort_default"
    );
    btn.classList.toggle("active-sort", sortMode !== "none");
  }

  /* ── Mode nav ───────────────────────────────────────────────────────────────── */
  function switchMode(mode) {
    const b = mode === "browse";
    $("browsePanel").classList.toggle("hidden", !b);
    $("predictPanel").classList.toggle("hidden", b);
  }

  /* ── Language toggle ────────────────────────────────────────────────────────── */
  function bindLangToggle() {
    ["langEN", "langJP"].forEach(id => {
      $(id).addEventListener("click", () => {
        setLang(id === "langJP" ? "ja" : "en");
        applyI18n();
        populateFilterSelects();
        updateSortBtn();
        populatePredictForm();
        buildSidebar();
        render();
      });
    });
  }

  /* ── Render browse list ─────────────────────────────────────────────────────── */
  function getFiltered() {
    let list = materials.filter(m => m.machine === activeMachine);
    if (filterCat)    list = list.filter(m => m.category === filterCat);
    if (filterBlade)  list = list.filter(m => m.blade_en === filterBlade);
    if (searchQuery)  list = list.filter(m =>
      m.name_en.toLowerCase().includes(searchQuery) ||
      m.name_jp.toLowerCase().includes(searchQuery)
    );
    if (sortMode === "asc")  list = [...list].sort((a, b) => a.pressure - b.pressure);
    if (sortMode === "desc") list = [...list].sort((a, b) => b.pressure - a.pressure);
    return list;
  }

  function render() {
    const list = getFiltered();
    const lang = currentLang();
    $("resultCount").textContent = t("result_count", list.length);

    if (!list.length) {
      $("matList").innerHTML = emptyHtml(t("empty_title"), t("empty_sub"));
      return;
    }

    const frag = document.createDocumentFragment();

    const groups = {};
    CATEGORIES.forEach(c => { groups[c] = []; });
    list.forEach(m => { if (groups[m.category]) groups[m.category].push(m); });

    CATEGORIES.forEach(cat => {
      const items = groups[cat];
      if (!items.length) return;

      if (sortMode === "asc")  items.sort((a, b) => a.pressure - b.pressure);
      if (sortMode === "desc") items.sort((a, b) => b.pressure - a.pressure);

      const color = CAT_COLOR[cat] || "#6b7280";

      const section = document.createElement("div");
      section.className = "cat-section";
      section.style.setProperty("--cat-color", color);

      const header = document.createElement("div");
      header.className = "cat-header";
      header.textContent = tCat(cat);
      section.appendChild(header);

      const catCards = document.createElement("div");
      catCards.className = "cat-cards" + (viewMode === "grid" ? " cat-cards-grid" : "");

      items.forEach(m => {
        const primary   = lang === "ja" ? m.name_jp : m.name_en;
        const secondary = lang === "ja" ? m.name_en : m.name_jp;
        const bladeDisp = lang === "ja" ? m.blade_jp : m.blade_en;
        const catDisp   = tCat(m.category);

        const card = document.createElement("div");
        card.className = "mat-card" + (viewMode === "grid" ? " mat-card-grid" : "");
        card.setAttribute("role", "listitem");
        card.innerHTML =
          '<div class="mat-card-head">' +
            '<div class="mat-name-wrap">' +
              '<span class="mat-name">' + esc(primary) + '</span>' +
              (secondary && secondary !== primary
                ? '<span class="mat-name-alt">' + esc(secondary) + '</span>'
                : '') +
            '</div>' +
            (m.custom ? '<span class="mat-badge badge-custom">' + esc(t("custom_badge")) + '</span>' : '') +
            '<span class="mat-badge">' + esc(catDisp) + '</span>' +
          '</div>' +
          '<div class="mat-chips">' +
            '<div class="mat-chip">' +
              '<span class="chip-label">' + esc(t("pressure_label")) + '</span>' +
              '<span class="chip-val">' + m.pressure + '</span>' +
            '</div>' +
            '<div class="mat-chip">' +
              '<span class="chip-label">' + esc(t("multicut_label")) + '</span>' +
              '<span class="chip-val">' + esc(m.multicut) + '</span>' +
            '</div>' +
            '<div class="mat-chip mat-chip-blade">' +
              '<span class="chip-label">' + esc(t("blade_label")) + '</span>' +
              '<span class="chip-val">' + esc(bladeDisp) + '</span>' +
            '</div>' +
            (m.thickness_mm
              ? '<div class="mat-chip">' +
                  '<span class="chip-label">' + esc(t("thickness_label")) + '</span>' +
                  '<span class="chip-val">' + m.thickness_mm + ' mm</span>' +
                '</div>'
              : '') +
          '</div>';
        catCards.appendChild(card);
      });

      section.appendChild(catCards);
      frag.appendChild(section);
    });

    $("matList").innerHTML = "";
    $("matList").appendChild(frag);
  }

  /* ── Sidebar ────────────────────────────────────────────────────────────────── */
  function buildSidebar() {
    const machineMats = materials.filter(m => m.machine === activeMachine);
    const counts = {};
    CATEGORIES.forEach(c => { counts[c] = 0; });
    machineMats.forEach(m => { if (counts[m.category] !== undefined) counts[m.category]++; });

    let html =
      '<button class="sidebar-item' + (!filterCat ? " active" : "") + '" data-cat="">' +
      esc(t("sidebar_all")) + '<span class="sidebar-count">' + machineMats.length + '</span></button>';

    CATEGORIES.forEach(cat => {
      const count = counts[cat];
      if (!count) return;
      const color = CAT_COLOR[cat] || "#6b7280";
      html +=
        '<button class="sidebar-item' + (filterCat === cat ? " active" : "") +
        '" data-cat="' + esc(cat) + '" style="--cat-color:' + color + '">' +
        '<span class="sidebar-dot"></span>' +
        esc(tCat(cat)) +
        '<span class="sidebar-count">' + count + '</span></button>';
    });

    const list = $("sidebarList");
    list.innerHTML = html;
    list.querySelectorAll(".sidebar-item").forEach(btn => {
      btn.addEventListener("click", () => {
        filterCat = btn.dataset.cat;
        $("filterCategory").value = filterCat;
        buildSidebar();
        render();
      });
    });

    // Export button — only shown when the user has saved custom materials
    const sidebar = $("catSidebar");
    let exportEl = sidebar.querySelector(".sidebar-export");
    const customs = loadCustomMaterials();
    if (customs.length > 0) {
      if (!exportEl) {
        exportEl = document.createElement("div");
        exportEl.className = "sidebar-export";
        sidebar.appendChild(exportEl);
      }
      exportEl.innerHTML =
        '<div class="sidebar-export-label">' + esc(t("sidebar_custom_count", customs.length)) + '</div>' +
        '<button class="btn-sidebar-export" id="btnExportCustom">' + esc(t("sidebar_export_btn")) + '</button>';
      sidebar.querySelector("#btnExportCustom").addEventListener("click", exportCustomMaterials);
    } else if (exportEl) {
      exportEl.remove();
    }
  }

  function exportCustomMaterials() {
    const customs = loadCustomMaterials();
    if (!customs.length) return;
    const blob = new Blob([JSON.stringify(customs, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "cricut_custom_materials.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ── View toggle ────────────────────────────────────────────────────────────── */
  function bindViewToggle() {
    $("viewList").addEventListener("click", () => {
      if (viewMode === "list") return;
      viewMode = "list";
      $("viewList").classList.add("active");
      $("viewGrid").classList.remove("active");
      render();
    });
    $("viewGrid").addEventListener("click", () => {
      if (viewMode === "grid") return;
      viewMode = "grid";
      $("viewGrid").classList.add("active");
      $("viewList").classList.remove("active");
      render();
    });
  }

  /* ── Hero buttons ───────────────────────────────────────────────────────────── */
  function bindHeroButtons() {
    $("heroBtnBrowse").addEventListener("click", () => {
      switchMode("browse");
      $("browsePanel").scrollIntoView({ behavior: "smooth" });
    });
    $("heroBtnPredict").addEventListener("click", () => {
      switchMode("predict");
      $("predictPanel").scrollIntoView({ behavior: "smooth" });
    });
  }

  /* ── Predict form ───────────────────────────────────────────────────────────── */
  function bindPredictForm() {
    populatePredictForm();
    $("pCategory").addEventListener("change", () => {
      const cat = $("pCategory").value;
      $("pThickness").value = THICKNESS_DEFAULTS[cat] || 0.5;
      populateMaterialNameList();
    });
    $("pMaterialName").addEventListener("input", onMaterialNameInput);
    $("pMaterialName").addEventListener("change", onMaterialNameInput);
    $("btnPredict").addEventListener("click", runPredict);
  }

  /* Populate <datalist> with known names of the selected type only */
  function populateMaterialNameList() {
    const dl = $("materialNameList");
    if (!dl) return;
    const cat   = $("pCategory").value;
    const names = getKnownNames ? getKnownNames(cat) : [];
    dl.innerHTML = names.map(n => '<option value="' + esc(n) + '">').join("");
  }

  /* Auto-fill thickness + category when a known material is selected */
  function onMaterialNameInput() {
    const name  = $("pMaterialName").value.trim();
    if (!name || !getMaterialProps) return;
    const props = getMaterialProps(name);
    if (!props) return;
    /* auto-fill category */
    if (props.category && props.category !== $("pCategory").value) {
      const cSel = $("pCategory");
      for (let i = 0; i < cSel.options.length; i++) {
        if (cSel.options[i].value === props.category) { cSel.selectedIndex = i; break; }
      }
      populateMaterialNameList();
    }
    /* auto-fill thickness */
    if (props.thickness_mm && props.thickness_mm > 0) {
      $("pThickness").value = props.thickness_mm.toFixed(2);
    }
  }

  function populatePredictForm() {
    const mSel = $("pMachine");
    const mVal = mSel.value || activeMachine;
    mSel.innerHTML = MACHINES.map(m =>
      '<option value="' + esc(m) + '"' + (m === mVal ? " selected" : "") + '>' + esc(m) + '</option>'
    ).join("");

    const cSel = $("pCategory");
    const cVal = cSel.value || "Vinyl";
    cSel.innerHTML = CATEGORIES.map(c =>
      '<option value="' + esc(c) + '"' + (c === cVal ? " selected" : "") + '>' +
      esc(tCat(c)) + '</option>'
    ).join("");

    if (!$("pThickness").value) {
      $("pThickness").value = THICKNESS_DEFAULTS[cVal] || 0.5;
    }
  }

  async function runPredict() {
    const errEl = $("errThickness");
    errEl.classList.add("hidden");

    const raw = $("pThickness").value;
    const mm  = parseFloat(raw);
    if (!raw || isNaN(mm)) {
      errEl.textContent = t("err_thickness_required");
      errEl.classList.remove("hidden");
      return;
    }
    if (mm < 0.01 || mm > 60) {
      errEl.textContent = t("err_thickness_range");
      errEl.classList.remove("hidden");
      return;
    }

    const machine      = $("pMachine").value;
    const category     = $("pCategory").value;
    const materialName = $("pMaterialName") ? $("pMaterialName").value.trim() : "";
    const btn          = $("btnPredict");

    btn.disabled    = true;
    btn.textContent = t("btn_predicting");
    $("predictResult").classList.add("hidden");

    try {
      const result = await predict({ machine, materialName, category, thicknessMm: mm });
      lastPrediction = {
        machine:     machine,
        name:        materialName,
        category:    result.resolvedCategory || category,
        thicknessMm: mm,
        pressure:    result.pressure,
        bladeEn:     result.bladeEn,
        multicut:    result.multicut,
      };
      showResult(result, machine, materialName || category);
    } catch {
      $("predictResult").innerHTML =
        '<div class="result-box result-error">' +
        '<div class="result-header"><span class="result-header-title">Error</span></div>' +
        '<div class="result-body">' + t("err_model_load") + '</div></div>';
      $("predictResult").classList.remove("hidden");
    } finally {
      btn.disabled    = false;
      btn.textContent = t("btn_predict");
    }
  }

  function showResult(r, machine, materialOrCat) {
    const lang      = currentLang();
    const bladeDisp = lang === "ja" ? r.bladeJp : r.bladeEn;
    /* Show material name if typed, else fall back to resolved category */
    const catLabel  = r.resolvedCategory ? tCat(r.resolvedCategory) : tCat(materialOrCat);
    const subLabel  = materialOrCat && materialOrCat !== r.resolvedCategory
      ? esc(materialOrCat) + ' · ' + esc(catLabel)
      : esc(catLabel);
    $("predictResult").innerHTML =
      '<div class="result-box">' +
        '<div class="result-header">' +
          '<span class="result-header-title">' + t("result_title") + '</span>' +
          '<span class="result-header-sub">' + esc(machine) + ' · ' + subLabel + '</span>' +
        '</div>' +
        '<div class="result-body">' +
          '<div class="result-specs">' +
            resultSpec(t("pressure_label"), r.pressure, false) +
            resultSpec(t("blade_label"),    bladeDisp,  true) +
            resultSpec(t("multicut_label"), r.multicut, false) +
          '</div>' +
          '<p class="result-disclaimer">' + t("disclaimer") + '</p>' +
          '<div class="result-footer">' +
            '<button class="btn-correct" id="btnAddMaterial">' + esc(t("btn_add_material")) + '</button>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div id="correctWrap"></div>';
    $("predictResult").classList.remove("hidden");
    $("btnAddMaterial").addEventListener("click", toggleCorrectForm);
  }

  /* ── "Add a new material" correction form ──────────────────────────────────── */
  function toggleCorrectForm() {
    const wrap = $("correctWrap");
    if (wrap.innerHTML) { wrap.innerHTML = ""; return; }
    if (!lastPrediction) return;
    const p = lastPrediction;

    wrap.innerHTML =
      '<div class="correct-form">' +
        '<div class="correct-title">' + esc(t("correct_title")) + '</div>' +
        '<div class="correct-grid">' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfName">' + esc(t("form_material_name")) + '</label>' +
            '<input class="form-input" type="text" id="cfName" value="' + esc(p.name) + '">' +
          '</div>' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfCategory">' + esc(t("form_category")) + '</label>' +
            '<select class="form-select" id="cfCategory">' +
              CATEGORIES.map(c =>
                '<option value="' + esc(c) + '"' + (c === p.category ? ' selected' : '') + '>' +
                esc(tCat(c)) + '</option>').join("") +
            '</select>' +
          '</div>' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfThickness">' + esc(t("form_thickness")) + '</label>' +
            '<div class="input-group">' +
              '<input class="form-input" type="number" id="cfThickness" min="0.01" max="60" step="0.01" value="' + p.thicknessMm + '">' +
              '<span class="input-unit">mm</span>' +
            '</div>' +
          '</div>' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfPressure">' + esc(t("pressure_label")) + '</label>' +
            '<input class="form-input" type="number" id="cfPressure" min="1" max="1500" step="1" value="' + p.pressure + '">' +
          '</div>' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfBlade">' + esc(t("blade_label")) + '</label>' +
            '<select class="form-select" id="cfBlade">' +
              BLADES_EN.map(b =>
                '<option value="' + esc(b) + '"' + (b === p.bladeEn ? ' selected' : '') + '>' +
                esc(tBlade(b)) + '</option>').join("") +
            '</select>' +
          '</div>' +
          '<div class="form-row">' +
            '<label class="form-label" for="cfMC">' + esc(t("multicut_label")) + '</label>' +
            '<select class="form-select" id="cfMC">' +
              MC_LABELS.map(m =>
                '<option value="' + esc(m) + '"' + (m === p.multicut ? ' selected' : '') + '>' +
                esc(m) + '</option>').join("") +
            '</select>' +
          '</div>' +
        '</div>' +
        '<span class="field-error hidden" id="cfError"></span>' +
        '<div class="correct-actions">' +
          '<button class="btn-secondary" id="cfCancel">' + esc(t("btn_cancel")) + '</button>' +
          '<button class="btn-cta btn-cta-compact" id="cfSave">' + esc(t("btn_save_material")) + '</button>' +
        '</div>' +
      '</div>';

    $("cfCancel").addEventListener("click", () => { wrap.innerHTML = ""; });
    $("cfSave").addEventListener("click", saveCorrectedMaterial);
  }

  function saveCorrectedMaterial() {
    const errEl = $("cfError");
    errEl.classList.add("hidden");

    const name      = $("cfName").value.trim();
    const category  = $("cfCategory").value;
    const thickness = parseFloat($("cfThickness").value);
    const pressure  = parseInt($("cfPressure").value, 10);
    const bladeEn   = $("cfBlade").value;
    const multicut  = $("cfMC").value;

    const fail = msg => { errEl.textContent = msg; errEl.classList.remove("hidden"); };
    if (!name) return fail(t("err_name_required"));
    if (isNaN(thickness) || thickness < 0.01 || thickness > 60) return fail(t("err_thickness_range"));
    if (isNaN(pressure) || pressure < 1) return fail(t("err_thickness_required"));

    /* Only materials NOT already in the list may be added:
       same name + type + thickness (machine-independent) is a duplicate. */
    const dup = materials.some(m =>
      m.category === category &&
      Math.abs((m.thickness_mm || 0) - thickness) < 0.005 &&
      (m.name_en.toLowerCase() === name.toLowerCase() ||
       (m.name_jp && m.name_jp.toLowerCase() === name.toLowerCase()))
    );
    if (dup) return fail(t("err_duplicate_material"));

    const entry = {
      machine:      lastPrediction.machine,
      category:     category,
      name_en:      name,
      name_jp:      name,
      pressure:     pressure,
      multicut:     multicut,
      blade_en:     bladeEn,
      blade_jp:     BLADE_EN_TO_JP[bladeEn] || bladeEn,
      thickness_mm: Math.round(thickness * 100) / 100,
      custom:       true,
    };

    const customs = loadCustomMaterials();
    customs.push(entry);
    saveCustomMaterials(customs);
    materials.push(entry);

    buildSidebar();
    render();

    $("correctWrap").innerHTML =
      '<div class="correct-saved" role="status">✓ ' + esc(t("msg_material_saved")) + '</div>';
  }

  function resultSpec(label, value, small) {
    return '<div class="result-spec">' +
      '<span class="result-spec-label">' + esc(label) + '</span>' +
      '<span class="result-spec-val' + (small ? ' small' : '') + '">' + esc(String(value)) + '</span>' +
      '</div>';
  }

  /* ── Helpers ────────────────────────────────────────────────────────────────── */
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function emptyHtml(title, sub) {
    return '<div class="empty-state" role="status">' +
      '<div class="empty-icon">✦</div>' +
      '<p class="empty-title">' + esc(title) + '</p>' +
      (sub ? '<p class="empty-sub">' + esc(sub) + '</p>' : '') +
      '</div>';
  }

  document.addEventListener("DOMContentLoaded", init);
})();