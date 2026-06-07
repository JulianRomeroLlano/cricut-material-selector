/* app.js — browse, filter, predict */
(function () {
  const { t, tCat, tBlade, currentLang, setLang } = window.i18n;
  const { predict, loadPreprocessor }              = window.CricutPredict;

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

  /* ── State ──────────────────────────────────────────────────────────────────── */
  let materials     = [];
  let activeMachine = MACHINES[0];
  let filterCat     = "";
  let filterBlade   = "";
  let sortMode      = "none"; // none | asc | desc
  let searchQuery   = "";
  let searchTimer   = null;
  let viewMode      = "list"; // "list" | "grid"

  const $ = id => document.getElementById(id);

  /* ── Init ───────────────────────────────────────────────────────────────────── */
  async function init() {
    applyI18n();
    buildMachineBar();
    populateFilterSelects();
    bindToolbar();
    bindModeNav();
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
    buildSidebar();
    render();

    // Warm the first machine's ONNX model in the background
    loadPreprocessor().then(pp => {
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
  function bindModeNav() {
    $("tabBrowse").addEventListener("click",  () => switchMode("browse"));
    $("tabPredict").addEventListener("click", () => switchMode("predict"));
  }
  function switchMode(mode) {
    const b = mode === "browse";
    $("browsePanel").classList.toggle("hidden", !b);
    $("predictPanel").classList.toggle("hidden", b);
    $("tabBrowse").classList.toggle("active", b);
    $("tabBrowse").setAttribute("aria-selected", b);
    $("tabPredict").classList.toggle("active", !b);
    $("tabPredict").setAttribute("aria-selected", !b);
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
    });
    $("btnPredict").addEventListener("click", runPredict);
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

    const machine  = $("pMachine").value;
    const category = $("pCategory").value;
    const btn      = $("btnPredict");

    btn.disabled    = true;
    btn.textContent = t("btn_predicting");
    $("predictResult").classList.add("hidden");

    try {
      const result = await predict({ machine, category, thicknessMm: mm });
      showResult(result, machine, category);
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

  function showResult(r, machine, category) {
    const lang      = currentLang();
    const bladeDisp = lang === "ja" ? r.bladeJp : r.bladeEn;
    $("predictResult").innerHTML =
      '<div class="result-box">' +
        '<div class="result-header">' +
          '<span class="result-header-title">' + t("result_title") + '</span>' +
          '<span class="result-header-sub">' + esc(machine) + ' · ' + esc(tCat(category)) + '</span>' +
        '</div>' +
        '<div class="result-body">' +
          '<div class="result-specs">' +
            resultSpec(t("pressure_label"), r.pressure, false) +
            resultSpec(t("blade_label"),    bladeDisp,  true) +
            resultSpec(t("multicut_label"), r.multicut, false) +
          '</div>' +
          '<p class="result-disclaimer">' + t("disclaimer") + '</p>' +
        '</div>' +
      '</div>';
    $("predictResult").classList.remove("hidden");
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