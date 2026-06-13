/* i18n.js — EN / JP string tables */
(function () {
  const STRINGS = {
    en: {
      app_title:   "Material Selector",
      app_for:     "for Cricut",
      mode_browse: "Browse Materials",
      mode_predict:"✦ AI Predict",
      search_placeholder: "Search materials…",
      filter_all_categories: "All Categories",
      filter_all_blades:     "All Blades",
      sort_default: "Sort: Default",
      sort_asc:     "Sort: ↑ Pressure",
      sort_desc:    "Sort: ↓ Pressure",
      result_count: function (n) { return n === 1 ? "1 result" : n + " results"; },
      predict_title:"AI Cut Settings Predictor",
      predict_sub:  "Get estimated settings for any material not in the official list.",
      form_machine:   "Machine",
      form_material_name:     "Material Name",
      form_material_name_ph:  "e.g. Premium Vinyl, Cardstock…",
      form_material_name_hint:"Start typing to see known materials of the selected type, or enter any name for AI prediction.",
      form_category:  "Material Type",
      form_category_hint: "Filters the name suggestions; also used as fallback if material name is unknown.",
      form_thickness: "Thickness",
      btn_predict:    "Predict Cut Settings",
      btn_predicting: "Predicting…",
      result_title:   "✦ AI-Predicted Settings",
      disclaimer:     "⚠ AI estimate — always verify with a test cut before production.",
      btn_add_material:   "Incorrect? Add a new material",
      correct_title:      "Correct the values and save as a new material",
      btn_save_material:  "Save Material",
      btn_cancel:         "Cancel",
      err_name_required:  "Enter a material name.",
      err_duplicate_material: "This material already exists in the list (same name, type and thickness).",
      msg_material_saved: "Material saved — it now appears in the material browser.",
      custom_badge:       "Custom",
      err_thickness_required: "Enter a thickness value.",
      err_thickness_range:    "Thickness must be between 0.01 mm and 60 mm.",
      err_model_load: "Could not load AI model. Try refreshing the page.",
      empty_title: "No materials found",
      empty_sub:   "Try a different search or filter.",
      pressure_label:  "Pressure",
      blade_label:     "Blade",
      multicut_label:  "Multi-Cut",
      thickness_label: "Thickness",
      cat_board:      "Board / Cardboard",
      cat_cardstock:  "Cardstock",
      cat_fabric:     "Fabric",
      cat_infusible:  "Infusible Ink",
      cat_ironon:     "Iron-On",
      cat_leather:    "Leather",
      cat_others:     "Others",
      cat_paper:      "Paper",
      cat_plastic:    "Plastic",
      cat_printable:  "Printable Materials",
      cat_smart:      "Smart Materials",
      cat_vinyl:      "Vinyl",
      hero_headline:  "Find the right settings for every material.",
      hero_sub:       "Various cut settings for 1,039+ Cricut materials. AI prediction for anything else.",
      hero_cta_browse:"Browse Materials",
      hero_cta_predict:"✦ AI Predict",
      hero_stat_materials: "Materials",
      hero_stat_machines:  "Machines",
      hero_stat_categories:"Categories",
      sidebar_all:    "All Materials",
      sidebar_title:  "Categories",
      blade_fine:     "Fine-Point Blade",
      blade_deep:     "Deep-Point Blade",
      blade_rotary:   "Rotary Blade",
      blade_bonded:   "Bonded Fabric Blade",
      blade_knife:    "Knife Blade",
    },
    ja: {
      app_title:   "カット素材セレクター",
      app_for:     "for Cricut",
      mode_browse: "素材を探す",
      mode_predict:"✦ AIで予測",
      search_placeholder: "素材名で検索…",
      filter_all_categories: "すべてのカテゴリ",
      filter_all_blades:     "すべてのブレード",
      sort_default: "並べ替え：デフォルト",
      sort_asc:     "並べ替え：↑ カット圧",
      sort_desc:    "並べ替え：↓ カット圧",
      result_count: function (n) { return n + "件"; },
      predict_title:"AI カット設定予測",
      predict_sub:  "公式リストにない素材のカット設定を推定します。",
      form_machine:   "機種",
      form_material_name:     "素材名",
      form_material_name_ph:  "例：プレミアムビニール、カードストック…",
      form_material_name_hint:"選択したタイプの既知素材を検索、または任意の名前でAI予測。",
      form_category:  "素材タイプ",
      form_category_hint: "素材名の候補を絞り込みます。素材名が不明な場合のフォールバックにも使用します。",
      form_thickness: "厚さ",
      btn_predict:    "カット設定を予測する",
      btn_predicting: "予測中…",
      result_title:   "✦ AI予測カット設定",
      disclaimer:     "⚠ AI予測です — 本番カット前に必ずテストカットで確認してください。",
      btn_add_material:   "予測が違う？新しい素材を追加",
      correct_title:      "値を修正して新しい素材として保存",
      btn_save_material:  "素材を保存",
      btn_cancel:         "キャンセル",
      err_name_required:  "素材名を入力してください。",
      err_duplicate_material: "この素材は既にリストに存在します（同じ名前・タイプ・厚さ）。",
      msg_material_saved: "素材を保存しました — 素材ブラウザに表示されます。",
      custom_badge:       "カスタム",
      err_thickness_required: "厚さを入力してください。",
      err_thickness_range:    "厚さは 0.01 mm ～ 60 mm の範囲で入力してください。",
      err_model_load: "AIモデルを読み込めませんでした。ページを再読み込みしてください。",
      empty_title: "素材が見つかりません",
      empty_sub:   "別のキーワードやフィルターをお試しください。",
      pressure_label:  "カット圧",
      blade_label:     "ブレード",
      multicut_label:  "カット回数",
      thickness_label: "厚さ",
      cat_board:      "アートボード / 段ボール",
      cat_cardstock:  "カードストック",
      cat_fabric:     "布",
      cat_infusible:  "インフュージブルインク",
      cat_ironon:     "アイロン接着タイプ",
      cat_leather:    "革",
      cat_others:     "その他",
      cat_paper:      "紙",
      cat_plastic:    "プラスチック",
      cat_printable:  "印刷可能素材",
      cat_smart:      "スマート素材",
      cat_vinyl:      "ビニール",
      hero_headline:  "すべての素材に最適なカット設定を。",
      hero_sub:       "1,039件以上のCricut公式カット設定。未登録素材はAIで予測。",
      hero_cta_browse:"素材を探す",
      hero_cta_predict:"✦ AIで予測",
      hero_stat_materials: "素材数",
      hero_stat_machines:  "対応機種",
      hero_stat_categories:"カテゴリ",
      sidebar_all:    "すべての素材",
      sidebar_title:  "カテゴリ",
      blade_fine:     "ファインポイントブレード",
      blade_deep:     "ディープポイントブレード",
      blade_rotary:   "ロータリーブレード",
      blade_bonded:   "ボンデッドファブリックブレード",
      blade_knife:    "ナイフの刃",
    },
  };

  const CAT_KEY = {
    "Board/Cardboard":     "cat_board",
    "Cardstock":           "cat_cardstock",
    "Fabric":              "cat_fabric",
    "Infusible Ink":       "cat_infusible",
    "Iron-On":             "cat_ironon",
    "Leather":             "cat_leather",
    "Others":              "cat_others",
    "Paper":               "cat_paper",
    "Plastic":             "cat_plastic",
    "Printable Materials": "cat_printable",
    "Smart Materials":     "cat_smart",
    "Vinyl":               "cat_vinyl",
  };
  const BLADE_KEY = {
    "Fine-Point Blade":    "blade_fine",
    "Deep-Point Blade":    "blade_deep",
    "Rotary Blade":        "blade_rotary",
    "Bonded Fabric Blade": "blade_bonded",
    "Knife Blade":         "blade_knife",
  };

  let _lang = localStorage.getItem("cricutLang") ||
    ((navigator.language || "en").startsWith("ja") ? "ja" : "en");

  function t(key, arg) {
    const v = (STRINGS[_lang] || STRINGS.en)[key];
    if (typeof v === "function") return v(arg);
    return v !== undefined ? v : key;
  }
  function tCat(catEn)   { const k = CAT_KEY[catEn];   return k ? t(k) : catEn; }
  function tBlade(blEn)  { const k = BLADE_KEY[blEn];  return k ? t(k) : blEn; }
  function currentLang() { return _lang; }

  function setLang(lang) {
    _lang = lang;
    localStorage.setItem("cricutLang", lang);
    document.documentElement.lang = lang;
    if (lang === "ja") {
      if (!document.getElementById("noto-jp")) {
        const el = document.createElement("link");
        el.id   = "noto-jp";
        el.rel  = "stylesheet";
        el.href = "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap";
        document.head.appendChild(el);
      }
    }
  }

  window.i18n = { t, tCat, tBlade, currentLang, setLang };
})();