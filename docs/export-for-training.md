# Export for Training — Admin Feature

This feature lets you download all custom materials saved by a browser session
as a JSON file, which you can then feed into the training pipeline.

It is **disabled by default** because it is intended for use by the maintainer only.

---

## How to re-enable the Export button

Open `assets/js/app.js` and uncomment **two blocks**:

### Block 1 — the sidebar button (inside `buildSidebar()`, around line 355)

Remove the `/*` and `*/` surrounding this block:

```js
// EXPORT FOR TRAINING — admin-only feature, currently disabled.
// To re-enable: see docs/export-for-training.md
/*
const sidebar = $("catSidebar");
let exportEl = sidebar.querySelector(".sidebar-export");
...
*/
```

### Block 2 — the download function (just below `buildSidebar()`, around line 375)

Remove the `/*` and `*/` surrounding this block:

```js
// EXPORT FOR TRAINING — admin-only feature, currently disabled.
// To re-enable: see docs/export-for-training.md
/*
function exportCustomMaterials() {
  ...
}
*/
```

Once both blocks are uncommented, an **"Export for Training"** button appears at the
bottom of the sidebar whenever the current browser session has saved custom materials.

---

## Full active-learning workflow

1. **Enable** the export button (steps above).
2. Open the live site in the browser where custom materials were saved.
3. Click **"Export for Training"** — downloads `cricut_custom_materials.json`.
4. Copy the file into the project root.
5. Run the import script:
   ```bash
   source venv/bin/activate
   python scripts/import_custom_materials.py cricut_custom_materials.json
   ```
6. Review each material interactively (`[A]ccept / [E]dit / [S]kip`).
7. Rebuild, retrain, and deploy:
   ```bash
   python scripts/build_materials_json.py
   python scripts/train_model_v3.py
   cp assets/model/preprocessor_v3.json assets/model/preprocessor.json
   git add assets/data assets/model
   git commit -m "Promote custom materials — retrain vX.Y"
   git push
   ```
8. Once deployed, the next page load on the user's browser will **automatically**
   remove the promoted materials from their localStorage (they are now in the
   official list and no longer need to be stored locally).
9. **Disable** the export button again (re-add the `/*` … `*/` wrappers).
