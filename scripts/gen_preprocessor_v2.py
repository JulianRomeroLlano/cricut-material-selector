"""
gen_preprocessor_v2.py — Regenerate preprocessor_v2.json

Re-runs Phase 1 (global embedding training) to recover the embedding matrix,
then builds preprocessor_v2.json.  Skips Phase 2 (ONNX files already exist).

Use SEED=42 (same as train_model_v2.py) → deterministic, identical embeddings.

Run: source venv/bin/activate && python scripts/gen_preprocessor_v2.py
"""
import os, re, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

ROOT      = os.path.join(os.path.dirname(__file__), "..")
DATA_CSV  = os.path.join(ROOT, "assets", "data", "Material List (Augmented).csv")
MODEL_DIR = os.path.join(ROOT, "assets", "model")
PREP_JSON = os.path.join(MODEL_DIR, "preprocessor.json")   # deploy as the active file

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMB_DIM  = 16; N_PHYSICS = 7; N_MACH_TARGET = 5
LR_PHASE1 = 1e-3; MAX_EPOCHS = 800; PATIENCE = 80; EVAL_EVERY = 5
N_BLADE = 5; N_MCUT = 6
W_PRESSURE, W_BLADE, W_MCUT = 0.4, 0.4, 0.2

BLADE_JP_TO_EN = {
    "ディープポイントブレード":     "Deep-Point Blade",
    "ナイフの刃":                   "Knife Blade",
    "ファインポイントブレード":      "Fine-Point Blade",
    "ボンデッドファブリックブレード": "Bonded Fabric Blade",
    "ロータリーブレード":            "Rotary Blade",
}
TEXTURE_MAP = {
    "plain": 0.0, "matte": 0.1, "glossy": 0.15, "satin": 0.1,
    "shimmer": 0.2, "pearl": 0.15, "holographic": 0.25, "iridescent": 0.25,
    "embossed": 0.2, "woven": 0.3, "rough": 0.35, "smooth": 0.05,
}
MULTICUT_MAP = {"-":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10}
THICKNESS_DEFAULTS = {
    "Paper":0.08,"Cardstock":0.22,"Iron-On":0.10,"Vinyl":0.08,
    "Smart Materials":0.10,"Infusible Ink":0.10,"Printable Materials":0.12,
    "Board/Cardboard":1.0,"Leather":1.6,"Fabric":0.50,"Plastic":0.10,"Others":2.0,
}
MACHINE_CONFIGS = {
    "Cricut Joy":      {"slug": "cricut_joy_v2",       "hidden":[32,16],      "dropout":[0.45,0.35]},
    "Cricut Joy 2":    {"slug": "cricut_joy2_v2",      "hidden":[32,16],      "dropout":[0.45,0.35]},
    "Cricut Joy Xtra": {"slug": "cricut_joy_xtra_v2",  "hidden":[64,32],      "dropout":[0.40,0.30]},
    "Explore 3":       {"slug": "explore3_v2",         "hidden":[96,48],      "dropout":[0.35,0.25]},
    "Maker 3":         {"slug": "maker3_v2",           "hidden":[128,64,32],  "dropout":[0.30,0.25,0.0]},
}

# ─── Load & clean ─────────────────────────────────────────────────────────────

def normalize_name(name):
    c = re.sub(r'\s*\([^)]*\)\s*$', '', name.strip()).strip()
    return c or name.strip()

df_all = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
df_all = df_all[df_all["Cutting Pressure"].notna()].copy()
df_all["Cutting Pressure"] = df_all["Cutting Pressure"].astype(float)
df_all["Blade Type"] = df_all["Blade Type"].replace("ファインポイント", "ファインポイントブレード")
df_all = df_all[df_all["Category"] != "Pens & Markers"].copy()
print(f"Rows: {len(df_all)}")

if "Material Name Base" not in df_all.columns:
    df_all["Material Name Base"] = df_all["Material Name (EN)"].apply(normalize_name)

all_names   = sorted(df_all["Material Name Base"].unique().tolist())
name_to_idx = {n: i for i, n in enumerate(all_names)}
N_NAMES     = len(all_names)
MACHINES    = sorted(df_all["Machine"].unique().tolist())
N_MACH      = len(MACHINES)
mach_to_idx = {m: i for i, m in enumerate(MACHINES)}

BLADE_TYPES    = sorted(df_all["Blade Type"].unique().tolist())
BLADE_TYPES_EN = [BLADE_JP_TO_EN.get(jp, jp) for jp in BLADE_TYPES]
blade_idx      = {b: i for i, b in enumerate(BLADE_TYPES)}

def lb_to_mm(lb):
    pts = [(60,0.15),(65,0.18),(80,0.22),(100,0.27),(140,0.38)]
    for i in range(len(pts)-1):
        x0,y0=pts[i]; x1,y1=pts[i+1]
        if x0<=lb<=x1: return y0+(y1-y0)*(lb-x0)/(x1-x0)
    return 0.27 if lb>100 else 0.15

def infer_thickness(name, cat):
    m = re.search(r'(\d+\.?\d*)\s*mm', name, re.I)
    if m: return float(m.group(1))
    m = re.search(r'(\d+)\s*lb', name, re.I)
    if m: return lb_to_mm(int(m.group(1)))
    m = re.search(r'(\d+)\s*gsm', name, re.I)
    if m: return max(0.04, int(m.group(1))*0.001)
    return THICKNESS_DEFAULTS.get(cat, 0.5)

def bucket_mc(v):
    n = MULTICUT_MAP.get(str(v).strip(), 0)
    if n==0: return 0
    if n==2: return 1
    if n==3: return 2
    if 4<=n<=5: return 3
    if 6<=n<=8: return 4
    return 5

df_all["thickness"]   = df_all.apply(lambda r: infer_thickness(r["Material Name (EN)"], r["Category"]), axis=1)
df_all["mc_bucket"]   = df_all["Multi-Cut"].apply(bucket_mc)
df_all["is_bonded"]   = (df_all["Material Name (EN)"].str.contains("Bonded",case=False,na=False)
                          | (df_all["Blade Type"]=="ボンデッドファブリックブレード")).astype(float)
df_all["surface_mod"] = df_all["Surface Texture"].map(TEXTURE_MAP).fillna(0.0) \
                         if "Surface Texture" in df_all.columns else 0.0
df_all["has_adhesive"]= df_all["Has Adhesive"].astype(float) \
                         if "Has Adhesive" in df_all.columns else 0.0
df_all["gsm"]         = df_all["GSM"].astype(float)
df_all["density"]     = df_all["Density (kg/m3)"].astype(float) \
                         if "Density (kg/m3)" in df_all.columns else 500.0
df_all["shore"]       = df_all["Shore Hardness A"].astype(float) \
                         if "Shore Hardness A" in df_all.columns else 40.0
df_all["name_idx"]    = df_all["Material Name Base"].map(name_to_idx).astype(int)

gsm_log   = np.log1p(df_all["gsm"].values)
thick_log  = np.log1p(df_all["thickness"].values)
dens_log   = np.log1p(df_all["density"].values)
plog       = np.log(df_all["Cutting Pressure"].values)
G_GSM_MIN, G_GSM_MAX     = float(gsm_log.min()),   float(gsm_log.max())
G_THICK_MIN, G_THICK_MAX = float(thick_log.min()),  float(thick_log.max())
G_DENS_MIN, G_DENS_MAX   = float(dens_log.min()),   float(dens_log.max())
G_P_MEAN  = float(plog.mean())
G_P_STD   = float(plog.std())

def clamp01(x): return np.clip(x, 0.0, 1.0)

def build_physics(df):
    g = clamp01((np.log1p(df["gsm"].values)      - G_GSM_MIN)   / (G_GSM_MAX   - G_GSM_MIN   + 1e-9))
    t = clamp01((np.log1p(df["thickness"].values) - G_THICK_MIN) / (G_THICK_MAX - G_THICK_MIN + 1e-9))
    ib= df["is_bonded"].values.astype(np.float32)
    sm= df["surface_mod"].values.astype(np.float32)
    ha= df["has_adhesive"].values.astype(np.float32)
    d = clamp01((np.log1p(df["density"].values)   - G_DENS_MIN)  / (G_DENS_MAX  - G_DENS_MIN  + 1e-9))
    s = clamp01(df["shore"].values / 100.0)
    return np.stack([g, t, ib, sm, ha, d, s], axis=1).astype(np.float32)

physics_all  = build_physics(df_all)
name_idx_all = df_all["name_idx"].values.astype(np.int64)
mach_oh_all  = np.zeros((len(df_all), N_MACH), dtype=np.float32)
for i, m in enumerate(df_all["Machine"]): mach_oh_all[i, mach_to_idx[m]] = 1.0
p_tgt  = ((np.log(df_all["Cutting Pressure"].values) - G_P_MEAN) / (G_P_STD + 1e-9)).astype(np.float32)
b_tgt  = np.array([blade_idx.get(b, 0) for b in df_all["Blade Type"]], dtype=np.int64)
mc_tgt = df_all["mc_bucket"].values.astype(np.int64)

idx = np.arange(len(df_all))
tr_idx, va_idx = train_test_split(idx, test_size=0.10, random_state=SEED)

# ─── Phase 1 Model ────────────────────────────────────────────────────────────

class Phase1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(N_NAMES, EMB_DIM)
        nn.init.normal_(self.embedding.weight, std=0.1)
        in_dim = EMB_DIM + N_PHYSICS + N_MACH
        hidden = [256, 128, 64]; dropout = [0.30, 0.25, 0.15]
        layers = []
        for h, d in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(d)]
            in_dim = h
        self.bb = nn.Sequential(*layers)
        self.hp = nn.Linear(in_dim, 1)
        self.hb = nn.Linear(in_dim, N_BLADE)
        self.hm = nn.Linear(in_dim, N_MCUT)

    def forward(self, ni, ph, mo):
        x = torch.cat([self.embedding(ni), ph, mo], dim=1)
        f = self.bb(x)
        return self.hp(f).squeeze(1), self.hb(f), self.hm(f)

def loss_fn(pp, pb, pm, pt, bt, mt):
    return (W_PRESSURE * nn.functional.mse_loss(pp, pt)
          + W_BLADE    * nn.functional.cross_entropy(pb, bt)
          + W_MCUT     * nn.functional.cross_entropy(pm, mt))

print("\n" + "═"*60)
print("Phase 1 — Global Embedding Training")
print("═"*60)

tr_ds = TensorDataset(
    torch.from_numpy(name_idx_all[tr_idx]), torch.from_numpy(physics_all[tr_idx]),
    torch.from_numpy(mach_oh_all[tr_idx]),  torch.from_numpy(p_tgt[tr_idx]),
    torch.from_numpy(b_tgt[tr_idx]),        torch.from_numpy(mc_tgt[tr_idx]),
)
va_ds = TensorDataset(
    torch.from_numpy(name_idx_all[va_idx]), torch.from_numpy(physics_all[va_idx]),
    torch.from_numpy(mach_oh_all[va_idx]),  torch.from_numpy(p_tgt[va_idx]),
    torch.from_numpy(b_tgt[va_idx]),        torch.from_numpy(mc_tgt[va_idx]),
)
tr_ld = DataLoader(tr_ds, batch_size=512, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=1024)

model = Phase1Model().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
best_val, best_ep, no_imp, best_st = float("inf"), 0, 0, None

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for ni, ph, mo, pt, bt, mt in tr_ld:
        ni,ph,mo,pt,bt,mt = ni.to(DEVICE),ph.to(DEVICE),mo.to(DEVICE),pt.to(DEVICE),bt.to(DEVICE),mt.to(DEVICE)
        opt.zero_grad()
        pp,pb,pm = model(ni,ph,mo)
        loss_fn(pp,pb,pm,pt,bt,mt).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

    if epoch % EVAL_EVERY == 0:
        model.eval(); vl = 0
        with torch.no_grad():
            for ni,ph,mo,pt,bt,mt in va_ld:
                ni,ph,mo,pt,bt,mt = ni.to(DEVICE),ph.to(DEVICE),mo.to(DEVICE),pt.to(DEVICE),bt.to(DEVICE),mt.to(DEVICE)
                pp,pb,pm = model(ni,ph,mo)
                vl += loss_fn(pp,pb,pm,pt,bt,mt).item()
        vl /= len(va_ld)
        if vl < best_val: best_val,best_ep,no_imp,best_st = vl,epoch,0,copy.deepcopy(model.state_dict())
        else: no_imp += EVAL_EVERY
        if epoch % 100 == 0:
            print(f"  ep {epoch:4d}  val={vl:.5f}  best={best_val:.5f}@{best_ep}")
        if no_imp >= PATIENCE: print(f"  Early stop at {epoch}"); break

model.load_state_dict(best_st)
EMB_MATRIX = model.embedding.weight.detach().cpu().numpy()
np.save(os.path.join(MODEL_DIR, "embedding_v2.npy"), EMB_MATRIX)
print(f"Phase 1 done — best={best_val:.5f}@ep{best_ep}  emb={EMB_MATRIX.shape}")

# ─── Build material lookup & category avg embeddings ─────────────────────────

orig_mask = np.isclose(df_all["aug_factor"].values
                        if "aug_factor" in df_all.columns else np.ones(len(df_all)), 1.0)
df_orig   = df_all[orig_mask].copy() if orig_mask.any() else df_all.copy()

material_lookup = {}
for base_name in all_names:
    rows = df_orig[df_orig["Material Name Base"] == base_name]
    if rows.empty: rows = df_all[df_all["Material Name Base"] == base_name]
    r = rows.iloc[0]
    material_lookup[base_name] = {
        "category":    str(r["Category"]),
        "gsm":         round(float(rows["gsm"].median()), 2),
        "density":     round(float(rows["density"].median()), 2),
        "shore":       round(float(rows["shore"].median()), 2),
        "texture":     round(float(r["surface_mod"]) if "surface_mod" in r.index else 0.0, 4),
        "has_adhesive":round(float(r["has_adhesive"]) if "has_adhesive" in r.index else 0.0, 4),
        "is_bonded":   round(float(r["is_bonded"]) if "is_bonded" in r.index else 0.0, 4),
        "thickness_mm":round(float(rows["thickness"].median()), 4),
    }

category_avg_emb = {}
for cat in df_all["Category"].unique():
    cat_rows = df_all[df_all["Category"] == cat]
    idxs = cat_rows["name_idx"].values.astype(int)
    avg  = EMB_MATRIX[idxs].mean(axis=0)
    category_avg_emb[str(cat)] = [float(x) for x in avg]

# ─── Machine metadata (from existing ONNX files) ─────────────────────────────

per_machine_meta = {}
for mname, cfg in MACHINE_CONFIGS.items():
    slug     = cfg["slug"]
    onnx_path = os.path.join(MODEL_DIR, f"material_predictor_{slug}.onnx")
    size_kb   = os.path.getsize(onnx_path) // 1024 if os.path.exists(onnx_path) else 0
    per_machine_meta[mname] = {"slug": slug, "onnx_kb": size_kb}

# ─── Save preprocessor.json ──────────────────────────────────────────────────

compatible_blades = {
    "Cricut Joy":      ["Fine-Point Blade"],
    "Cricut Joy 2":    ["Fine-Point Blade"],
    "Cricut Joy Xtra": ["Fine-Point Blade", "Deep-Point Blade"],
    "Explore 3":       ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade"],
    "Maker 3":         ["Fine-Point Blade", "Deep-Point Blade", "Rotary Blade",
                        "Bonded Fabric Blade", "Knife Blade"],
}

preprocessor = {
    "version":           "v2",
    "feature_dim":       EMB_DIM + N_PHYSICS,
    "emb_dim":           EMB_DIM,
    "n_physics":         N_PHYSICS,
    "n_names":           N_NAMES,
    "pressure_log_mean": G_P_MEAN,
    "pressure_log_std":  G_P_STD,
    "gsm_log_min":       G_GSM_MIN,
    "gsm_log_max":       G_GSM_MAX,
    "thickness_log_min": G_THICK_MIN,
    "thickness_log_max": G_THICK_MAX,
    "density_log_min":   G_DENS_MIN,
    "density_log_max":   G_DENS_MAX,
    "blade_types_en":    BLADE_TYPES_EN,
    "blade_types_jp":    BLADE_TYPES,
    "name_vocab":        {k: int(v) for k, v in name_to_idx.items()},
    "name_embeddings":   {n: [float(x) for x in EMB_MATRIX[i]] for n, i in name_to_idx.items()},
    "category_avg_embeddings": category_avg_emb,
    "material_lookup":   material_lookup,
    "machines":          per_machine_meta,
    "notes": {
        "compatible_blades": compatible_blades,
        "feature_order": [
            "name_emb[0..15]", "gsm_lognorm", "thickness_lognorm",
            "is_bonded", "surface_texture", "has_adhesive", "density_lognorm", "shore_norm",
        ],
    },
}

with open(PREP_JSON, "w", encoding="utf-8") as f:
    json.dump(preprocessor, f, indent=2, ensure_ascii=False)

kb = os.path.getsize(PREP_JSON) // 1024
print(f"\nSaved preprocessor.json  ({kb} KB)")
print(f"  Names: {N_NAMES}  Material lookup: {len(material_lookup)}  Categories: {len(category_avg_emb)}")
print("\nAll done — preprocessor.json ready for deployment.")
