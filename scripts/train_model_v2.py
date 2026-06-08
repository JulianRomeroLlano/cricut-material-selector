"""
train_model_v2.py — Material Name Embedding + GSM Augmentation

Architecture change from v1:
  v1: Category one-hot (11) + 8 physics = 19 features
  v2: Name embedding  (16) + 7 physics = 23 features
      (hardness_norm dropped — implicit in name embedding)

Features (N_IN = 23):
  name_embedding(16) + gsm_lognorm(1) + thickness_lognorm(1)
  + is_bonded(1) + surface_texture(1) + has_adhesive(1)
  + density_lognorm(1) + shore_norm(1)

Training procedure:
  Phase 1 — Global embedding (all machines together):
    Input: name_emb(16) + physics(7) + machine_onehot(5) = 28
    MLP: 28 → 256 → 128 → 64 → 3 heads
    Trains the embedding to encode universal material properties.

  Phase 2 — Per-machine fine-tuning (frozen embedding):
    Input: name_emb(16) + physics(7) = 23  [pre-computed in JS]
    MLP: machine-specific hidden dims → 3 heads
    Embedding matrix is fixed; only MLP weights are updated.
    ONNX model exported for each machine (input: float32, shape [1,23]).

Outputs:
  assets/model/material_predictor_{slug}_v2.onnx  (5 files)
  assets/model/preprocessor_v2.json

Run: source venv/bin/activate && python scripts/train_model_v2.py
"""
import os, re, json, copy, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_CSV   = os.path.join(ROOT, "assets", "data", "Material List (Augmented).csv")
MODEL_DIR  = os.path.join(ROOT, "assets", "model")
PREP_JSON  = os.path.join(MODEL_DIR, "preprocessor_v2.json")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EMB_DIM             = 16
N_PHYSICS           = 7
N_IN                = EMB_DIM + N_PHYSICS   # 23

LR_PHASE1           = 1e-3
LR_PHASE2           = 5e-4
LR_FINETUNE         = 2e-4
MAX_EPOCHS_PHASE1   = 800
MAX_EPOCHS_PHASE2   = 2000
PATIENCE_PHASE1     = 80
PATIENCE_PHASE2     = 250
EVAL_EVERY          = 5

N_BLADE  = 5
N_MCUT   = 6
W_PRESSURE, W_BLADE, W_MCUT = 0.4, 0.4, 0.2
MULTICUT_BUCKET_LABELS = [0, 2, 3, 4, 6, 10]

PRETRAIN_ORDER = ["Maker 3", "Explore 3", "Cricut Joy Xtra", "Cricut Joy 2", "Cricut Joy"]

MACHINE_CONFIGS = {
    "Cricut Joy":      {"slug": "cricut_joy_v2",      "hidden": [32, 16],      "dropout": [0.45, 0.35]},
    "Cricut Joy 2":    {"slug": "cricut_joy2_v2",     "hidden": [32, 16],      "dropout": [0.45, 0.35]},
    "Cricut Joy Xtra": {"slug": "cricut_joy_xtra_v2", "hidden": [64, 32],      "dropout": [0.40, 0.30]},
    "Explore 3":       {"slug": "explore3_v2",        "hidden": [96, 48],      "dropout": [0.35, 0.25]},
    "Maker 3":         {"slug": "maker3_v2",          "hidden": [128, 64, 32], "dropout": [0.30, 0.25, 0.0]},
}

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
MULTICUT_MAP = {
    "-": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & Clean Augmented Data
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', name.strip()).strip()
    return cleaned or name.strip()

df_all = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
df_all = df_all[df_all["Cutting Pressure"].notna()].copy()
df_all["Cutting Pressure"] = df_all["Cutting Pressure"].astype(float)
df_all["Blade Type"] = df_all["Blade Type"].replace("ファインポイント", "ファインポイントブレード")
df_all = df_all[df_all["Category"] != "Pens & Markers"].copy()
print(f"Augmented rows loaded: {len(df_all)}")

# Use pre-computed normalized name if present, else compute
if "Material Name Base" not in df_all.columns:
    df_all["Material Name Base"] = df_all["Material Name (EN)"].apply(normalize_name)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Name Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

all_names   = sorted(df_all["Material Name Base"].unique().tolist())
name_to_idx = {n: i for i, n in enumerate(all_names)}
N_NAMES     = len(all_names)
print(f"Unique base material names: {N_NAMES}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _lb_to_mm(lb):
    pts = [(60, 0.15), (65, 0.18), (80, 0.22), (100, 0.27), (140, 0.38)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]
        if x0 <= lb <= x1:
            return y0 + (y1 - y0) * (lb - x0) / (x1 - x0)
    return 0.27 if lb > 100 else 0.15

THICKNESS_DEFAULTS = {
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10, "Vinyl": 0.08,
    "Smart Materials": 0.10, "Printable Materials": 0.12, "Infusible Ink": 0.10,
    "Board/Cardboard": 1.0, "Leather": 1.6, "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
}

def infer_thickness(name, category):
    m = re.search(r"(\d+\.?\d*)\s*mm", name, re.IGNORECASE)
    if m: return float(m.group(1))
    m = re.search(r"(\d+)\s*lb", name, re.IGNORECASE)
    if m: return _lb_to_mm(int(m.group(1)))
    m = re.search(r"(\d+)\s*gsm", name, re.IGNORECASE)
    if m: return max(0.04, int(m.group(1)) * 0.001)
    return THICKNESS_DEFAULTS.get(category, 0.5)

def bucket_multicut(val):
    n = MULTICUT_MAP.get(str(val).strip(), 0)
    if n == 0:      return 0
    if n == 2:      return 1
    if n == 3:      return 2
    if 4 <= n <= 5: return 3
    if 6 <= n <= 8: return 4
    return 5

df_all["thickness"]    = df_all.apply(
    lambda r: infer_thickness(r["Material Name (EN)"], r["Category"]), axis=1)
df_all["mc_bucket"]    = df_all["Multi-Cut"].apply(lambda v: bucket_multicut(v))
df_all["is_bonded"]    = (
    df_all["Material Name (EN)"].str.contains("Bonded", case=False, na=False)
    | (df_all["Blade Type"] == "ボンデッドファブリックブレード")
).astype(float)
df_all["surface_mod"]  = df_all["Surface Texture"].map(TEXTURE_MAP).fillna(0.0) \
                          if "Surface Texture" in df_all.columns else 0.0
df_all["has_adhesive"] = df_all["Has Adhesive"].astype(float) \
                          if "Has Adhesive" in df_all.columns else 0.0
df_all["gsm"]          = df_all["GSM"].astype(float)
df_all["density"]      = df_all["Density (kg/m3)"].astype(float) \
                          if "Density (kg/m3)" in df_all.columns else 500.0
df_all["shore"]        = df_all["Shore Hardness A"].astype(float) \
                          if "Shore Hardness A" in df_all.columns else 40.0
df_all["name_idx"]     = df_all["Material Name Base"].map(name_to_idx).astype(int)

# Global normalization constants (computed over ALL augmented rows)
BLADE_TYPES    = sorted(df_all["Blade Type"].unique().tolist())
BLADE_TYPES_EN = [BLADE_JP_TO_EN.get(jp, jp) for jp in BLADE_TYPES]
blade_idx      = {b: i for i, b in enumerate(BLADE_TYPES)}

gsm_log_all    = np.log1p(df_all["gsm"].values)
thick_log_all  = np.log1p(df_all["thickness"].values)
dens_log_all   = np.log1p(df_all["density"].values)
plog_all       = np.log(df_all["Cutting Pressure"].values)

G_GSM_MIN,  G_GSM_MAX   = gsm_log_all.min(),   gsm_log_all.max()
G_THICK_MIN, G_THICK_MAX = thick_log_all.min(), thick_log_all.max()
G_DENS_MIN,  G_DENS_MAX  = dens_log_all.min(),  dens_log_all.max()
G_PRESSURE_LOG_MEAN      = float(plog_all.mean())
G_PRESSURE_LOG_STD       = float(plog_all.std())

MACHINES = sorted(df_all["Machine"].unique().tolist())
mach_to_idx = {m: i for i, m in enumerate(MACHINES)}
N_MACH = len(MACHINES)

print(f"Blade types: {BLADE_TYPES_EN}")
print(f"Global pressure: log_mean={G_PRESSURE_LOG_MEAN:.4f}  log_std={G_PRESSURE_LOG_STD:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature Matrix (physics only, 7 dims)
# ═══════════════════════════════════════════════════════════════════════════════

def clamp01(x): return np.clip(x, 0.0, 1.0)

def build_physics(df: pd.DataFrame) -> np.ndarray:
    g_n = clamp01((np.log1p(df["gsm"].values)      - G_GSM_MIN)   / (G_GSM_MAX   - G_GSM_MIN   + 1e-9))
    t_n = clamp01((np.log1p(df["thickness"].values) - G_THICK_MIN) / (G_THICK_MAX - G_THICK_MIN + 1e-9))
    ib  = df["is_bonded"].values.astype(np.float32)
    sm  = df["surface_mod"].values.astype(np.float32)
    ha  = df["has_adhesive"].values.astype(np.float32)
    d_n = clamp01((np.log1p(df["density"].values)   - G_DENS_MIN)  / (G_DENS_MAX  - G_DENS_MIN  + 1e-9))
    sh  = clamp01(df["shore"].values / 100.0)
    return np.stack([g_n, t_n, ib, sm, ha, d_n, sh], axis=1).astype(np.float32)

def build_pressure_target(df):
    return ((np.log(df["Cutting Pressure"].values) - G_PRESSURE_LOG_MEAN)
            / (G_PRESSURE_LOG_STD + 1e-9)).astype(np.float32)

def build_blade_target(df):
    return np.array([blade_idx.get(b, 0) for b in df["Blade Type"]], dtype=np.int64)

def build_mcut_target(df):
    return df["mc_bucket"].values.astype(np.int64)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class Phase1Model(nn.Module):
    """Global model: Embedding + physics + machine one-hot → heads."""
    def __init__(self, n_names, emb_dim, n_physics, n_mach, hidden, dropout):
        super().__init__()
        self.embedding = nn.Embedding(n_names, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        in_dim = emb_dim + n_physics + n_mach
        layers = []
        for h, d in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(d)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head_p  = nn.Linear(in_dim, 1)
        self.head_b  = nn.Linear(in_dim, N_BLADE)
        self.head_mc = nn.Linear(in_dim, N_MCUT)

    def forward(self, name_idx, physics, mach_oh):
        emb  = self.embedding(name_idx)          # (B, emb_dim)
        x    = torch.cat([emb, physics, mach_oh], dim=1)
        feat = self.backbone(x)
        return self.head_p(feat).squeeze(1), self.head_b(feat), self.head_mc(feat)


class Phase2Model(nn.Module):
    """Per-machine model: pre-computed embedding concat with physics → heads."""
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, in_dim = [], n_in
        for h, d in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(d)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head_p  = nn.Linear(in_dim, 1)
        self.head_b  = nn.Linear(in_dim, N_BLADE)
        self.head_mc = nn.Linear(in_dim, N_MCUT)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_p(feat).squeeze(1), self.head_b(feat), self.head_mc(feat)


def loss_fn(p_pred, b_pred, mc_pred, p_true, b_true, mc_true):
    lp  = nn.functional.mse_loss(p_pred, p_true)
    lb  = nn.functional.cross_entropy(b_pred, b_true)
    lmc = nn.functional.cross_entropy(mc_pred, mc_true)
    return W_PRESSURE * lp + W_BLADE * lb + W_MCUT * lmc

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Phase 1 — Global Embedding Training
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("PHASE 1 — Global Embedding Training (all machines)")
print("═"*70)

df_p1 = df_all.copy()
physics_all = build_physics(df_p1)
name_idx_all = df_p1["name_idx"].values.astype(np.int64)
mach_oh_all  = np.zeros((len(df_p1), N_MACH), dtype=np.float32)
for i, m in enumerate(df_p1["Machine"]):
    mach_oh_all[i, mach_to_idx[m]] = 1.0
p_tgt  = build_pressure_target(df_p1)
b_tgt  = build_blade_target(df_p1)
mc_tgt = build_mcut_target(df_p1)

idx = np.arange(len(df_p1))
tr_idx, va_idx = train_test_split(idx, test_size=0.10, random_state=SEED)

def make_phase1_loaders(batch=512):
    tr = TensorDataset(
        torch.from_numpy(name_idx_all[tr_idx]),
        torch.from_numpy(physics_all[tr_idx]),
        torch.from_numpy(mach_oh_all[tr_idx]),
        torch.from_numpy(p_tgt[tr_idx]),
        torch.from_numpy(b_tgt[tr_idx]),
        torch.from_numpy(mc_tgt[tr_idx]),
    )
    va = TensorDataset(
        torch.from_numpy(name_idx_all[va_idx]),
        torch.from_numpy(physics_all[va_idx]),
        torch.from_numpy(mach_oh_all[va_idx]),
        torch.from_numpy(p_tgt[va_idx]),
        torch.from_numpy(b_tgt[va_idx]),
        torch.from_numpy(mc_tgt[va_idx]),
    )
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(va, batch_size=batch*2))

p1_hidden  = [256, 128, 64]
p1_dropout = [0.30, 0.25, 0.15]
p1_model   = Phase1Model(N_NAMES, EMB_DIM, N_PHYSICS, N_MACH,
                         p1_hidden, p1_dropout).to(DEVICE)
n_params = sum(p.numel() for p in p1_model.parameters() if p.requires_grad)
print(f"Phase 1 model: {N_NAMES} names × {EMB_DIM} emb + {N_PHYSICS} physics + {N_MACH} mach = {EMB_DIM+N_PHYSICS+N_MACH} in")
print(f"  Architecture: {'→'.join(str(h) for h in p1_hidden)}→heads  ({n_params:,} params)")

tr_loader, va_loader = make_phase1_loaders()
optimizer = torch.optim.AdamW(p1_model.parameters(), lr=LR_PHASE1, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_PHASE1)

best_val, best_epoch, no_improve = float("inf"), 0, 0
best_state = None

for epoch in range(1, MAX_EPOCHS_PHASE1 + 1):
    p1_model.train()
    for nidx, ph, moh, pt, bt, mt in tr_loader:
        nidx, ph, moh = nidx.to(DEVICE), ph.to(DEVICE), moh.to(DEVICE)
        pt, bt, mt = pt.to(DEVICE), bt.to(DEVICE), mt.to(DEVICE)
        optimizer.zero_grad()
        pp, pb, pm = p1_model(nidx, ph, moh)
        loss_fn(pp, pb, pm, pt, bt, mt).backward()
        nn.utils.clip_grad_norm_(p1_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    if epoch % EVAL_EVERY == 0:
        p1_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for nidx, ph, moh, pt, bt, mt in va_loader:
                nidx, ph, moh = nidx.to(DEVICE), ph.to(DEVICE), moh.to(DEVICE)
                pt, bt, mt = pt.to(DEVICE), bt.to(DEVICE), mt.to(DEVICE)
                pp, pb, pm = p1_model(nidx, ph, moh)
                val_loss += loss_fn(pp, pb, pm, pt, bt, mt).item()
        val_loss /= len(va_loader)

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            best_state = copy.deepcopy(p1_model.state_dict())
        else:
            no_improve += EVAL_EVERY

        if epoch % 100 == 0:
            print(f"  ep {epoch:4d}  val_loss={val_loss:.5f}  best={best_val:.5f}@ep{best_epoch}")

        if no_improve >= PATIENCE_PHASE1:
            print(f"  Early stop at epoch {epoch}")
            break

p1_model.load_state_dict(best_state)
print(f"Phase 1 complete — best val_loss={best_val:.5f} at epoch {best_epoch}")

# Extract and checkpoint embedding matrix
EMB_MATRIX = p1_model.embedding.weight.detach().cpu().numpy()  # (N_NAMES, EMB_DIM)
EMB_CKPT   = os.path.join(MODEL_DIR, "embedding_v2.npy")
np.save(EMB_CKPT, EMB_MATRIX)
print(f"Embedding matrix extracted + saved: {EMB_MATRIX.shape} → {os.path.basename(EMB_CKPT)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Phase 2 — Per-Machine Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("PHASE 2 — Per-Machine Fine-Tuning (frozen embedding)")
print("═"*70)

# Pre-compute embedding vectors for all rows (embedding is now fixed)
emb_all = EMB_MATRIX[name_idx_all]           # (N, 16)
X_all   = np.concatenate([emb_all, physics_all], axis=1).astype(np.float32)  # (N, 23)

results = {}

def train_machine(machine_name, cfg):
    print(f"\n── {machine_name} ──")
    mask    = df_all["Machine"].values == machine_name
    X_mach  = X_all[mask]
    p_mach  = p_tgt[mask]
    b_mach  = b_tgt[mask]
    mc_mach = mc_tgt[mask]

    idx_m = np.arange(len(X_mach))
    tr_m, va_m = train_test_split(idx_m, test_size=0.10, random_state=SEED)
    print(f"  train={len(tr_m)}  val={len(va_m)}")

    tr_ds = TensorDataset(torch.from_numpy(X_mach[tr_m]),
                          torch.from_numpy(p_mach[tr_m]),
                          torch.from_numpy(b_mach[tr_m]),
                          torch.from_numpy(mc_mach[tr_m]))
    va_ds = TensorDataset(torch.from_numpy(X_mach[va_m]),
                          torch.from_numpy(p_mach[va_m]),
                          torch.from_numpy(b_mach[va_m]),
                          torch.from_numpy(mc_mach[va_m]))
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=512)

    model = Phase2Model(N_IN, cfg["hidden"], cfg["dropout"]).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: {N_IN}→{'→'.join(str(h) for h in cfg['hidden'])}→heads  ({n_p:,} params)")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_PHASE2)

    best_val, best_ep, no_imp, best_st = float("inf"), 0, 0, None

    for epoch in range(1, MAX_EPOCHS_PHASE2 + 1):
        model.train()
        for xb, pb, bb, mb in tr_ld:
            xb, pb, bb, mb = xb.to(DEVICE), pb.to(DEVICE), bb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            pp, bp, mp = model(xb)
            loss_fn(pp, bp, mp, pb, bb, mb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            vl, p_preds, b_preds, p_trues, b_trues = 0, [], [], [], []
            with torch.no_grad():
                for xb, pb, bb, mb in va_ld:
                    xb, pb, bb, mb = xb.to(DEVICE), pb.to(DEVICE), bb.to(DEVICE), mb.to(DEVICE)
                    pp, bp, mp = model(xb)
                    vl += loss_fn(pp, bp, mp, pb, bb, mb).item()
                    p_preds.extend(pp.cpu().numpy())
                    b_preds.extend(bp.argmax(1).cpu().numpy())
                    p_trues.extend(pb.cpu().numpy())
                    b_trues.extend(bb.cpu().numpy())
            vl /= len(va_ld)

            if vl < best_val:
                best_val, best_ep, no_imp, best_st = vl, epoch, 0, copy.deepcopy(model.state_dict())
            else:
                no_imp += EVAL_EVERY

            if epoch % 200 == 0:
                pa = accuracy_score(b_trues, b_preds)
                # decode pressure for MAE
                p_dec  = np.exp(np.array(p_preds) * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
                pt_dec = np.exp(np.array(p_trues) * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
                mae    = mean_absolute_error(pt_dec, p_dec)
                print(f"    ep {epoch:4d}  val_loss={vl:.5f}  blade_acc={pa:.3f}  p_MAE={mae:.1f}")

            if no_imp >= PATIENCE_PHASE2:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_st)
    model.eval()

    # Final validation metrics
    p_preds, b_preds, mc_preds, p_trues, b_trues, mc_trues = [], [], [], [], [], []
    with torch.no_grad():
        for xb, pb, bb, mb in va_ld:
            xb = xb.to(DEVICE)
            pp, bp, mp = model(xb)
            p_preds.extend(pp.cpu().numpy())
            b_preds.extend(bp.argmax(1).cpu().numpy())
            mc_preds.extend(mp.argmax(1).cpu().numpy())
            p_trues.extend(pb.numpy())
            b_trues.extend(bb.numpy())
            mc_trues.extend(mb.numpy())

    p_dec  = np.exp(np.array(p_preds) * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
    pt_dec = np.exp(np.array(p_trues) * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
    mae    = mean_absolute_error(pt_dec, p_dec)
    mape   = np.mean(np.abs(p_dec - pt_dec) / (pt_dec + 1e-9)) * 100
    bacc   = accuracy_score(b_trues, b_preds)
    bf1    = f1_score(b_trues, b_preds, average="weighted", zero_division=0)
    mcacc  = accuracy_score(mc_trues, mc_preds)
    print(f"  ✓ Pressure MAE={mae:.1f}  MAPE={mape:.1f}%  | Blade acc={bacc:.3f}  F1={bf1:.3f}  | MC acc={mcacc:.3f}")

    # ONNX export
    onnx_path = os.path.join(MODEL_DIR, f"material_predictor_{cfg['slug']}.onnx")
    dummy = torch.zeros(1, N_IN).to(DEVICE)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=12,
        input_names=["features"],
        output_names=["pressure_norm", "blade_logits", "multicut_logits"],
        dynamic_axes={"features": {0: "batch"}, "pressure_norm": {0: "batch"},
                      "blade_logits": {0: "batch"}, "multicut_logits": {0: "batch"}},
    )
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_path)
    out  = sess.run(None, {"features": np.zeros((1, N_IN), dtype=np.float32)})
    size_kb = os.path.getsize(onnx_path) // 1024
    print(f"  ONNX → {os.path.basename(onnx_path)}  ({size_kb} KB)  ✓")

    return {"n_train": int(len(tr_m)), "n_val": int(len(va_m)),
            "best_epoch": int(best_ep), "mae": float(mae), "mape": float(mape),
            "blade_acc": float(bacc), "mc_acc": float(mcacc)}

for machine_name, cfg in MACHINE_CONFIGS.items():
    results[machine_name] = train_machine(machine_name, cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Build Material Lookup Table
# ═══════════════════════════════════════════════════════════════════════════════

# For each unique base name: store representative physics properties + category + thickness
# (use the median GSM row where aug_factor=1.0, i.e. the original data point)
orig_mask = np.isclose(df_all["aug_factor"].values if "aug_factor" in df_all.columns
                       else np.ones(len(df_all)), 1.0)
df_orig   = df_all[orig_mask].copy() if orig_mask.any() else df_all.copy()

material_lookup = {}
category_avg_emb = {}   # category → average embedding vector

for base_name in all_names:
    rows = df_orig[df_orig["Material Name Base"] == base_name]
    if rows.empty:
        rows = df_all[df_all["Material Name Base"] == base_name]

    # Median properties across all machines for this material
    r = rows.iloc[0]
    gsm_val = float(rows["gsm"].median())
    density_val = float(rows["density"].median())
    thickness_mm = float(rows["thickness"].median())

    material_lookup[base_name] = {
        "category":    r["Category"],
        "gsm":         round(gsm_val, 2),
        "density":     round(density_val, 2),
        "shore":       round(float(rows["shore"].median()), 2),
        "texture":     float(r["surface_mod"]) if "surface_mod" in r.index else 0.0,
        "has_adhesive":float(r["has_adhesive"]) if "has_adhesive" in r.index else 0.0,
        "is_bonded":   float(r["is_bonded"]) if "is_bonded" in r.index else 0.0,
        "thickness_mm":round(thickness_mm, 4),
    }

# Category average embeddings (fallback for unknown materials)
for cat in df_all["Category"].unique():
    cat_rows = df_all[df_all["Category"] == cat]
    cat_name_idxs = cat_rows["name_idx"].values.astype(int)
    cat_embs = EMB_MATRIX[cat_name_idxs]
    category_avg_emb[cat] = cat_embs.mean(axis=0).tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save preprocessor_v2.json
# ═══════════════════════════════════════════════════════════════════════════════

compatible_blades = {
    "Cricut Joy":      ["Fine-Point Blade"],
    "Cricut Joy 2":    ["Fine-Point Blade"],
    "Cricut Joy Xtra": ["Fine-Point Blade", "Deep-Point Blade"],
    "Explore 3":       ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade"],
    "Maker 3":         ["Fine-Point Blade", "Deep-Point Blade", "Rotary Blade",
                        "Bonded Fabric Blade", "Knife Blade"],
}

per_machine_meta = {}
for mname, cfg in MACHINE_CONFIGS.items():
    r = results[mname]
    per_machine_meta[mname] = {
        "slug":          cfg["slug"],
        "n_train":       r["n_train"],
        "n_val":         r["n_val"],
        "best_epoch":    r["best_epoch"],
        "pressure_mae":  round(r["mae"], 1),
        "pressure_mape": round(r["mape"], 1),
        "blade_acc":     round(r["blade_acc"], 4),
        "mc_acc":        round(r["mc_acc"], 4),
    }

preprocessor = {
    "version":           "v2",
    "feature_dim":       N_IN,
    "emb_dim":           EMB_DIM,
    "n_physics":         N_PHYSICS,
    "n_names":           N_NAMES,
    "pressure_log_mean": G_PRESSURE_LOG_MEAN,
    "pressure_log_std":  G_PRESSURE_LOG_STD,
    "gsm_log_min":       float(G_GSM_MIN),
    "gsm_log_max":       float(G_GSM_MAX),
    "thickness_log_min": float(G_THICK_MIN),
    "thickness_log_max": float(G_THICK_MAX),
    "density_log_min":   float(G_DENS_MIN),
    "density_log_max":   float(G_DENS_MAX),
    "blade_types_en":    BLADE_TYPES_EN,
    "blade_types_jp":    BLADE_TYPES,
    "name_vocab":        name_to_idx,
    "name_embeddings":   {n: EMB_MATRIX[i].tolist() for n, i in name_to_idx.items()},
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
    json.dump(preprocessor, f, indent=2, ensure_ascii=False, cls=NpEncoder)
prep_kb = os.path.getsize(PREP_JSON) // 1024
print(f"\nSaved preprocessor_v2.json  ({prep_kb} KB)")
print(f"  Names: {N_NAMES}  |  Embedding: {N_NAMES}×{EMB_DIM}")
print(f"  Material lookup entries: {len(material_lookup)}")
print(f"  Category avg embeddings: {len(category_avg_emb)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("RESULTS SUMMARY")
print("═"*70)
print(f"{'Machine':<18} {'N':>6} {'Ep':>5} {'P_MAE':>7} {'P_MAPE':>8} {'Blade':>7} {'MC':>7}")
print("-"*70)
for mname, r in results.items():
    print(f"{mname:<18} {r['n_train']+r['n_val']:>6} {r['best_epoch']:>5} "
          f"{r['mae']:>7.1f} {r['mape']:>7.1f}% {r['blade_acc']:>7.3f} {r['mc_acc']:>7.3f}")
print("═"*70)
print("\nAll done. Next steps:")
print("  1. Copy preprocessor_v2.json → preprocessor.json")
print("  2. Copy material_predictor_*_v2.onnx → remove _v2 suffix")
print("  3. Run: python scripts/build_materials_json.py")
print("  4. Update predict.js + index.html + app.js")
