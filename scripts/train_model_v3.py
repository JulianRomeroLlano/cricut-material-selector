"""
train_model_v3.py — Single Global Model with Online Augmentation

Changes from v2:
  • No pre-augmented CSV — jitter applied online (p=25%) during training only
  • Single global model for ALL machine families (no per-machine fine-tuning)
  • Machine families as features instead of per-machine models:
      Family 0 = Joy       (Cricut Joy, Cricut Joy 2, Cricut Joy Xtra)
      Family 1 = Explore   (Explore 3, Explore 5 — same materials, same family encoding)
      Family 2 = Maker     (Maker 3)
  • Explore 5 added as a supported machine (uses Explore 3 training data)
  • Thickness jitter reduced from ±10% to ±5%

Features (feature_dim = 26):
  name_embedding(16) + gsm_lognorm(1) + thickness_lognorm(1) + is_bonded(1)
  + surface_texture(1) + has_adhesive(1) + density_lognorm(1) + shore_norm(1)
  + family_onehot(3)

Outputs:
  assets/model/material_predictor_global_v3.onnx
  assets/model/preprocessor_v3.json

Run: source venv/bin/activate && python scripts/train_model_v3.py

v3.7 additions:
  • Dataset balancing (--balance, default on): every material name is replicated
    until all names have exactly the same number of rows (= the largest group).
    Only the augmented training loaders are balanced; eval loaders stay as-is.
  • LR schedule: linear warmup (per batch) → cosine decay from --lr-peak down to
    lr_peak × --lr-min-ratio (default 1e-3) over the planned epochs.
  • Resume: --resume [auto|path.pt|path.onnx] continues from a checkpoint with a
    1-epoch warmup (skips the split phase — the checkpoint already saw all data).
    'auto' prefers assets/model/checkpoints/global_v3_best.pt and otherwise
    reconstructs the model from the deployed ONNX + preprocessor_v3.json.
  • Best state is saved to assets/model/checkpoints/global_v3_best.pt.

Example — 20-epoch balanced continuation from the best checkpoint:
  python scripts/train_model_v3.py --resume auto --epochs 20
"""
import os, re, json, copy, math, random, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.join(os.path.dirname(__file__), "..")
DATA_CSV  = os.path.join(ROOT, "assets", "data", "Material List (Combined).csv")
MODEL_DIR = os.path.join(ROOT, "assets", "model")
PP_V2     = os.path.join(MODEL_DIR, "preprocessor_v2.json")  # for embedding warm-start
OUT_ONNX  = os.path.join(MODEL_DIR, "material_predictor_global_v3.onnx")
OUT_JSON  = os.path.join(MODEL_DIR, "preprocessor_v3.json")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.stdout.reconfigure(line_buffering=True)   # epoch lines reach a redirected log immediately
print(f"Device: {DEVICE}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EMB_DIM    = 16
N_FAMILIES = 5
N_PHYSICS  = 7
FEATURE_DIM = EMB_DIM + N_PHYSICS + N_FAMILIES   # 28

LR          = 1e-3
MAX_EPOCHS  = 3000
PATIENCE    = 300
EVAL_EVERY  = 10
BATCH_SIZE  = 256
LR_MIN_RATIO = 1e-3      # cosine floor = lr_peak × LR_MIN_RATIO

CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
CKPT_BEST  = os.path.join(CKPT_DIR, "global_v3_best.pt")
CKPT_BASE  = os.path.join(CKPT_DIR, "global_v3.6_baseline.pt")   # frozen copy of the pre-resume model

ap = argparse.ArgumentParser(description="Train / continue the v3 global model")
ap.add_argument("--resume", nargs="?", const="auto", default=None,
                help="continue from a checkpoint: 'auto', a .pt file or an .onnx file")
ap.add_argument("--epochs", type=int, default=None,
                help="planned epochs (default: MAX_EPOCHS; cosine decay spans this)")
ap.add_argument("--patience", type=int, default=None, help="early-stop patience in epochs")
ap.add_argument("--lr-peak", type=float, default=LR, help="peak learning rate")
ap.add_argument("--lr-min-ratio", type=float, default=LR_MIN_RATIO,
                help="cosine floor as a fraction of lr-peak")
ap.add_argument("--warmup-epochs", type=float, default=None,
                help="linear warmup length in epochs (default: 1 when resuming, else 0)")
ap.add_argument("--balance", dest="balance", action="store_true", default=True,
                help="replicate rows so every material name has the same count (default)")
ap.add_argument("--no-balance", dest="balance", action="store_false")
ap.add_argument("--eval-every", type=int, default=None,
                help="evaluate every N epochs (default: 1 when resuming, else EVAL_EVERY)")
ARGS = ap.parse_args()

RESUME        = ARGS.resume
RUN_EPOCHS    = ARGS.epochs if ARGS.epochs else MAX_EPOCHS
LR_PEAK       = ARGS.lr_peak
LR_MIN_RATIO  = ARGS.lr_min_ratio
WARMUP_EPOCHS = ARGS.warmup_epochs if ARGS.warmup_epochs is not None else (1.0 if RESUME else 0.0)
RUN_EVAL_EVERY = ARGS.eval_every if ARGS.eval_every else (1 if RESUME else EVAL_EVERY)
BALANCE       = ARGS.balance
print(f"Run config: resume={RESUME}  epochs={RUN_EPOCHS}  lr_peak={LR_PEAK:g}  "
      f"lr_min={LR_PEAK*LR_MIN_RATIO:g}  warmup={WARMUP_EPOCHS}ep  balance={BALANCE}  "
      f"eval_every={RUN_EVAL_EVERY}")

N_BLADE  = 5
N_MCUT   = 6
W_PRESSURE, W_BLADE, W_MCUT = 0.40, 0.40, 0.20

AUG_PROB     = 0.25
AUG_JITTER   = 0.05   # ±5%

MACHINE_FAMILIES = {
    "Cricut Joy":      0,
    "Cricut Joy 2":    1,
    "Cricut Joy Xtra": 2,
    "Explore 3":       3,
    "Explore 5":       3,   # same materials as Explore 3
    "Maker 3":         4,
}
FAMILY_NAMES = {0: "Joy", 1: "Joy2", 2: "JoyXtra", 3: "Explore", 4: "Maker"}

BLADE_JP_TO_EN = {
    "ディープポイントブレード":      "Deep-Point Blade",
    "ナイフの刃":                    "Knife Blade",
    "ファインポイントブレード":       "Fine-Point Blade",
    "ボンデッドファブリックブレード":  "Bonded Fabric Blade",
    "ロータリーブレード":             "Rotary Blade",
}
TEXTURE_MAP = {
    "plain": 0.0, "matte": 0.1, "glossy": 0.15, "satin": 0.1,
    "shimmer": 0.2, "pearl": 0.15, "holographic": 0.25, "iridescent": 0.25,
    "embossed": 0.2, "woven": 0.3, "rough": 0.35, "smooth": 0.05,
}
MULTICUT_MAP = {
    "-": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "2倍": 2, "3倍": 3, "4倍": 4, "5倍": 5, "6倍": 6, "7倍": 7, "8倍": 8,
    "10倍": 10, "12倍": 12, "14倍": 14, "16倍": 16, "17倍": 17, "18倍": 18, "24倍": 24,
}
THICKNESS_DEFAULTS = {
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10, "Vinyl": 0.08,
    "Smart Materials": 0.10, "Printable Materials": 0.12, "Infusible Ink": 0.10,
    "Board/Cardboard": 1.0, "Leather": 1.6, "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Name Normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    # v3.1: keep the full name. Measurement parentheticals are NOT stripped —
    # variants like "Touring Leather (2-3 oz. / 0.8 mm)" vs "(6-7 oz. / 2.4 mm)"
    # are distinct materials with different pressure/blade/multi-cut and must get
    # their own embedding and physics lookup entry.
    return name.strip()

def _lb_to_mm(lb):
    pts = [(60, 0.15), (65, 0.18), (80, 0.22), (100, 0.27), (140, 0.38)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        if x0 <= lb <= x1:
            return y0 + (y1-y0) * (lb-x0) / (x1-x0)
    return 0.27 if lb > 100 else 0.15

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
    if n == 0:       return 0
    if n == 2:       return 1
    if n == 3:       return 2
    if 4 <= n <= 5:  return 3
    if 6 <= n <= 8:  return 4
    return 5

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load CSV + Add Explore 5
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
df = df[df["Cutting Pressure"].notna()].copy()
df = df[df["Category"] != "Pens & Markers"].copy()
# Smart materials are Cricut-only products that ship with factory-preset
# cutting settings — they are never AI-predicted, so they are excluded from
# training entirely. They stay in materials.json for the browser list.
smart_mask = df["Material Name (EN)"].str.strip().str.startswith("Smart")
print(f"Excluding {smart_mask.sum()} Smart material rows from training")
df = df[~smart_mask].copy()
df["Cutting Pressure"] = df["Cutting Pressure"].astype(float)
df["Blade Type"] = df["Blade Type"].replace("ファインポイント", "ファインポイントブレード")

# Add Explore 5 rows (same materials as Explore 3, same family encoding)
explore3 = df[df["Machine"] == "Explore 3"].copy()
explore5 = explore3.copy(); explore5["Machine"] = "Explore 5"
df = pd.concat([df, explore5], ignore_index=True)
print(f"Loaded {len(df)} rows  (incl. {len(explore5)} Explore 5 duplicates of Explore 3)")

# ─── derive physics columns ──────────────────────────────────────────────────
df["base_name"]  = df["Material Name (EN)"].apply(normalize_name)
df["thickness"]  = df.apply(
    lambda r: float(r["thickness_mm"]) if ("thickness_mm" in r.index and pd.notna(r["thickness_mm"]))
              else infer_thickness(r["Material Name (EN)"], r["Category"]),
    axis=1
)
df["mc_bucket"]  = df["Multi-Cut"].apply(bucket_multicut)
df["is_bonded"]  = (
    df["Material Name (EN)"].str.contains("Bonded", case=False, na=False)
    | (df["Blade Type"] == "ボンデッドファブリックブレード")
).astype(float)
df["texture"]    = df["Surface Texture"].map(TEXTURE_MAP).fillna(0.0) \
                    if "Surface Texture" in df.columns else 0.0
df["adhesive"]   = df["Has Adhesive"].astype(float) if "Has Adhesive" in df.columns else 0.0
df["gsm"]        = df["GSM"].astype(float)
df["density"]    = df["Density (kg/m3)"].astype(float) if "Density (kg/m3)" in df.columns else 500.0
df["shore"]      = df["Shore Hardness A"].astype(float) if "Shore Hardness A" in df.columns else 40.0
df["family"]     = df["Machine"].map(MACHINE_FAMILIES)
df["blade_en"]   = df["Blade Type"].map(BLADE_JP_TO_EN).fillna(df["Blade Type"])

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Vocabulary, Normalization Constants, Group Split
# ═══════════════════════════════════════════════════════════════════════════════

BLADE_TYPES_EN = sorted(df["blade_en"].unique().tolist())
BLADE_TYPES_JP = [next(jp for jp, en in BLADE_JP_TO_EN.items() if en == e) for e in BLADE_TYPES_EN]
blade_to_idx   = {b: i for i, b in enumerate(BLADE_TYPES_EN)}

all_base_names = sorted(df["base_name"].unique().tolist())
name_to_idx    = {n: i for i, n in enumerate(all_base_names)}
N_NAMES        = len(all_base_names)
df["name_idx"] = df["base_name"].map(name_to_idx).astype(int)

print(f"Unique base names: {N_NAMES}")
print(f"Blade types: {BLADE_TYPES_EN}")

# Normalization over original data (original machines only, no Explore 5 duplicates)
df_orig = df[df["Machine"] != "Explore 5"]
gsm_log   = np.log1p(df_orig["gsm"].values)
thick_log = np.log1p(df_orig["thickness"].values)
dens_log  = np.log1p(df_orig["density"].values)
plog      = np.log(df_orig["Cutting Pressure"].values)

G_GSM_MIN   = float(gsm_log.min());   G_GSM_MAX   = float(gsm_log.max())
G_THICK_MIN = float(thick_log.min()); G_THICK_MAX = float(thick_log.max())
G_DENS_MIN  = float(dens_log.min());  G_DENS_MAX  = float(dens_log.max())
G_P_MEAN    = float(plog.mean());     G_P_STD     = float(plog.std())
print(f"Global pressure: log_mean={G_P_MEAN:.4f}  log_std={G_P_STD:.4f}")

# Group split by base name — same material stays in same split for ALL machines
unique_names = np.array(sorted(df["base_name"].unique()))
tr_names, va_names = train_test_split(unique_names, test_size=0.10, random_state=SEED)
tr_mask = df["base_name"].isin(tr_names)
va_mask = df["base_name"].isin(va_names)
print(f"Group split: {tr_mask.sum()} train rows ({len(tr_names)} names) | "
      f"{va_mask.sum()} val rows ({len(va_names)} names)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dataset with Online Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

NORM = {
    "gsm_log_min": G_GSM_MIN,   "gsm_log_range": G_GSM_MAX - G_GSM_MIN,
    "thick_log_min": G_THICK_MIN, "thick_log_range": G_THICK_MAX - G_THICK_MIN,
    "dens_log_min": G_DENS_MIN,  "dens_log_range": G_DENS_MAX - G_DENS_MIN,
    "p_log_mean": G_P_MEAN,     "p_log_std": G_P_STD,
}

def clamp01(v): return max(0.0, min(1.0, v))

def _make_records(sub_df):
    records = []
    for _, r in sub_df.iterrows():
        records.append({
            "name_idx":  int(r["name_idx"]),
            "gsm":       float(r["gsm"]),
            "thickness": float(r["thickness"]),
            "density":   float(r["density"]),
            "shore":     float(r["shore"]),
            "bonded":    float(r["is_bonded"]),
            "texture":   float(r["texture"]),
            "adhesive":  float(r["adhesive"]),
            "family":    int(r["family"]),
            "p_log":     float(math.log(r["Cutting Pressure"])),
            "blade_idx": int(blade_to_idx[r["blade_en"]]),
            "mc_bucket": int(r["mc_bucket"]),
        })
    return records

tr_records = _make_records(df[tr_mask])
va_records = _make_records(df[va_mask])

class MaterialDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r       = self.records[idx]
        gsm     = r["gsm"]
        thick   = r["thickness"]
        density = r["density"]
        shore   = r["shore"]
        p_log   = r["p_log"]

        if self.augment and random.random() < AUG_PROB:
            scale   = random.uniform(1 - AUG_JITTER, 1 + AUG_JITTER)
            gsm     = gsm * scale
            p_log   = p_log + math.log(scale)     # pressure tracks GSM linearly
            thick   = max(0.01, thick * random.uniform(1 - AUG_JITTER, 1 + AUG_JITTER))
            density = max(1.0,  density * random.uniform(1 - AUG_JITTER, 1 + AUG_JITTER))
            shore   = max(0.0,  shore * random.uniform(1 - AUG_JITTER, 1 + AUG_JITTER))

        n   = NORM
        g_n = clamp01((math.log1p(gsm)     - n["gsm_log_min"])   / (n["gsm_log_range"]   + 1e-9))
        t_n = clamp01((math.log1p(thick)   - n["thick_log_min"]) / (n["thick_log_range"] + 1e-9))
        d_n = clamp01((math.log1p(density) - n["dens_log_min"])  / (n["dens_log_range"]  + 1e-9))
        s_n = clamp01(shore / 100.0)

        physics   = [g_n, t_n, r["bonded"], r["texture"], r["adhesive"], d_n, s_n]
        family_oh = [0.0] * N_FAMILIES; family_oh[r["family"]] = 1.0
        p_norm    = (p_log - n["p_log_mean"]) / (n["p_log_std"] + 1e-9)

        return (
            torch.tensor(r["name_idx"], dtype=torch.long),
            torch.tensor(physics,    dtype=torch.float32),
            torch.tensor(family_oh,  dtype=torch.float32),
            torch.tensor(p_norm,     dtype=torch.float32),
            torch.tensor(r["blade_idx"], dtype=torch.long),
            torch.tensor(r["mc_bucket"], dtype=torch.long),
        )

def balance_records(records, key="name_idx"):
    """Replicate rows so every group (default: material name) has exactly the
    same number of rows as the largest group. Rows are cycled deterministically
    within each group; online augmentation supplies the variability."""
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    target = max(len(g) for g in groups.values())
    out = []
    for g in groups.values():
        out.extend(g[i % len(g)] for i in range(target))
    return out, target

def make_train_records(records, label):
    if not BALANCE:
        return records
    bal, target = balance_records(records)
    n_groups = len({r["name_idx"] for r in records})
    print(f"  [{label}] balanced: {len(records)} rows → {len(bal)} rows "
          f"({n_groups} names × {target} each)")
    return bal

tr_ds = MaterialDataset(make_train_records(tr_records, "split"), augment=True)
va_ds = MaterialDataset(va_records, augment=False)
tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
va_ld = DataLoader(va_ds, batch_size=512,        shuffle=False, num_workers=0)
print(f"DataLoaders: {len(tr_ds)} train | {len(va_ds)} val")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalModel(nn.Module):
    def __init__(self, n_names, emb_dim, feature_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_names, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        # backbone (without embedding — this is what gets exported to ONNX)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128),         nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(128, 64),          nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.15),
        )
        self.head_p  = nn.Linear(64, 1)
        self.head_b  = nn.Linear(64, N_BLADE)
        self.head_mc = nn.Linear(64, N_MCUT)
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  Architecture: {feature_dim}→256→128→64→heads  ({n_p:,} params)")

    def forward(self, name_idx, physics, family_oh):
        emb  = self.embedding(name_idx)
        x    = torch.cat([emb, physics, family_oh], dim=1)
        feat = self.backbone(x)
        return self.head_p(feat).squeeze(1), self.head_b(feat), self.head_mc(feat)


class ExportModel(nn.Module):
    """Wraps backbone+heads for ONNX export — takes pre-computed 26-dim features."""
    def __init__(self, src: GlobalModel):
        super().__init__()
        self.backbone = src.backbone
        self.head_p   = src.head_p
        self.head_b   = src.head_b
        self.head_mc  = src.head_mc

    def forward(self, features):
        feat = self.backbone(features)
        return self.head_p(feat).squeeze(1), self.head_b(feat), self.head_mc(feat)


def loss_fn(p_pred, b_pred, mc_pred, p_true, b_true, mc_true):
    lp  = nn.functional.mse_loss(p_pred, p_true)
    lb  = nn.functional.cross_entropy(b_pred, b_true)
    lmc = nn.functional.cross_entropy(mc_pred, mc_true)
    return W_PRESSURE * lp + W_BLADE * lb + W_MCUT * lmc


def warm_start_embeddings(m):
    """Initialise name embeddings from v2 preprocessor where names match."""
    if not os.path.exists(PP_V2):
        return
    pp_v2 = json.load(open(PP_V2))
    emb_v2 = pp_v2.get("name_embeddings", {})
    n_init = 0
    with torch.no_grad():
        for name, idx in name_to_idx.items():
            if name in emb_v2:
                m.embedding.weight[idx] = torch.tensor(emb_v2[name], dtype=torch.float32)
                n_init += 1
    print(f"  Warm-started {n_init}/{N_NAMES} embeddings from preprocessor_v2.json")


def make_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs, min_ratio):
    """Per-batch LR schedule: linear warmup from ~0 to peak over `warmup_epochs`,
    then cosine decay to peak × min_ratio at the end of the planned run."""
    total = max(1, epochs * steps_per_epoch)
    warm  = int(round(warmup_epochs * steps_per_epoch))
    def factor(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def eval_loss(m, loader):
    m.eval()
    vl = 0.0
    with torch.no_grad():
        for nidx, ph, foh, pt, bt, mt in loader:
            nidx, ph, foh = nidx.to(DEVICE), ph.to(DEVICE), foh.to(DEVICE)
            pt, bt, mt    = pt.to(DEVICE), bt.to(DEVICE), mt.to(DEVICE)
            pp_, pb_, pm_ = m(nidx, ph, foh)
            vl += loss_fn(pp_, pb_, pm_, pt, bt, mt).item()
    return vl / len(loader)


def save_checkpoint(m, path, epoch, loss, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
        "name_vocab": all_base_names,
        "epoch": epoch, "eval_loss": loss, "note": note,
        "norm": NORM, "feature_dim": FEATURE_DIM, "emb_dim": EMB_DIM,
    }, path)
    print(f"  Checkpoint saved → {os.path.relpath(path, ROOT)}  (epoch {epoch}, loss {loss:.5f})")


def _load_embeddings_by_name(m, name_to_vec, category_avg=None):
    """Copy embeddings by material name; new names fall back to the category
    average embedding (if given) or keep their random init."""
    n_hit, n_cat = 0, 0
    with torch.no_grad():
        for name, idx in name_to_idx.items():
            if name in name_to_vec:
                m.embedding.weight[idx] = torch.tensor(name_to_vec[name], dtype=torch.float32)
                n_hit += 1
            elif category_avg is not None:
                cat = df[df["base_name"] == name]["Category"].iloc[0]
                if cat in category_avg:
                    m.embedding.weight[idx] = torch.tensor(category_avg[cat], dtype=torch.float32)
                    n_cat += 1
    print(f"  Embeddings: {n_hit}/{N_NAMES} restored by name, {n_cat} from category average, "
          f"{N_NAMES - n_hit - n_cat} random")


def load_checkpoint(m, spec):
    """Restore weights into `m` from 'auto', a .pt file or an .onnx file (the
    ONNX path also needs preprocessor_v3.json for the name embeddings)."""
    if spec == "auto":
        spec = CKPT_BEST if os.path.exists(CKPT_BEST) else OUT_ONNX
    print(f"\nResuming from {os.path.relpath(spec, ROOT)}")

    if spec.endswith(".pt"):
        ck = torch.load(spec, map_location="cpu", weights_only=True)
        sd = ck["state_dict"]
        vocab = ck.get("name_vocab")
        if vocab == all_base_names:
            m.load_state_dict(sd)
            print(f"  Full state restored (epoch {ck.get('epoch')}, loss {ck.get('eval_loss')})")
        else:
            # vocab changed: restore backbone/heads, then embeddings by name
            backbone_sd = {k: v for k, v in sd.items() if not k.startswith("embedding.")}
            m.load_state_dict(backbone_sd, strict=False)
            emb = sd["embedding.weight"]
            _load_embeddings_by_name(m, {n: emb[i].tolist() for i, n in enumerate(vocab or [])})
        return spec

    # ONNX (backbone + heads) + preprocessor JSON (embeddings)
    import onnx
    from onnx import numpy_helper
    g = onnx.load(spec).graph
    sd = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy()) for init in g.initializer}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("embedding.")]
    assert not missing and not unexpected, (missing, unexpected)
    pp_path = OUT_JSON if os.path.exists(OUT_JSON) else os.path.join(MODEL_DIR, "preprocessor.json")
    pp = json.load(open(pp_path, encoding="utf-8"))
    _load_embeddings_by_name(m, pp.get("name_embeddings", {}), pp.get("category_avg_embeddings"))
    for k in ("pressure_log_mean", "pressure_log_std"):
        if abs(pp[k] - NORM["p_log_mean" if "mean" in k else "p_log_std"]) > 1e-6:
            print(f"  WARNING: {k} differs from checkpoint ({pp[k]:.6f} vs current); "
                  f"pressure targets are re-normalised with current stats")

    # sanity: reconstructed export must reproduce the ONNX model
    import onnxruntime as ort_rt
    sess = ort_rt.InferenceSession(spec)
    x = np.random.RandomState(0).randn(64, FEATURE_DIM).astype(np.float32)
    ref = sess.run(None, {"features": x})
    exp = ExportModel(m).to(DEVICE).eval()
    with torch.no_grad():
        out = [o.cpu().numpy() for o in exp(torch.from_numpy(x).to(DEVICE))]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, out))
    print(f"  ONNX reconstruction check: max |Δ| = {diff:.2e}  {'✓' if diff < 1e-4 else '✗'}")
    assert diff < 1e-4, "reconstructed model does not match ONNX"
    return spec


def train_loop(m, train_loader, eval_loader, label, patience=PATIENCE,
               epochs=None, eval_every=None, warmup_epochs=0.0,
               lr_peak=None, include_start=False):
    """Train with warmup + cosine LR and early stopping on eval_loader loss;
    returns best epoch. With include_start=True the untouched starting weights
    compete as 'epoch 0' so a continuation can never end worse than it began."""
    epochs     = epochs or RUN_EPOCHS
    eval_every = eval_every or RUN_EVAL_EVERY
    lr_peak    = lr_peak or LR_PEAK
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr_peak, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, len(train_loader), epochs, warmup_epochs, LR_MIN_RATIO)
    best_val, best_ep, no_imp, best_state = float("inf"), 0, 0, None

    if include_start:
        best_val   = eval_loss(m, eval_loader)
        best_state = copy.deepcopy(m.state_dict())
        print(f"  [{label}] ep    0  eval_loss={best_val:.5f}  (starting checkpoint)")

    print(f"\n[{label}] Training up to {epochs} epochs (patience={patience}, "
          f"{len(train_loader)} batches/epoch, lr {lr_peak:g}→{lr_peak*LR_MIN_RATIO:g}, "
          f"warmup {warmup_epochs} ep)…")
    for epoch in range(1, epochs + 1):
        m.train()
        for nidx, ph, foh, pt, bt, mt in train_loader:
            nidx, ph, foh = nidx.to(DEVICE), ph.to(DEVICE), foh.to(DEVICE)
            pt, bt, mt    = pt.to(DEVICE), bt.to(DEVICE), mt.to(DEVICE)
            optimizer.zero_grad()
            pp_, pb_, pm_ = m(nidx, ph, foh)
            loss_fn(pp_, pb_, pm_, pt, bt, mt).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if epoch % eval_every != 0:
            continue

        vl = eval_loss(m, eval_loader)
        cur_lr = optimizer.param_groups[0]["lr"]

        if vl < best_val:
            best_val, best_ep, no_imp = vl, epoch, 0
            best_state = copy.deepcopy(m.state_dict())
        else:
            no_imp += eval_every

        if epochs <= 100 or epoch % 200 == 0:
            print(f"  [{label}] ep {epoch:4d}  eval_loss={vl:.5f}  lr={cur_lr:.2e}"
                  f"{'  *' if best_ep == epoch else ''}")

        if no_imp >= patience:
            print(f"  [{label}] Early stop at epoch {epoch}  (best ep {best_ep}, loss {best_val:.5f})")
            break

    m.load_state_dict(best_state)
    m.eval()
    print(f"  [{label}] best epoch {best_ep}  eval_loss={best_val:.5f}")
    return best_ep, best_val


model = GlobalModel(N_NAMES, EMB_DIM, FEATURE_DIM).to(DEVICE)
if not RESUME:
    warm_start_embeddings(model)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Training
# ═══════════════════════════════════════════════════════════════════════════════

if RESUME:
    # A resumed checkpoint was fitted on ALL rows, so a group-split validation
    # would be leaky and meaningless — skip straight to the continuation fit.
    print("\n[split] skipped — resuming from a checkpoint that already saw all data")
    best_ep = None
else:
    best_ep, _ = train_loop(model, tr_ld, va_ld, "split", patience=ARGS.patience or PATIENCE,
                            warmup_epochs=WARMUP_EPOCHS)
    print(f"Validation run complete — best epoch {best_ep}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Final Validation Metrics (per machine family + overall)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_subset(records_sub):
    model.eval()   # dropout off — a freshly loaded checkpoint is still in train mode
    ds = MaterialDataset(records_sub, augment=False)
    ld = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    p_preds, b_preds, mc_preds = [], [], []
    p_trues, b_trues, mc_trues = [], [], []
    with torch.no_grad():
        for nidx, ph, foh, pt, bt, mt in ld:
            nidx, ph, foh = nidx.to(DEVICE), ph.to(DEVICE), foh.to(DEVICE)
            pp_, pb_, pm_ = model(nidx, ph, foh)
            p_preds.extend(pp_.cpu().numpy())
            b_preds.extend(pb_.argmax(1).cpu().numpy())
            mc_preds.extend(pm_.argmax(1).cpu().numpy())
            p_trues.extend(pt.numpy())
            b_trues.extend(bt.numpy())
            mc_trues.extend(mt.numpy())
    p_dec  = np.exp(np.array(p_preds) * G_P_STD + G_P_MEAN)
    pt_dec = np.exp(np.array(p_trues) * G_P_STD + G_P_MEAN)
    mae  = mean_absolute_error(pt_dec, p_dec)
    mape = np.mean(np.abs(p_dec - pt_dec) / (pt_dec + 1e-9)) * 100
    bacc = accuracy_score(b_trues, b_preds)
    bf1  = f1_score(b_trues, b_preds, average="weighted", zero_division=0)
    mcac = accuracy_score(mc_trues, mc_preds)
    return {"n": len(records_sub), "mae": mae, "mape": mape,
            "blade_acc": bacc, "blade_f1": bf1, "mc_acc": mcac}

results_by_family, r_all = {}, None
if not RESUME:
    print("\n── Validation metrics by family ─────────────────────────────────────────")
    # Exclude Explore 5 from val report (identical to Explore 3 with same family encoding)
    va_df_no5 = df[va_mask & (df["Machine"] != "Explore 5")]
    for fam_idx, fam_name in FAMILY_NAMES.items():
        sub_df   = va_df_no5[va_df_no5["family"] == fam_idx]
        if sub_df.empty: continue
        sub_recs = _make_records(sub_df)
        r = eval_subset(sub_recs)
        results_by_family[fam_name] = r
        print(f"  {fam_name:<10}  N={r['n']:4d}  MAE={r['mae']:6.1f}  MAPE={r['mape']:5.1f}%  "
              f"Blade={r['blade_acc']:.3f}  MC={r['mc_acc']:.3f}")

    all_va_recs = _make_records(va_df_no5)
    r_all = eval_subset(all_va_recs)
    print(f"  {'ALL':<10}  N={r_all['n']:4d}  MAE={r_all['mae']:6.1f}  MAPE={r_all['mape']:5.1f}%  "
          f"Blade={r_all['blade_acc']:.3f}  MC={r_all['mc_acc']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.5 Final Fit — retrain on 100% of the data
# ═══════════════════════════════════════════════════════════════════════════════
# The split run above validated that the architecture generalizes. The deployed
# model is a lookup-style predictor for known materials, so the final model is
# trained on ALL rows (train+val). Early stopping monitors loss on the full
# un-augmented data — i.e. it stops when the data is fit, not when a holdout
# starts degrading.

print("\n── Final fit on all data ───────────────────────────────────────────────")
all_records = tr_records + va_records
full_tr_ld = DataLoader(MaterialDataset(make_train_records(all_records, "final"), augment=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
full_ev_ld = DataLoader(MaterialDataset(all_records, augment=False),
                        batch_size=512, shuffle=False, num_workers=0)

df_no5 = df[df["Machine"] != "Explore 5"]
final_model = GlobalModel(N_NAMES, EMB_DIM, FEATURE_DIM).to(DEVICE)
r_mem_before = None
if RESUME:
    src = load_checkpoint(final_model, RESUME)
    if src.endswith(".onnx") and not os.path.exists(CKPT_BASE):
        # freeze the pre-continuation model as a .pt so it is never lost when
        # the ONNX below is overwritten
        model = final_model
        save_checkpoint(final_model, CKPT_BASE, 0, eval_loss(final_model, full_ev_ld),
                        note=f"reconstructed from {os.path.basename(src)} before continuation")
    model = final_model
    r_mem_before = eval_subset(_make_records(df_no5))
    print(f"  Memorization BEFORE (all {r_mem_before['n']} rows): MAE={r_mem_before['mae']:.1f}  "
          f"MAPE={r_mem_before['mape']:.1f}%  Blade={r_mem_before['blade_acc']:.3f}  "
          f"MC={r_mem_before['mc_acc']:.3f}")
else:
    warm_start_embeddings(final_model)
# Larger patience for the final fit: the goal is memorization, and the loss
# plateaus for long stretches before fitting the extreme-pressure outliers.
final_ep, final_loss = train_loop(final_model, full_tr_ld, full_ev_ld, "final",
                                  patience=ARGS.patience or 600,
                                  warmup_epochs=WARMUP_EPOCHS, include_start=bool(RESUME))
print(f"Final fit complete — best epoch {final_ep}")
save_checkpoint(final_model, CKPT_BEST, final_ep, final_loss,
                note=f"resume={RESUME} epochs={RUN_EPOCHS} lr_peak={LR_PEAK} balance={BALANCE}")

model = final_model   # sections below export & evaluate the final model

# Memorization check: the final model over every original row (no Explore 5 dup)
r_mem = eval_subset(_make_records(df_no5))
print(f"  Memorization (all {r_mem['n']} rows): MAE={r_mem['mae']:.1f}  "
      f"MAPE={r_mem['mape']:.1f}%  Blade={r_mem['blade_acc']:.3f}  MC={r_mem['mc_acc']:.3f}")
if r_mem_before:
    print(f"  Δ vs checkpoint: MAE {r_mem_before['mae']:.1f}→{r_mem['mae']:.1f}  "
          f"MAPE {r_mem_before['mape']:.1f}%→{r_mem['mape']:.1f}%  "
          f"Blade {r_mem_before['blade_acc']:.3f}→{r_mem['blade_acc']:.3f}  "
          f"MC {r_mem_before['mc_acc']:.3f}→{r_mem['mc_acc']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ONNX Export (backbone only — 26-dim input)
# ═══════════════════════════════════════════════════════════════════════════════

export_model = ExportModel(model).to(DEVICE).eval()
dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32).to(DEVICE)
torch.onnx.export(
    export_model, dummy, OUT_ONNX,
    opset_version=12,
    input_names=["features"],
    output_names=["pressure_norm", "blade_logits", "multicut_logits"],
    dynamic_axes={"features":         {0: "batch"},
                  "pressure_norm":    {0: "batch"},
                  "blade_logits":     {0: "batch"},
                  "multicut_logits":  {0: "batch"}},
)
import onnxruntime as ort_rt
sess = ort_rt.InferenceSession(OUT_ONNX)
out  = sess.run(None, {"features": np.zeros((1, FEATURE_DIM), dtype=np.float32)})
size_kb = os.path.getsize(OUT_ONNX) // 1024
print(f"\nONNX → {os.path.basename(OUT_ONNX)}  ({size_kb} KB)  input={sess.get_inputs()[0].shape}  ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Extract Embeddings + Material Lookup
# ═══════════════════════════════════════════════════════════════════════════════

EMB_MATRIX = model.embedding.weight.detach().cpu().numpy()   # (N_NAMES, EMB_DIM)

# Material lookup: representative physics per normalized name (original data only)
df_base = df[df["Machine"] != "Explore 5"].copy()
material_lookup = {}
for base_name in all_base_names:
    rows = df_base[df_base["base_name"] == base_name]
    if rows.empty: continue
    r0 = rows.iloc[0]
    material_lookup[base_name] = {
        "category":     r0["Category"],
        "gsm":          round(float(rows["gsm"].median()), 2),
        "density":      round(float(rows["density"].median()), 2),
        "shore":        round(float(rows["shore"].median()), 2),
        "texture":      float(r0["texture"]),
        "has_adhesive": float(r0["adhesive"]),
        "is_bonded":    float(r0["is_bonded"]),
        "thickness_mm": round(float(rows["thickness"].median()), 4),
    }

# Category average embeddings (fallback for unknown names)
category_avg_emb = {}
for cat in df_base["Category"].unique():
    idxs = df_base[df_base["Category"] == cat]["name_idx"].values.astype(int)
    category_avg_emb[cat] = EMB_MATRIX[idxs].mean(axis=0).tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Save preprocessor_v3.json
# ═══════════════════════════════════════════════════════════════════════════════

COMPATIBLE_BLADES = {
    "Cricut Joy":      ["Fine-Point Blade"],
    "Cricut Joy 2":    ["Fine-Point Blade"],
    "Cricut Joy Xtra": ["Fine-Point Blade", "Deep-Point Blade"],
    "Explore 3":       ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade"],
    "Maker 3":         ["Fine-Point Blade", "Deep-Point Blade", "Rotary Blade",
                        "Bonded Fabric Blade", "Knife Blade"],
    "Explore 5":       ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade"],
}

machines_meta = {}
for mname, fam in MACHINE_FAMILIES.items():
    machines_meta[mname] = {"slug": "global_v3", "family": fam}

preprocessor = {
    "version":            "v3",
    "feature_dim":        FEATURE_DIM,
    "emb_dim":            EMB_DIM,
    "n_physics":          N_PHYSICS,
    "n_families":         N_FAMILIES,
    "n_names":            N_NAMES,
    "pressure_log_mean":  G_P_MEAN,
    "pressure_log_std":   G_P_STD,
    "gsm_log_min":        G_GSM_MIN,
    "gsm_log_max":        G_GSM_MAX,
    "thickness_log_min":  G_THICK_MIN,
    "thickness_log_max":  G_THICK_MAX,
    "density_log_min":    G_DENS_MIN,
    "density_log_max":    G_DENS_MAX,
    "blade_types_en":     BLADE_TYPES_EN,
    "blade_types_jp":     BLADE_TYPES_JP,
    "name_vocab":         name_to_idx,
    "name_embeddings":    {n: EMB_MATRIX[i].tolist() for n, i in name_to_idx.items()},
    "category_avg_embeddings": category_avg_emb,
    "material_lookup":    material_lookup,
    "machines":           machines_meta,
    "notes": {
        "compatible_blades": COMPATIBLE_BLADES,
        "machine_families": {str(k): v for k, v in FAMILY_NAMES.items()},
        "feature_order": [
            "name_emb[0..15]", "gsm_lognorm", "thickness_lognorm",
            "is_bonded", "surface_texture", "has_adhesive",
            "density_lognorm", "shore_norm",
            "family_joy", "family_explore", "family_maker",
        ],
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(preprocessor, f, indent=2, ensure_ascii=False, cls=NpEncoder)
size_kb = os.path.getsize(OUT_JSON) // 1024
print(f"Saved {os.path.basename(OUT_JSON)}  ({size_kb} KB)")
print(f"  Names: {N_NAMES}  |  Embedding: {N_NAMES}×{EMB_DIM}  |  feature_dim: {FEATURE_DIM}")
print(f"  Material lookup: {len(material_lookup)} entries")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
if r_all is not None:
    print("RESULTS SUMMARY  (validation set, original 5 machines, no Explore 5 dup)")
    print("═"*70)
    print(f"  {'Family':<10} {'N':>5}  {'P_MAE':>7} {'P_MAPE':>8} {'Blade':>7} {'MC':>7}")
    print(f"  {'-'*10} {'-'*5}  {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    for fname, r in results_by_family.items():
        print(f"  {fname:<10} {r['n']:>5}  {r['mae']:>7.1f} {r['mape']:>7.1f}% "
              f"{r['blade_acc']:>7.3f} {r['mc_acc']:>7.3f}")
    print(f"  {'ALL':<10} {r_all['n']:>5}  {r_all['mae']:>7.1f} {r_all['mape']:>7.1f}% "
          f"{r_all['blade_acc']:>7.3f} {r_all['mc_acc']:>7.3f}")
else:
    print(f"RESULTS SUMMARY  (continuation from checkpoint, {RUN_EPOCHS} epochs, "
          f"best epoch {final_ep})")
    print("═"*70)
    print(f"  Memorization over {r_mem['n']} rows: MAE={r_mem['mae']:.1f}  MAPE={r_mem['mape']:.1f}%  "
          f"Blade={r_mem['blade_acc']:.3f}  MC={r_mem['mc_acc']:.3f}")
print("═"*70)
print("\nAll done. Next steps:")
print("  1. cp assets/model/preprocessor_v3.json assets/model/preprocessor.json")
print("  2. cp assets/model/material_predictor_global_v3.onnx assets/model/material_predictor_global.onnx")
print("  3. predict.js already updated for v3 feature_dim=26 + family one-hot + Explore 5")
