"""
Step 2 (v4): Sequential transfer learning — one ONNX model per machine.

Training procedure for each target machine:
  1. Pre-train on other machines (largest → smallest) with LR=1e-3
  2. Fine-tune on target machine with LR=2e-4
  → backbone learns universal material patterns; fine-tune calibrates to each machine

Features (19, no machine one-hot):
  Category one-hot (11) + thickness_lognorm (1) + hardness_norm (1)
  + is_bonded_fabric (1) + gsm_lognorm (1) + surface_texture_norm (1)
  + has_adhesive (1) + density_lognorm (1) + shore_norm (1)

Pressure normalization: GLOBAL (all 720 rows) — same decode formula for all ONNX models.
Feature normalization:  GLOBAL — same encode formula for all ONNX models.

Run with:   source venv/bin/activate && python scripts/train_model.py
Outputs:    assets/model/material_predictor_{slug}.onnx  (one per machine)
            assets/model/preprocessor.json
            .claude/research/ml_training_report_2026-06-06.md
"""
import os, re, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.join(os.path.dirname(__file__), "..")
DATA_CSV  = os.path.join(ROOT, "assets", "data", "Material List (Combined).csv")
MODEL_DIR = os.path.join(ROOT, "assets", "model")
REPORT    = os.path.join(ROOT, ".claude", "research", "ml_training_report_2026-06-06.md")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
LR_PRETRAIN       = 1e-3
LR_FINETUNE       = 2e-4
PATIENCE_PRETRAIN = 60    # epochs without improvement → stop pre-training stage
PATIENCE_FINETUNE = 200   # epochs without improvement → stop fine-tuning
MAX_EPOCHS_PRETRAIN = 600
MAX_EPOCHS_FINETUNE = 2000
EVAL_EVERY          = 5

N_BLADE  = 5
N_MCUT   = 6
W_PRESSURE, W_BLADE, W_MCUT = 0.4, 0.4, 0.2
MULTICUT_BUCKET_LABELS = [0, 2, 3, 4, 6, 10]

# Pre-training order: largest dataset first (most diverse → establishes backbone)
PRETRAIN_ORDER = ["Maker 3", "Explore 3", "Cricut Joy Xtra", "Cricut Joy 2", "Cricut Joy"]

# Per-machine architecture: sized to ~n_train/10 params
MACHINE_CONFIGS = {
    "Cricut Joy":      {"slug": "cricut_joy",      "hidden": [24, 12],     "dropout": [0.50, 0.40]},
    "Cricut Joy 2":    {"slug": "cricut_joy2",      "hidden": [24, 12],     "dropout": [0.50, 0.40]},
    "Cricut Joy Xtra": {"slug": "cricut_joy_xtra",  "hidden": [48, 24],     "dropout": [0.45, 0.35]},
    "Explore 3":       {"slug": "explore3",          "hidden": [64, 32],     "dropout": [0.40, 0.30]},
    "Maker 3":         {"slug": "maker3",            "hidden": [96, 48, 24], "dropout": [0.35, 0.30, 0.00]},
}

BLADE_JP_TO_EN = {
    "ディープポイントブレード":      "Deep-Point Blade",
    "ナイフの刃":                "Knife Blade",
    "ファインポイントブレード":       "Fine-Point Blade",
    "ボンデッドファブリックブレード":  "Bonded Fabric Blade",
    "ロータリーブレード":           "Rotary Blade",
}
TEXTURE_MAP = {
    "plain": 0.0, "matte": 0.1, "glossy": 0.15, "satin": 0.1,
    "shimmer": 0.2, "pearl": 0.15, "holographic": 0.25, "iridescent": 0.25,
    "foil": 0.3, "metallic": 0.3, "textured": 0.35,
    "flock": 0.45, "glitter": 0.5,
}
MULTICUT_MAP = {"-": 0, "2倍": 2, "3倍": 3, "4倍": 4, "5倍": 5,
                "6倍": 6, "7倍": 7, "8倍": 8, "10倍": 10,
                "12倍": 12, "14倍": 14, "16倍": 16,
                "17倍": 17, "18倍": 18, "24倍": 24}
THICKNESS_DEFAULTS = {
    "Paper": 0.08, "Cardstock": 0.22, "Iron-On": 0.10,
    "Vinyl": 0.08, "Smart Materials": 0.10, "Printable Materials": 0.12,
    "Infusible Ink": 0.10, "Board/Cardboard": 1.0, "Leather": 1.6,
    "Fabric": 0.50, "Plastic": 0.10, "Others": 2.0,
}
HARDNESS_DEFAULTS = {
    "Paper": 2, "Cardstock": 4, "Iron-On": 3,
    "Vinyl": 3, "Smart Materials": 3, "Printable Materials": 3,
    "Infusible Ink": 3, "Board/Cardboard": 7, "Leather": 6,
    "Fabric": 3, "Plastic": 4, "Others": 5,
}
HARDNESS_KEYWORDS = [
    ("washi",            2), ("tissue",           1), ("vellum",           2),
    ("glitter iron",     5), ("foil iron",         4), ("metallic iron",    4),
    ("flocked iron",     4), ("felt",              3), ("craft foam",       4),
    ("eva foam",         4), ("foam",              4), ("aluminum foil",    3),
    ("wax paper",        2), ("wrapping paper",    2), ("butcher paper",    2),
    ("sublimation",      3), ("transparency",      3), ("acetate",          4),
    ("glitter cardstock", 6), ("kraft cardstock",  5), ("cardstock (h)",    5),
    ("corrugated",       6), ("flat cardboard",    6), ("foil poster",      7),
    ("kraft board",      7), ("chipboard",         7), ("matboard",         9),
    ("wood veneer",      9), ("basswood",         10), ("rubber",           6),
    ("magnetic",         8), ("faux suede",        5), ("faux leather",     5),
    ("leather",          6), ("bonded",            4), ("wool fabric",      3),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & Clean
# ═══════════════════════════════════════════════════════════════════════════════

df_all = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
df_all["Blade Type"] = df_all["Blade Type"].replace("ファインポイント", "ファインポイントブレード")
df_all = df_all[df_all["Category"] != "Pens & Markers"].copy()
df_all = df_all[df_all["Cutting Pressure"].notna()].copy()
df_all["Cutting Pressure"] = df_all["Cutting Pressure"].astype(float)
print(f"Total rows after cleaning: {len(df_all)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_mm(name):
    m = re.search(r"(\d+\.?\d*)\s*mm", name, re.IGNORECASE)
    return float(m.group(1)) if m else None

def _lb_to_mm(lb):
    pts = [(60, 0.15), (65, 0.18), (80, 0.22), (100, 0.27), (140, 0.38)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]
        if x0 <= lb <= x1:
            return y0 + (y1 - y0) * (lb - x0) / (x1 - x0)
    return 0.27 if lb > 100 else 0.15

def infer_thickness(name, category):
    mm = _extract_mm(name)
    if mm: return mm
    m = re.search(r"(\d+)\s*lb", name, re.IGNORECASE)
    if m: return _lb_to_mm(int(m.group(1)))
    m = re.search(r"(\d+)\s*gsm", name, re.IGNORECASE)
    if m: return max(0.04, int(m.group(1)) * 0.001)
    m = re.search(r"(\d+)[-–]?(\d+)?\s*oz", name, re.IGNORECASE)
    if m:
        oz = (float(m.group(1)) + float(m.group(2) or m.group(1))) / 2
        return oz * 0.40
    return THICKNESS_DEFAULTS.get(category, 0.5)

def infer_hardness(name, category):
    h = HARDNESS_DEFAULTS.get(category, 5)
    n = name.lower()
    for kw, val in HARDNESS_KEYWORDS:
        if kw in n: h = val
    return float(h)

def bucket_multicut(val):
    n = MULTICUT_MAP.get(val, 0)
    if n == 0:      return 0
    if n == 2:      return 1
    if n == 3:      return 2
    if 4 <= n <= 5: return 3
    if 6 <= n <= 8: return 4
    return 5

df_all["thickness"]    = df_all.apply(lambda r: infer_thickness(r["Material Name (EN)"], r["Category"]), axis=1)
df_all["hardness"]     = df_all.apply(lambda r: infer_hardness(r["Material Name (EN)"], r["Category"]), axis=1)
df_all["mc_bucket"]    = df_all["Multi-Cut"].map(bucket_multicut)
df_all["is_bonded"]    = (
    df_all["Material Name (EN)"].str.contains("Bonded", case=False, na=False)
    | (df_all["Blade Type"] == "ボンデッドファブリックブレード")
).astype(float)
df_all["surface_mod"]  = df_all["Surface Texture"].map(TEXTURE_MAP).fillna(0.0) if "Surface Texture" in df_all.columns else 0.0
df_all["has_adhesive"] = df_all["Has Adhesive"].astype(float) if "Has Adhesive" in df_all.columns else 0.0
df_all["gsm"]          = df_all["GSM"].astype(float) if "GSM" in df_all.columns else 100.0
df_all["density"]      = df_all["Density (kg/m3)"].astype(float) if "Density (kg/m3)" in df_all.columns else 500.0
df_all["shore"]        = df_all["Shore Hardness A"].astype(float) if "Shore Hardness A" in df_all.columns else 40.0

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Global normalization constants (computed from ALL 720 rows)
#    Using global stats ensures the ONNX input/output interface is identical
#    for every per-machine model — the browser uses one decode formula.
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES  = sorted(df_all["Category"].unique().tolist())
BLADE_TYPES = sorted(df_all["Blade Type"].unique().tolist())
BLADE_TYPES_EN = [BLADE_JP_TO_EN[jp] for jp in BLADE_TYPES]
blade_idx = {b: i for i, b in enumerate(BLADE_TYPES)}

thick_log_all = np.log1p(df_all["thickness"].values)
gsm_log_all   = np.log1p(df_all["gsm"].values)
dens_log_all  = np.log1p(df_all["density"].values)
plog_all      = np.log(df_all["Cutting Pressure"].values)

G_THICK_MIN,   G_THICK_MAX   = thick_log_all.min(), thick_log_all.max()
G_GSM_MIN,     G_GSM_MAX     = gsm_log_all.min(),   gsm_log_all.max()
G_DENS_MIN,    G_DENS_MAX    = dens_log_all.min(),  dens_log_all.max()
G_PRESSURE_LOG_MEAN = float(plog_all.mean())
G_PRESSURE_LOG_STD  = float(plog_all.std())

print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
print(f"Blade types ({len(BLADE_TYPES)}): {BLADE_TYPES_EN}")
print(f"Global pressure: log_mean={G_PRESSURE_LOG_MEAN:.4f}  log_std={G_PRESSURE_LOG_STD:.4f}")

N_IN = len(CATEGORIES) + 8  # 11 + 8 = 19


def build_X(df: pd.DataFrame) -> np.ndarray:
    """Encode feature matrix (N, 19) using global normalization constants."""
    cat_oh = np.zeros((len(df), len(CATEGORIES)), dtype=np.float32)
    cat_i  = {c: i for i, c in enumerate(CATEGORIES)}
    for i, v in enumerate(df["Category"]):
        if v in cat_i: cat_oh[i, cat_i[v]] = 1.0

    t_n = ((np.log1p(df["thickness"].values) - G_THICK_MIN) / (G_THICK_MAX - G_THICK_MIN + 1e-9)).reshape(-1, 1)
    h_n = ((df["hardness"].values - 1.0) / 9.0).reshape(-1, 1)
    b   = df["is_bonded"].values.reshape(-1, 1)
    g_n = ((np.log1p(df["gsm"].values) - G_GSM_MIN) / (G_GSM_MAX - G_GSM_MIN + 1e-9)).reshape(-1, 1)
    s   = (df["surface_mod"].values / 0.5).reshape(-1, 1)
    a   = df["has_adhesive"].values.reshape(-1, 1)
    d_n = ((np.log1p(df["density"].values) - G_DENS_MIN) / (G_DENS_MAX - G_DENS_MIN + 1e-9)).reshape(-1, 1)
    sh  = (df["shore"].values / 100.0).reshape(-1, 1)

    return np.hstack([cat_oh, t_n, h_n, b, g_n, s, a, d_n, sh]).astype(np.float32)


def build_y(df: pd.DataFrame):
    """Encode targets using global pressure normalization."""
    plog   = np.log(df["Cutting Pressure"].values.astype(np.float32))
    y_p    = ((plog - G_PRESSURE_LOG_MEAN) / G_PRESSURE_LOG_STD).astype(np.float32)
    y_b    = df["Blade Type"].map(blade_idx).values.astype(np.int64)
    y_m    = df["mc_bucket"].values.astype(np.int64)
    return y_p, y_b, y_m


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Model
# ═══════════════════════════════════════════════════════════════════════════════

class MaterialMLP(nn.Module):
    def __init__(self, hidden: list[int], dropout: list[float]):
        super().__init__()
        layers, in_dim = [], N_IN
        for width, drop in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_dim = width
        self.shared         = nn.Sequential(*layers)
        self.pressure_head  = nn.Linear(in_dim, 1)
        self.blade_head     = nn.Linear(in_dim, N_BLADE)
        self.mcut_head      = nn.Linear(in_dim, N_MCUT)

    def forward(self, x):
        h = self.shared(x)
        return self.pressure_head(h).squeeze(-1), self.blade_head(h), self.mcut_head(h)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_loss_fns(y_blade_tr, y_mcut_tr):
    """Class-weighted cross-entropy losses for blade and multi-cut."""
    bc = np.bincount(y_blade_tr, minlength=N_BLADE)
    bw = torch.tensor(1.0 / (bc + 1e-6), dtype=torch.float32).to(DEVICE)
    bw = bw / bw.sum() * N_BLADE

    mc = np.bincount(y_mcut_tr, minlength=N_MCUT)
    mw = torch.tensor(1.0 / (mc + 1e-6), dtype=torch.float32).to(DEVICE)
    mw = mw / mw.sum() * N_MCUT

    return nn.MSELoss(), nn.CrossEntropyLoss(weight=bw), nn.CrossEntropyLoss(weight=mw)


def run_train_stage(
    model, X_tr, yp_tr, yb_tr, ym_tr,
    X_val, yp_val, yb_val, ym_val,
    lr: float, max_epochs: int, patience: int,
    weight_decay: float, label: str,
) -> int:
    """
    Train model on (X_tr, y*_tr), evaluate on (X_val, y*_val).
    Returns best epoch reached.
    """
    loss_p, loss_b, loss_m = make_loss_fns(yb_tr, ym_tr)

    bs = min(32, max(8, len(X_tr) // 4))  # batch size: ~4 batches per epoch minimum
    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(yp_tr).float(),
        torch.from_numpy(yb_tr),
        torch.from_numpy(ym_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(yp_val).float(),
        torch.from_numpy(yb_val),
        torch.from_numpy(ym_val),
    )
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=len(X_tr) > bs)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6
    )

    best_val  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yp, yb, ym in train_dl:
            Xb  = Xb.to(DEVICE); yp = yp.to(DEVICE)
            yb  = yb.to(DEVICE); ym = ym.to(DEVICE)
            pp, bl, ml = model(Xb)
            loss = W_PRESSURE * loss_p(pp, yp) + W_BLADE * loss_b(bl, yb) + W_MCUT * loss_m(ml, ym)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for Xb, yp, yb, ym in val_dl:
                    Xb  = Xb.to(DEVICE); yp = yp.to(DEVICE)
                    yb  = yb.to(DEVICE); ym = ym.to(DEVICE)
                    pp, bl, ml = model(Xb)
                    l = W_PRESSURE * loss_p(pp, yp) + W_BLADE * loss_b(bl, yb) + W_MCUT * loss_m(ml, ym)
                    total += l.item() * len(Xb)
            vl = total / len(val_ds)
            scheduler.step(vl)

            if vl < best_val:
                best_val   = vl
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += EVAL_EVERY
                if no_improve >= patience:
                    break

    model.load_state_dict(best_state)
    print(f"    [{label}] best_epoch={best_epoch}  val_loss={best_val:.4f}")
    return best_epoch


def evaluate(model, X, y_pressure, y_blade, y_multicut) -> dict:
    model.eval()
    with torch.no_grad():
        pn, bl, ml = model(torch.from_numpy(X).float().to(DEVICE))
        pn = pn.cpu().numpy()
        bp = bl.argmax(1).cpu().numpy()
        mp = ml.argmax(1).cpu().numpy()
    p_hat  = np.exp(pn * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
    p_true = np.exp(y_pressure * G_PRESSURE_LOG_STD + G_PRESSURE_LOG_MEAN)
    return {
        "pressure_mae":  mean_absolute_error(p_true, p_hat),
        "pressure_mape": float(np.mean(np.abs(p_hat - p_true) / p_true) * 100),
        "blade_acc":     accuracy_score(y_blade, bp),
        "blade_f1":      f1_score(y_blade, bp, average="weighted", zero_division=0),
        "mcut_acc":      accuracy_score(y_multicut, mp),
        "mcut_f1":       f1_score(y_multicut, mp, average="weighted", zero_division=0),
        "p_pred": p_hat,
        "b_pred": bp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Per-machine sequential training
# ═══════════════════════════════════════════════════════════════════════════════

def train_machine(target_machine: str, cfg: dict) -> dict:
    print(f"\n{'='*64}")
    print(f"  Target: {target_machine}  (slug={cfg['slug']})")
    print(f"{'='*64}")

    # ── Target machine val split (held out throughout all phases) ──────────────
    df_target = df_all[df_all["Machine"] == target_machine].copy()
    X_tgt     = build_X(df_target)
    yp_tgt, yb_tgt, ym_tgt = build_y(df_target)

    idx_all = np.arange(len(X_tgt))
    try:
        idx_tr, idx_val = train_test_split(
            idx_all, test_size=0.1, random_state=SEED, stratify=df_target["Category"].values
        )
    except ValueError:
        idx_tr, idx_val = train_test_split(idx_all, test_size=0.1, random_state=SEED)

    X_tgt_tr, X_tgt_val   = X_tgt[idx_tr],   X_tgt[idx_val]
    yp_tgt_tr, yp_tgt_val = yp_tgt[idx_tr],  yp_tgt[idx_val]
    yb_tgt_tr, yb_tgt_val = yb_tgt[idx_tr],  yb_tgt[idx_val]
    ym_tgt_tr, ym_tgt_val = ym_tgt[idx_tr],  ym_tgt[idx_val]
    print(f"  Target rows: {len(idx_tr)} train / {len(idx_val)} val")

    # ── Initialize model ───────────────────────────────────────────────────────
    model    = MaterialMLP(cfg["hidden"], cfg["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {N_IN}→{'→'.join(str(h) for h in cfg['hidden'])}→heads  ({n_params:,} params)")
    wd = 1e-3 if len(idx_tr) < 100 else 5e-4

    # ── Phase 1: Sequential pre-training on other machines ────────────────────
    pretrain_machines = [m for m in PRETRAIN_ORDER if m != target_machine]
    print(f"\n  Pre-training order: {' → '.join(pretrain_machines)}")

    for source_machine in pretrain_machines:
        df_src = df_all[df_all["Machine"] == source_machine].copy()
        X_src  = build_X(df_src)
        yp_src, yb_src, ym_src = build_y(df_src)

        # 90/10 split of source for pre-training early stopping
        src_idx = np.arange(len(X_src))
        try:
            si_tr, si_val = train_test_split(
                src_idx, test_size=0.1, random_state=SEED,
                stratify=df_src["Category"].values
            )
        except ValueError:
            si_tr, si_val = train_test_split(src_idx, test_size=0.1, random_state=SEED)

        run_train_stage(
            model,
            X_src[si_tr],  yp_src[si_tr],  yb_src[si_tr],  ym_src[si_tr],
            X_src[si_val], yp_src[si_val], yb_src[si_val], ym_src[si_val],
            lr=LR_PRETRAIN, max_epochs=MAX_EPOCHS_PRETRAIN,
            patience=PATIENCE_PRETRAIN, weight_decay=wd,
            label=f"pre-train {source_machine} ({len(si_tr)} rows)",
        )

    # ── Phase 2: Fine-tuning on target machine ─────────────────────────────────
    print(f"\n  Fine-tuning on {target_machine} ...")
    best_finetune_epoch = run_train_stage(
        model,
        X_tgt_tr, yp_tgt_tr, yb_tgt_tr, ym_tgt_tr,
        X_tgt_val, yp_tgt_val, yb_tgt_val, ym_tgt_val,
        lr=LR_FINETUNE, max_epochs=MAX_EPOCHS_FINETUNE,
        patience=PATIENCE_FINETUNE, weight_decay=wd,
        label=f"fine-tune {target_machine} ({len(idx_tr)} rows)",
    )

    # ── Evaluation ─────────────────────────────────────────────────────────────
    tr_m = evaluate(model, X_tgt_tr, yp_tgt_tr, yb_tgt_tr, ym_tgt_tr)
    va_m = evaluate(model, X_tgt_val, yp_tgt_val, yb_tgt_val, ym_tgt_val)

    print(f"\n  {'Metric':<22} {'Train':>8} {'Val':>8}  {'Gap':>7}")
    print(f"  {'-'*50}")
    print(f"  {'Pressure MAPE':<22} {tr_m['pressure_mape']:>7.1f}% {va_m['pressure_mape']:>7.1f}%  {va_m['pressure_mape']-tr_m['pressure_mape']:>+6.1f}%")
    print(f"  {'Pressure MAE':<22} {tr_m['pressure_mae']:>7.1f}  {va_m['pressure_mae']:>7.1f}   {va_m['pressure_mae']-tr_m['pressure_mae']:>+6.1f}")
    print(f"  {'Blade accuracy':<22} {tr_m['blade_acc']:>8.3f} {va_m['blade_acc']:>8.3f}  {va_m['blade_acc']-tr_m['blade_acc']:>+7.3f}")
    print(f"  {'Multi-cut accuracy':<22} {tr_m['mcut_acc']:>8.3f} {va_m['mcut_acc']:>8.3f}  {va_m['mcut_acc']-tr_m['mcut_acc']:>+7.3f}")

    # ── ONNX export ────────────────────────────────────────────────────────────
    model.eval(); model.cpu()
    dummy     = torch.zeros(1, N_IN)
    onnx_path = os.path.join(MODEL_DIR, f"material_predictor_{cfg['slug']}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=12,
        input_names=["features"],
        output_names=["pressure_norm", "blade_logits", "multicut_logits"],
        dynamic_axes={
            "features":        {0: "batch_size"},
            "pressure_norm":   {0: "batch_size"},
            "blade_logits":    {0: "batch_size"},
            "multicut_logits": {0: "batch_size"},
        },
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"features": np.zeros((1, N_IN), dtype=np.float32)})
    size_kb = os.path.getsize(onnx_path) // 1024
    print(f"  ONNX → {os.path.basename(onnx_path)}  ({size_kb} KB)  ✓")
    model.to(DEVICE)

    return {
        "machine": target_machine, "slug": cfg["slug"],
        "n_train": len(idx_tr), "n_val": len(idx_val),
        "n_params": n_params,
        "finetune_best_epoch": best_finetune_epoch,
        "onnx_path": onnx_path, "onnx_size_kb": size_kb,
        "tr": tr_m, "va": va_m,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Train all machines
# ═══════════════════════════════════════════════════════════════════════════════

results = {}
for machine_name, cfg in MACHINE_CONFIGS.items():
    results[machine_name] = train_machine(machine_name, cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Consolidated summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print("  CONSOLIDATED RESULTS (Val 20% of target machine)")
print(f"{'='*80}")
print(f"{'Machine':<18} {'Rows':>5} {'FT-Ep':>6} {'Params':>7}  "
      f"{'MAPE%':>7} {'MAE':>7} {'BldAcc':>7} {'MCAcc':>7}")
print("-" * 72)
for mname, r in results.items():
    print(f"{mname:<18} {r['n_train']+r['n_val']:>5} {r['finetune_best_epoch']:>6} "
          f"{r['n_params']:>7}  "
          f"{r['va']['pressure_mape']:>6.1f}% "
          f"{r['va']['pressure_mae']:>7.1f} "
          f"{r['va']['blade_acc']:>7.3f} "
          f"{r['va']['mcut_acc']:>7.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Preprocessor JSON
# ═══════════════════════════════════════════════════════════════════════════════

per_machine_meta = {}
for mname, r in results.items():
    per_machine_meta[mname] = {
        "slug":              r["slug"],
        "onnx_file":         f"material_predictor_{r['slug']}.onnx",
        "n_training_rows":   r["n_train"],
        "finetune_best_epoch": r["finetune_best_epoch"],
        "val_pressure_mape": round(r["va"]["pressure_mape"], 2),
        "val_pressure_mae":  round(r["va"]["pressure_mae"], 1),
        "val_blade_acc":     round(r["va"]["blade_acc"], 4),
        "val_mcut_acc":      round(r["va"]["mcut_acc"], 4),
    }

preprocessor = {
    "version": "4.0",
    "feature_dim": N_IN,
    "machine_aliases": {
        "Cricut Explore 5": "Explore 3",
        "Cricut Maker 4":   "Maker 3",
    },
    "categories":              CATEGORIES,
    "blade_types_jp":          BLADE_TYPES,
    "blade_types_en":          BLADE_TYPES_EN,
    "multicut_bucket_labels":  MULTICUT_BUCKET_LABELS,
    "multicut_bucket_display": ["Off", "2×", "3×", "4–5×", "6–8×", "10+×"],
    # Global normalization — same for ALL machine models
    "pressure_log_mean": G_PRESSURE_LOG_MEAN,
    "pressure_log_std":  G_PRESSURE_LOG_STD,
    "thickness_log_min": float(G_THICK_MIN),
    "thickness_log_max": float(G_THICK_MAX),
    "gsm_log_min":       float(G_GSM_MIN),
    "gsm_log_max":       float(G_GSM_MAX),
    "density_log_min":   float(G_DENS_MIN),
    "density_log_max":   float(G_DENS_MAX),
    # Per-machine performance metadata
    "machines": per_machine_meta,
    # Inference defaults
    "thickness_defaults_mm": THICKNESS_DEFAULTS,
    "hardness_defaults":     HARDNESS_DEFAULTS,
    "surface_texture_map":   TEXTURE_MAP,
    "feature_order": (
        "category_onehot (11) + thickness_lognorm (1) + hardness_norm (1) "
        "+ is_bonded_fabric (1) + gsm_lognorm (1) + surface_texture_norm (1) "
        "+ has_adhesive (1) + density_lognorm (1) + shore_norm (1)"
    ),
    "notes": {
        "pressure_decode":  "exp(pressure_norm × pressure_log_std + pressure_log_mean)",
        "blade_decode":     "argmax(blade_logits) → blade_types_en[index]",
        "multicut_decode":  "argmax(multicut_logits) → multicut_bucket_labels[index]",
        "surface_norm":     "surface_texture_map[label] / 0.5",
        "training_method":  "Sequential transfer learning: pre-train on other machines (Maker3→Explore3→JoyXtra→Joy2→Joy), fine-tune on target",
        "compatible_blades": {
            "Cricut Joy":      ["Fine-Point Blade"],
            "Cricut Joy 2":    ["Fine-Point Blade"],
            "Cricut Joy Xtra": ["Fine-Point Blade", "Deep-Point Blade"],
            "Explore 3":       ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade"],
            "Maker 3":         ["Fine-Point Blade", "Deep-Point Blade", "Bonded Fabric Blade", "Knife Blade", "Rotary Blade"],
        }
    }
}

prep_path = os.path.join(MODEL_DIR, "preprocessor.json")
with open(prep_path, "w", encoding="utf-8") as f:
    json.dump(preprocessor, f, indent=2, ensure_ascii=False)
print(f"\nPreprocessor saved → {prep_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Training Report
# ═══════════════════════════════════════════════════════════════════════════════

rows_md = []
for mname, r in results.items():
    pretrain = " → ".join(m for m in PRETRAIN_ORDER if m != mname)
    rows_md.append(
        f"| {mname} | {r['n_train']+r['n_val']} | {r['finetune_best_epoch']} "
        f"| {r['n_params']:,} | {r['va']['pressure_mape']:.1f}% "
        f"| {r['va']['pressure_mae']:.1f} | {r['va']['blade_acc']:.3f} "
        f"| {r['va']['mcut_acc']:.3f} |"
    )

report_md = f"""# ML Training Report — v4 (Sequential Transfer Learning)
**Date**: 2026-06-06
**Method**: Sequential pre-training on other machines → fine-tune on target
**Pre-train order**: {" → ".join(PRETRAIN_ORDER)} (largest dataset first)
**Global pressure normalization**: log_mean={G_PRESSURE_LOG_MEAN:.4f}, log_std={G_PRESSURE_LOG_STD:.4f}

---

## Results (val = 20% of target machine)

| Machine | Rows | FT Epoch | Params | MAPE | MAE | Blade Acc | MC Acc |
|---------|------|----------|--------|------|-----|-----------|--------|
{chr(10).join(rows_md)}

---

## Feature Vector (19 dims, global normalization)

| Feature | Dim | Normalization |
|---------|-----|--------------|
| Category one-hot | 11 | Binary |
| Thickness (mm) | 1 | log1p → MinMax [0,1] |
| Hardness (1–10) | 1 | (h-1)/9 |
| Is Bonded Fabric | 1 | Binary |
| GSM (g/m²) | 1 | log1p → MinMax [0,1] |
| Surface Texture | 1 | texture_map / 0.5 |
| Has Adhesive | 1 | Binary |
| Density (kg/m³) | 1 | log1p → MinMax [0,1] |
| Shore Hardness A | 1 | / 100 |
| **Total** | **19** | |

---

## Architecture Per Machine

| Machine | Architecture | Dropout | Weight Decay |
|---------|-------------|---------|-------------|
| Cricut Joy | 19→24→12→heads | 0.50 / 0.40 | 1e-3 |
| Cricut Joy 2 | 19→24→12→heads | 0.50 / 0.40 | 1e-3 |
| Cricut Joy Xtra | 19→48→24→heads | 0.45 / 0.35 | 5e-4 |
| Explore 3 | 19→64→32→heads | 0.40 / 0.30 | 5e-4 |
| Maker 3 | 19→96→48→24→heads | 0.35 / 0.30 | 5e-4 |

All models: BatchNorm1d on every hidden layer, AdamW, ReduceLROnPlateau (factor=0.5, patience=30).
Pre-training: LR=1e-3, early stopping patience={PATIENCE_PRETRAIN}.
Fine-tuning: LR=2e-4, early stopping patience={PATIENCE_FINETUNE}.

---

## ONNX Files

All models share the **same input/output interface** (global normalization):
- Input: `features` float32 (batch, 19)
- `pressure_norm` → decode: `exp(x × {G_PRESSURE_LOG_STD:.4f} + {G_PRESSURE_LOG_MEAN:.4f})`
- `blade_logits` (batch, 5) → argmax → `blade_types_en[i]`
- `multicut_logits` (batch, 6) → argmax → `multicut_bucket_labels[i]`
"""

with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"Report saved → {REPORT}")
print("\n✓ Step 2 (v4) — sequential transfer learning complete.")