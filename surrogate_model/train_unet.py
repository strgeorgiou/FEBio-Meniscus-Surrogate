import os
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from meniscus_dataset import MeniscusFullDataset
from final_unet import MeniscusUNet


# ===================== CONFIG =====================
DATA_ROOT   = r"path\to\dataset"
BATCH_SIZE  = 2
MAX_EPOCHS  = 150
LR          = 1e-4
NUM_WORKERS = 0
PATIENCE    = 10   # early stopping patience (epochs without val improvement)

# Range-enforcement strength (start here; try 0.1 if you still see many out-of-range pixels)
LAMBDA_RANGE = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Fixed unseen test cases
FIXED_TEST_CASES = ["walking_10", "walking_18"]


# ===================== DATASET & CASE-WISE SPLIT =====================
full_ds = MeniscusFullDataset(DATA_ROOT)

# Build mapping: case_name -> list of sample indices
case_to_indices = defaultdict(list)
for idx, s in enumerate(full_ds.samples):
    case_name = s["case_dir"].parent.name  # e.g. "walking_1", "force_3"
    case_to_indices[case_name].append(idx)

all_cases = sorted(case_to_indices.keys())
n_cases = len(all_cases)
print(f"Total cases found: {n_cases}")
print("All cases:", all_cases)

# Check that fixed test cases exist
for cname in FIXED_TEST_CASES:
    if cname not in case_to_indices:
        raise ValueError(
            f"Fixed test case '{cname}' not found in dataset. "
            f"Available cases: {all_cases}"
        )

# Remove fixed test cases from the pool
remaining_cases = [c for c in all_cases if c not in FIXED_TEST_CASES]

# We want: 26 train + 5 val + 2 fixed test = 33 total cases
if len(remaining_cases) != 31:
    print("WARNING: Expected 31 remaining cases after removing test cases, "
          f"but got {len(remaining_cases)}. Adjust counts if needed.")

# Shuffle remaining cases deterministically
rng = random.Random(42)
rng.shuffle(remaining_cases)

# Take 5 for validation, rest for training
val_cases   = remaining_cases[:5]
train_cases = remaining_cases[5:]

# Build test cases from fixed list
test_cases = FIXED_TEST_CASES

print("\n=== CASE SPLIT ===")
print("Train cases:", sorted(train_cases))
print("Val cases  :", sorted(val_cases))
print("Test cases :", sorted(test_cases))

# Sanity checks
if len(train_cases) != 26:
    print(f"WARNING: train_cases = {len(train_cases)} (expected 26)")
if len(val_cases) != 5:
    print(f"WARNING: val_cases = {len(val_cases)} (expected 5)")
if len(test_cases) != 2:
    print(f"WARNING: test_cases = {len(test_cases)} (expected 2)")

# Build index lists for each split
train_indices = [i for cname in train_cases for i in case_to_indices[cname]]
val_indices   = [i for cname in val_cases   for i in case_to_indices[cname]]
test_indices  = [i for cname in test_cases  for i in case_to_indices[cname]]

train_ds = Subset(full_ds, train_indices)
val_ds   = Subset(full_ds, val_indices)
test_ds  = Subset(full_ds, test_indices)

print(f"\nSamples per split:")
print(f"  Train: {len(train_ds)}")
print(f"  Val  : {len(val_ds)}")
print(f"  Test : {len(test_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ===================== MODEL & OPTIMIZER =====================
model = MeniscusUNet(
    in_channels=2,   # [mask, height]
    out_channels=4,  # [uz, strain, stress, cp]
    scalar_dim=3,    # [t, F_norm, theta_norm]
    base_channels=32,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ===================== LOSSES =====================
def masked_l1_with_channels(pred, target, mask):
    """
    pred, target: (B, C, H, W)
    mask:        (B, 1, H, W)  1 inside meniscus, 0 outside

    Returns:
      total_loss: scalar (masked mean absolute error over all channels)
      per_channel: (C,) tensor (masked MAE per output channel)
    """
    mask_c = mask.expand_as(pred)               # (B, C, H, W)
    loss_map = torch.abs(pred - target) * mask_c

    num_masked = mask.sum()                     # total pixels with mask=1 (over batch)
    total_loss = loss_map.sum() / (num_masked * pred.shape[1] + 1e-8)

    per_ch = loss_map.sum(dim=[0, 2, 3]) / (num_masked + 1e-8)  # (C,)

    return total_loss, per_ch


def masked_range_penalty(pred, mask, eps=0.0):
    """
    Penalize predictions outside [0,1] inside the meniscus mask only.
    pred: (B, C, H, W)  (unbounded regression output)
    mask: (B, 1, H, W)  (1 inside meniscus)

    Penalty is 0 if pred is in [0,1]. It grows quadratically outside the range.
    """
    mask_c = mask.expand_as(pred)

    below = F.relu((0.0 + eps) - pred)     # >0 when pred < 0
    above = F.relu(pred - (1.0 - eps))     # >0 when pred > 1

    pen_map = (below**2 + above**2) * mask_c
    num_masked = mask.sum()

    return pen_map.sum() / (num_masked * pred.shape[1] + 1e-8)


def total_loss_with_range(pred, target, mask, lambda_range=LAMBDA_RANGE):
    """
    Total loss = masked L1 + lambda * masked range penalty
    Returns: total, l1, penalty, per_channel_l1
    """
    l1, per_ch = masked_l1_with_channels(pred, target, mask)
    pen = masked_range_penalty(pred, mask)
    total = l1 + lambda_range * pen
    return total, l1, pen, per_ch


# ===================== TRAINING LOOP WITH EARLY STOPPING =====================
best_val_total = float("inf")
epochs_no_improve = 0

os.makedirs("checkpoints", exist_ok=True)
best_ckpt_path = "checkpoints/meniscus_unet_best.pth"

for epoch in range(1, MAX_EPOCHS + 1):
    # ------- TRAIN -------
    model.train()
    train_total_sum = 0.0
    train_l1_sum = 0.0
    train_pen_sum = 0.0
    train_ch_sum = torch.zeros(4, device=DEVICE)

    for x_img, x_scalars, y, mask_out in train_loader:
        x_img     = x_img.to(DEVICE)
        x_scalars = x_scalars.to(DEVICE)
        y         = y.to(DEVICE)
        mask_out  = mask_out.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x_img, x_scalars)

        total, l1, pen, per_ch = total_loss_with_range(pred, y, mask_out)
        total.backward()
        optimizer.step()

        batch_size = x_img.size(0)
        train_total_sum += total.item() * batch_size
        train_l1_sum    += l1.item() * batch_size
        train_pen_sum   += pen.item() * batch_size
        train_ch_sum    += per_ch * batch_size

    train_total_avg = train_total_sum / len(train_ds)
    train_l1_avg    = train_l1_sum / len(train_ds)
    train_pen_avg   = train_pen_sum / len(train_ds)
    train_ch_avg    = (train_ch_sum / len(train_ds)).detach().cpu().numpy()

    # ------- VALIDATION -------
    model.eval()
    val_total_sum = 0.0
    val_l1_sum = 0.0
    val_pen_sum = 0.0
    val_ch_sum = torch.zeros(4, device=DEVICE)

    with torch.no_grad():
        for x_img, x_scalars, y, mask_out in val_loader:
            x_img     = x_img.to(DEVICE)
            x_scalars = x_scalars.to(DEVICE)
            y         = y.to(DEVICE)
            mask_out  = mask_out.to(DEVICE)

            pred = model(x_img, x_scalars)
            total, l1, pen, per_ch = total_loss_with_range(pred, y, mask_out)

            batch_size = x_img.size(0)
            val_total_sum += total.item() * batch_size
            val_l1_sum    += l1.item() * batch_size
            val_pen_sum   += pen.item() * batch_size
            val_ch_sum    += per_ch * batch_size

    val_total_avg = val_total_sum / len(val_ds)
    val_l1_avg    = val_l1_sum / len(val_ds)
    val_pen_avg   = val_pen_sum / len(val_ds)
    val_ch_avg    = (val_ch_sum / len(val_ds)).detach().cpu().numpy()

    uz_t, strain_t, stress_t, cp_t = train_ch_avg
    uz_v, strain_v, stress_v, cp_v = val_ch_avg

    print(
        f"Epoch {epoch:03d} | "
        f"train total: {train_total_avg:.5f} (L1 {train_l1_avg:.5f}, pen {train_pen_avg:.5f}) "
        f"(uz {uz_t:.4f}, strain {strain_t:.4f}, stress {stress_t:.4f}, cp {cp_t:.4f}) | "
        f"val total: {val_total_avg:.5f} (L1 {val_l1_avg:.5f}, pen {val_pen_avg:.5f}) "
        f"(uz {uz_v:.4f}, strain {strain_v:.4f}, stress {stress_v:.4f}, cp {cp_v:.4f})"
    )

    # ------- EARLY STOPPING (based on total validation loss) -------
    if val_total_avg < best_val_total - 1e-6:
        best_val_total = val_total_avg
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  -> New best val total, model saved to {best_ckpt_path}.")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}.")
        break

print("\nTraining finished. Best val total:", best_val_total)


# ===================== FINAL TEST EVAL ON UNSEEN CASES =====================
print("\n=== Evaluating on UNSEEN test cases ===")
print("Test cases:", sorted(test_cases))

# Load best checkpoint
model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

test_total_sum = 0.0
test_l1_sum = 0.0
test_pen_sum = 0.0
test_ch_sum = torch.zeros(4, device=DEVICE)

with torch.no_grad():
    for x_img, x_scalars, y, mask_out in test_loader:
        x_img     = x_img.to(DEVICE)
        x_scalars = x_scalars.to(DEVICE)
        y         = y.to(DEVICE)
        mask_out  = mask_out.to(DEVICE)

        pred = model(x_img, x_scalars)
        total, l1, pen, per_ch = total_loss_with_range(pred, y, mask_out)

        batch_size = x_img.size(0)
        test_total_sum += total.item() * batch_size
        test_l1_sum    += l1.item() * batch_size
        test_pen_sum   += pen.item() * batch_size
        test_ch_sum    += per_ch * batch_size

test_total_avg = test_total_sum / len(test_ds)
test_l1_avg    = test_l1_sum / len(test_ds)
test_pen_avg   = test_pen_sum / len(test_ds)
test_ch_avg    = (test_ch_sum / len(test_ds)).detach().cpu().numpy()

uz_te, strain_te, stress_te, cp_te = test_ch_avg
print(
    f"Test total: {test_total_avg:.5f} (L1 {test_l1_avg:.5f}, pen {test_pen_avg:.5f}) "
    f"(uz {uz_te:.4f}, strain {strain_te:.4f}, stress {stress_te:.4f}, cp {cp_te:.4f})"
)
