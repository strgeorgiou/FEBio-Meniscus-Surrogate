from pathlib import Path

import numpy as np
from PIL import Image
import torch
from matplotlib.colors import LinearSegmentedColormap

from meniscus_dataset import MeniscusFullDataset
from final_unet import MeniscusUNet


# --------- CONFIG ---------
DATA_ROOT  = r"path\to\dataset"
CKPT_PATH  = r"checkpoints\meniscus_unet_best.pth"
OUT_ROOT   = r"path\to\predictions_color"
TEST_CASES = ["walking_10", "walking_18"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------- FEBio-LIKE COLORMAP ---------
# positions: 0, 0.4, 0.5, 0.6, 1.0
points = [0.0, 0.4, 0.5, 0.6, 1.0]

# colours: blue -> cyan -> green -> yellow -> red
colors = [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
]

febio_cmap = LinearSegmentedColormap.from_list(
    "febio", list(zip(points, colors))
)


def save_color(arr, path, mask=None, vmin=0.0, vmax=1.0):
    """
    Save a 2D array as a color PNG using the FEBio colormap.
    arr   : 2D numpy array, assumed roughly in [0,1]
    mask  : 2D numpy array (0/1), same HxW; where mask==0 we paint black.
    """
    arr = np.clip(arr, vmin, vmax)

    if vmax > vmin:
        normed = (arr - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(arr)

    rgba = febio_cmap(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    if mask is not None:
        m = (mask > 0.5).astype(np.uint8)
        rgb = rgb * m[..., None]

    Image.fromarray(rgb).save(path)


# --------- LOAD DATASET & MODEL ---------
ds = MeniscusFullDataset(DATA_ROOT)

model = MeniscusUNet(
    in_channels=2,   # [mask, height]
    out_channels=4,  # [uz, strain, stress, cp]
    scalar_dim=3,    # [t, F_norm, theta_norm]
    base_channels=32,
).to(device)

print("Loading checkpoint:", CKPT_PATH)

state_dict = torch.load(
    CKPT_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

OUT_ROOT = Path(OUT_ROOT)
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# --------- FIND INDICES FOR TEST CASES ---------
test_indices = []

for idx, s in enumerate(ds.samples):
    case_name = s["case_dir"].parent.name

    if case_name in TEST_CASES:
        test_indices.append(idx)

print(f"Found {len(test_indices)} samples for test cases {TEST_CASES}")

names = ["uz", "strain", "stress", "cp"]


# --------- RUN INFERENCE & SAVE IMAGES ---------
for idx in test_indices:
    s = ds.samples[idx]

    case_name = s["case_dir"].parent.name
    step = s["step"]

    case_dir = OUT_ROOT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    x_img, x_scalars, y, mask_out = ds[idx]

    x_img_t = x_img.unsqueeze(0).to(device)
    x_scalars_t = x_scalars.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_img_t, x_scalars_t)[0]

    pred_np = pred.cpu().numpy()
    mask_np = mask_out.numpy()[0]

    pred_np_masked = pred_np * mask_np[None, :, :]

    for c, name in enumerate(names):
        pred_path = case_dir / f"step{step:02d}_{name}_pred.png"

        save_color(
            pred_np_masked[c],
            pred_path,
            mask=mask_np,
            vmin=0.0,
            vmax=1.0
        )