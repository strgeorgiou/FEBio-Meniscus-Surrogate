"""
Evaluate saved grayscale U-Net predictions against FEBio ground-truth images.

Metrics are calculated inside the meniscus mask in normalized image space
and physical units. Representative prediction and error heatmaps are also generated.
"""

from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

try:
    from meniscus_dataset import MeniscusFullDataset
except Exception as import_error:
    MeniscusFullDataset = None
    MENISCUS_DATASET_IMPORT_ERROR = import_error
else:
    MENISCUS_DATASET_IMPORT_ERROR = None


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_ROOT = Path(r"path\to\dataset")
PRED_ROOT = Path(r"path\to\predictions")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"metrics_FINAL_peak_abs_regenerated_heatmaps_{RUN_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=False)

EVAL_CASES = ["walking_10", "walking_18"]
REPRESENTATIVE_EXAMPLES = [("walking_10", 10)]

# Physical ranges used during export and normalization.
PHYS_RANGES = {
    "uz":     {"display": "z-displacement",   "range": (-1.2,  1.1), "unit": "mm",  "signed_peak": True},
    "strain": {"display": "Effective strain", "range": (0.0,  0.45), "unit": "-",   "signed_peak": False},
    "stress": {"display": "Von Mises stress", "range": (0.0, 21.0),  "unit": "MPa", "signed_peak": False},
    "cp":     {"display": "Contact pressure", "range": (0.0, 16.5),  "unit": "MPa", "signed_peak": False},
}

FIELDS = ["uz", "strain", "stress", "cp"]
EPS = 1e-12
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DATASET_PATH_KEY = {
    "uz": "uz_path",
    "strain": "strain_path",
    "stress": "stress_path",
    "cp": "cp_path",
}


# ============================================================
# 2. FEBIO-LIKE COLORMAP
# ============================================================

points = [0.0, 0.4, 0.5, 0.6, 1.0]

colors = [
    (0.0, 0.0, 1.0),  # blue
    (0.0, 1.0, 1.0),  # cyan
    (0.0, 1.0, 0.0),  # green
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 0.0),  # red
]

FEBIO_CMAP = LinearSegmentedColormap.from_list(
    "febio_like",
    list(zip(points, colors))
)


# ============================================================
# 3. HELPERS
# ============================================================

def load_gray_norm(path: Path) -> np.ndarray:
    """Load a grayscale image normalized to [0,1]."""
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    arr = np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float32
    )

    return arr / 255.0


def load_mask_required(path: Path) -> np.ndarray:
    """Load the binary dataset mask."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset mask: {path}"
        )

    arr = np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float32
    )

    return (arr > 127).astype(np.float32)


def norm_to_phys(
    arr01: np.ndarray,
    vmin: float,
    vmax: float
) -> np.ndarray:
    return vmin + arr01 * (vmax - vmin)


def masked_values(
    arr2d: np.ndarray,
    mask2d: np.ndarray
) -> np.ndarray:
    return arr2d[mask2d > 0.5]


def peak_value(
    vec: np.ndarray,
    signed: bool
) -> float:
    if vec.size == 0:
        return np.nan

    return float(
        np.max(np.abs(vec))
        if signed
        else np.max(vec)
    )


def fmt(x, digits: int = 6) -> str:
    try:
        x = float(x)
    except Exception:
        return ""

    if not np.isfinite(x):
        return ""

    return f"{x:.{digits}g}"


def sanitize(s: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_\-]+",
        "_",
        str(s)
    )


def save_febio_color(
    arr: np.ndarray,
    path: Path,
    mask: Optional[np.ndarray] = None,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    """Save a scalar map using the FEBio-like colormap."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    if vmax <= vmin:
        normed = np.zeros_like(arr)
    else:
        normed = (arr - vmin) / (vmax - vmin)

    normed = np.clip(
        normed,
        0.0,
        1.0
    )

    rgb = (
        FEBIO_CMAP(normed)[..., :3] * 255
    ).astype(np.uint8)

    if mask is not None:
        rgb[mask <= 0.5] = 0

    Image.fromarray(rgb).save(path)


def write_tsv(
    path: Path,
    headers: List[str],
    rows: List[Dict[str, object]]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=headers,
            delimiter="\t"
        )

        w.writeheader()

        for row in rows:
            w.writerow(
                {h: row.get(h, "") for h in headers}
            )


def write_csv(
    path: Path,
    headers: List[str],
    rows: List[Dict[str, object]]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=headers
        )

        w.writeheader()

        for row in rows:
            w.writerow(
                {h: row.get(h, "") for h in headers}
            )


# ============================================================
# 4. DATASET AND PREDICTION DISCOVERY
# ============================================================

def case_name_from_sample(sample: dict) -> str:
    d = Path(sample["case_dir"])
    return d.parent.name if d.name.lower() == "jobs" else d.name


def build_dataset_index() -> Dict[Tuple[str, int], dict]:
    """Build mapping from (case, step) to dataset sample."""

    if MeniscusFullDataset is None:
        raise RuntimeError(
            "Could not import MeniscusFullDataset from meniscus_dataset.py. "
            f"Original import error: {MENISCUS_DATASET_IMPORT_ERROR}"
        )

    ds = MeniscusFullDataset(DATA_ROOT)

    index: Dict[Tuple[str, int], dict] = {}

    for sample in ds.samples:
        cname = case_name_from_sample(sample)
        step = int(sample["step"])

        if EVAL_CASES and cname not in EVAL_CASES:
            continue

        key = (cname, step)

        if key in index:
            raise RuntimeError(
                f"Duplicate dataset sample for case={cname}, step={step}"
            )

        for path_key in [
            "mask_path",
            "uz_path",
            "strain_path",
            "stress_path",
            "cp_path"
        ]:
            p = Path(sample[path_key])

            if not p.exists():
                raise FileNotFoundError(
                    f"Dataset path missing for case={cname}, "
                    f"step={step}, {path_key}: {p}"
                )

        index[key] = sample

    return index


def find_case_folder(case: str) -> Path:
    direct = PRED_ROOT / case

    if direct.exists():
        return direct

    if not PRED_ROOT.exists():
        raise FileNotFoundError(
            f"PRED_ROOT does not exist: {PRED_ROOT}"
        )

    for p in PRED_ROOT.iterdir():
        if p.is_dir() and p.name.lower() == case.lower():
            return p

    raise FileNotFoundError(
        f"Could not find prediction case folder for {case}: "
        f"{PRED_ROOT / case}"
    )


def find_prediction_image(
    case_folder: Path,
    step: int,
    field: str
) -> Path:
    """Find the saved prediction image for a field and step."""

    patterns = [
        f"step{step:02d}_{field}_pred.*",
        f"step{step:03d}_{field}_pred.*",
        f"step{step}_{field}_pred.*",
        f"step_{step:02d}_{field}_pred.*",
        f"step_{step:03d}_{field}_pred.*",
        f"step_{step}_{field}_pred.*",
    ]

    matches: List[Path] = []

    for pat in patterns:
        matches.extend(
            case_folder.glob(pat)
        )

    matches = [
        m
        for m in matches
        if m.suffix.lower() in IMAGE_EXTS
    ]

    if not matches:
        raise FileNotFoundError(
            f"Missing saved prediction image for "
            f"case={case_folder.name}, step={step}, field={field}. "
            f"Expected something like step{step:02d}_{field}_pred.png "
            f"in {case_folder}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple prediction images for "
            f"{case_folder.name}, step={step}, field={field}. "
            f"Using {matches[0]}"
        )

    return matches[0]


# ============================================================
# 5. METRICS
# ============================================================

def compute_metrics(
    gt01: np.ndarray,
    pred01: np.ndarray,
    mask: np.ndarray,
    field: str
) -> Dict[str, float]:

    info = PHYS_RANGES[field]

    vmin, vmax = info["range"]
    phys_range = vmax - vmin
    signed_peak = bool(info["signed_peak"])

    gt_v = masked_values(gt01, mask)
    pr_v = masked_values(pred01, mask)

    if gt_v.size == 0:
        raise RuntimeError(
            "Mask contains no foreground pixels."
        )

    diff01 = pr_v - gt_v

    pixel_mae = float(
        np.mean(np.abs(diff01))
    )

    pixel_rmse = float(
        np.sqrt(np.mean(diff01 ** 2))
    )

    gt_phys = norm_to_phys(
        gt01,
        vmin,
        vmax
    )

    pr_phys = norm_to_phys(
        pred01,
        vmin,
        vmax
    )

    gt_phys_v = masked_values(
        gt_phys,
        mask
    )

    pr_phys_v = masked_values(
        pr_phys,
        mask
    )

    diff_phys = pr_phys_v - gt_phys_v

    mae = float(
        np.mean(np.abs(diff_phys))
    )

    rmse = float(
        np.sqrt(np.mean(diff_phys ** 2))
    )

    nmae_range = float(
        mae / (phys_range + EPS) * 100.0
    )

    nrmse_range = float(
        rmse / (phys_range + EPS) * 100.0
    )

    gt_peak = peak_value(
        gt_phys_v,
        signed_peak
    )

    pred_peak = peak_value(
        pr_phys_v,
        signed_peak
    )

    peak_abs_error = float(
        abs(gt_peak - pred_peak)
    )

    return {
        "pixel_mae": pixel_mae,
        "pixel_rmse": pixel_rmse,
        "mae": mae,
        "rmse": rmse,
        "nmae_range_pct": nmae_range,
        "nrmse_range_pct": nrmse_range,
        "gt_peak": gt_peak,
        "pred_peak": pred_peak,
        "peak_abs_error": peak_abs_error,
    }


def mean_rows(
    rows: List[Dict[str, object]],
    field: str
) -> Dict[str, float]:

    selected = [
        r
        for r in rows
        if r["field"] == field
    ]

    if not selected:
        raise RuntimeError(
            f"No rows found for field={field}"
        )

    keys = [
        "pixel_mae",
        "pixel_rmse",
        "mae",
        "rmse",
        "nmae_range_pct",
        "nrmse_range_pct",
        "gt_peak",
        "pred_peak",
        "peak_abs_error",
    ]

    return {
        k: float(
            np.nanmean(
                [float(r[k]) for r in selected]
            )
        )
        for k in keys
    }


# ============================================================
# 6. HEATMAP OUTPUTS
# ============================================================

def save_representative_images(
    case: str,
    step: int,
    field: str,
    gt01: np.ndarray,
    pred01: np.ndarray,
    mask: np.ndarray
) -> Dict[str, object]:
    """Save representative prediction and absolute-error heatmaps."""

    out = (
        OUT_DIR
        / "heatmaps"
        / f"{case}_step_{step:02d}"
        / field
    )

    # Normalized absolute error.
    err01 = np.abs(
        pred01 - gt01
    ).astype(np.float32)

    err01_masked = err01.copy()
    err01_masked[mask <= 0.5] = 0.0

    err01_max = (
        float(err01_masked[mask > 0.5].max())
        if np.any(mask > 0.5)
        else 0.0
    )

    # Physical-unit absolute error.
    info = PHYS_RANGES[field]

    vmin, vmax_range = info["range"]
    unit = info["unit"]

    gt_phys = norm_to_phys(
        gt01,
        vmin,
        vmax_range
    )

    pred_phys = norm_to_phys(
        pred01,
        vmin,
        vmax_range
    )

    err_phys = np.abs(
        pred_phys - gt_phys
    ).astype(np.float32)

    err_phys_masked = err_phys.copy()
    err_phys_masked[mask <= 0.5] = 0.0

    err_phys_max = (
        float(err_phys_masked[mask > 0.5].max())
        if np.any(mask > 0.5)
        else 0.0
    )

    # Colored prediction.
    save_febio_color(
        pred01,
        out / f"{field}_prediction_FEBio_color.png",
        mask,
        0.0,
        1.0,
    )

    # Normalized absolute-error heatmap.
    save_febio_color(
        err01,
        out / (
            f"{field}_absolute_error_PIXEL_"
            f"FEBio_color_scaled_to_max_{err01_max:.6f}.png"
        ),
        mask,
        0.0,
        err01_max if err01_max > 0 else 1.0,
    )

    # Physical-unit absolute-error heatmap.
    safe_unit = (
        sanitize(unit)
        if unit != "-"
        else "dimensionless"
    )

    save_febio_color(
        err_phys,
        out / (
            f"{field}_absolute_error_PHYSICAL_"
            f"FEBio_color_scaled_to_max_{err_phys_max:.6f}_"
            f"{safe_unit}.png"
        ),
        mask,
        0.0,
        err_phys_max if err_phys_max > 0 else 1.0,
    )

    return {
        "Case": case,
        "Step": step,
        "Output field": info["display"],
        "Field code": field,
        "Pixel error map scale":
            f"0 to {fmt(err01_max)} normalized intensity",
        "Physical error map scale":
            (
                f"0 to {fmt(err_phys_max)} {unit}"
                if unit != "-"
                else
                f"0 to {fmt(err_phys_max)} dimensionless strain"
            ),
    }


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:

    print("DATA_ROOT:", DATA_ROOT)
    print("PRED_ROOT:", PRED_ROOT)
    print("OUT_DIR  :", OUT_DIR)

    dataset_index = build_dataset_index()

    print(
        f"Dataset samples available for selected cases: "
        f"{len(dataset_index)}"
    )

    rows: List[Dict[str, object]] = []
    heatmap_scale_rows: List[Dict[str, object]] = []

    for case in EVAL_CASES:

        case_folder = find_case_folder(case)

        for step in range(1, 22):

            key = (case, step)

            if key not in dataset_index:
                raise RuntimeError(
                    f"Dataset does not contain required case/step: "
                    f"{case}, step {step}"
                )

            sample = dataset_index[key]

            mask = load_mask_required(
                Path(sample["mask_path"])
            )

            for field in FIELDS:

                gt_path = Path(
                    sample[DATASET_PATH_KEY[field]]
                )

                pred_path = find_prediction_image(
                    case_folder,
                    step,
                    field
                )

                gt01 = load_gray_norm(gt_path)
                pred01 = load_gray_norm(pred_path)

                if gt01.shape != pred01.shape:
                    raise RuntimeError(
                        f"Shape mismatch for case={case}, "
                        f"step={step}, field={field}: "
                        f"dataset GT {gt01.shape} from {gt_path}, "
                        f"prediction {pred01.shape} from {pred_path}"
                    )

                if mask.shape != gt01.shape:
                    raise RuntimeError(
                        f"Mask shape mismatch for case={case}, "
                        f"step={step}: "
                        f"mask {mask.shape} from {sample['mask_path']}, "
                        f"image {gt01.shape}"
                    )

                m = compute_metrics(
                    gt01,
                    pred01,
                    mask,
                    field
                )

                row = {
                    "case": case,
                    "step": step,
                    "field": field,
                    "field_display":
                        PHYS_RANGES[field]["display"],
                    "unit":
                        PHYS_RANGES[field]["unit"],
                    "dataset_gt_path":
                        str(gt_path),
                    "prediction_path":
                        str(pred_path),
                    "mask_path":
                        str(sample["mask_path"]),
                    **m,
                }

                rows.append(row)

                if (case, step) in REPRESENTATIVE_EXAMPLES:
                    heatmap_scale_rows.append(
                        save_representative_images(
                            case,
                            step,
                            field,
                            gt01,
                            pred01,
                            mask
                        )
                    )

    print(
        f"Evaluated {len(rows)} field maps."
    )

    print(
        f"Expected: {len(EVAL_CASES)} cases × "
        f"21 steps × 4 fields = "
        f"{len(EVAL_CASES) * 21 * 4}"
    )

    # Detailed per-field-map CSV.
    detail_headers = [
        "case",
        "step",
        "field",
        "field_display",
        "unit",
        "pixel_mae",
        "pixel_rmse",
        "mae",
        "rmse",
        "nmae_range_pct",
        "nrmse_range_pct",
        "gt_peak",
        "pred_peak",
        "peak_abs_error",
        "dataset_gt_path",
        "prediction_path",
        "mask_path",
    ]

    write_csv(
        OUT_DIR / "00_metrics_per_field_map.csv",
        detail_headers,
        rows
    )

    # Image-space full test table.
    image_full_rows: List[Dict[str, object]] = []

    for field in FIELDS:

        avg = mean_rows(
            rows,
            field
        )

        image_full_rows.append({
            "Output field":
                PHYS_RANGES[field]["display"],
            "Pixel MAE":
                fmt(avg["pixel_mae"]),
            "Pixel RMSE":
                fmt(avg["pixel_rmse"]),
        })

    write_tsv(
        OUT_DIR / "01_image_space_metrics_full_test_WORD.tsv",
        [
            "Output field",
            "Pixel MAE",
            "Pixel RMSE"
        ],
        image_full_rows
    )

    # Image-space representative table.
    rep_image_rows: List[Dict[str, object]] = []

    for case, step in REPRESENTATIVE_EXAMPLES:

        for field in FIELDS:

            matches = [
                r
                for r in rows
                if r["case"] == case
                and int(r["step"]) == step
                and r["field"] == field
            ]

            if not matches:
                raise RuntimeError(
                    f"Representative sample not found in results: "
                    f"{case}, step={step}, field={field}"
                )

            r = matches[0]

            rep_image_rows.append({
                "Case": case,
                "Step": step,
                "Output field":
                    PHYS_RANGES[field]["display"],
                "Pixel MAE":
                    fmt(r["pixel_mae"]),
                "Pixel RMSE":
                    fmt(r["pixel_rmse"]),
            })

    write_tsv(
        OUT_DIR / "01b_image_space_metrics_representative_WORD.tsv",
        [
            "Case",
            "Step",
            "Output field",
            "Pixel MAE",
            "Pixel RMSE"
        ],
        rep_image_rows
    )

    # Physical full test table.
    phys_full_rows: List[Dict[str, object]] = []

    for field in FIELDS:

        avg = mean_rows(
            rows,
            field
        )

        unit = PHYS_RANGES[field]["unit"]
        suffix = "" if unit == "-" else f" {unit}"

        phys_full_rows.append({
            "Output field":
                PHYS_RANGES[field]["display"],
            "MAE":
                fmt(avg["mae"]) + suffix,
            "RMSE":
                fmt(avg["rmse"]) + suffix,
            "NMAE range (%)":
                fmt(avg["nmae_range_pct"]),
            "NRMSE range (%)":
                fmt(avg["nrmse_range_pct"]),
        })

    write_tsv(
        OUT_DIR / "02_physical_unit_metrics_full_test_WORD.tsv",
        [
            "Output field",
            "MAE",
            "RMSE",
            "NMAE range (%)",
            "NRMSE range (%)"
        ],
        phys_full_rows
    )

    # Physical representative table.
    rep_phys_rows: List[Dict[str, object]] = []

    for case, step in REPRESENTATIVE_EXAMPLES:

        for field in FIELDS:

            matches = [
                r
                for r in rows
                if r["case"] == case
                and int(r["step"]) == step
                and r["field"] == field
            ]

            r = matches[0]

            unit = PHYS_RANGES[field]["unit"]
            suffix = "" if unit == "-" else f" {unit}"

            rep_phys_rows.append({
                "Case": case,
                "Step": step,
                "Output field":
                    PHYS_RANGES[field]["display"],
                "MAE":
                    fmt(r["mae"]) + suffix,
                "RMSE":
                    fmt(r["rmse"]) + suffix,
                "NMAE range (%)":
                    fmt(r["nmae_range_pct"]),
                "NRMSE range (%)":
                    fmt(r["nrmse_range_pct"]),
            })

    write_tsv(
        OUT_DIR / "02b_physical_unit_metrics_representative_WORD.tsv",
        [
            "Case",
            "Step",
            "Output field",
            "MAE",
            "RMSE",
            "NMAE range (%)",
            "NRMSE range (%)"
        ],
        rep_phys_rows
    )

    # Peak absolute error.
    peak_rows: List[Dict[str, object]] = []

    for field in FIELDS:

        avg = mean_rows(
            rows,
            field
        )

        unit = PHYS_RANGES[field]["unit"]
        suffix = "" if unit == "-" else f" {unit}"

        peak_rows.append({
            "Output field":
                PHYS_RANGES[field]["display"],
            "FEBio peak":
                fmt(avg["gt_peak"]) + suffix,
            "Prediction peak":
                fmt(avg["pred_peak"]) + suffix,
            "Peak absolute error":
                fmt(avg["peak_abs_error"]) + suffix,
        })

    write_tsv(
        OUT_DIR / "03_peak_absolute_error_full_test_WORD.tsv",
        [
            "Output field",
            "FEBio peak",
            "Prediction peak",
            "Peak absolute error"
        ],
        peak_rows
    )

    # Representative heatmap scales.
    write_tsv(
        OUT_DIR / "04_representative_heatmap_scales_WORD.tsv",
        [
            "Case",
            "Step",
            "Output field",
            "Field code",
            "Pixel error map scale",
            "Physical error map scale"
        ],
        heatmap_scale_rows
    )



    print(
        "Saved outputs to:",
        OUT_DIR.resolve()
    )

    print("Key Word-ready tables:")

    for name in [
        "01_image_space_metrics_full_test_WORD.tsv",
        "01b_image_space_metrics_representative_WORD.tsv",
        "02_physical_unit_metrics_full_test_WORD.tsv",
        "02b_physical_unit_metrics_representative_WORD.tsv",
        "03_peak_absolute_error_full_test_WORD.tsv",
        "04_representative_heatmap_scales_WORD.tsv",
    ]:
        print(
            "  ",
            OUT_DIR / name
        )

    print(
        "Representative colored prediction and error images saved under:",
        OUT_DIR / "heatmaps"
    )


if __name__ == "__main__":
    main()