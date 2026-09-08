from pathlib import Path
import csv
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def _find_image_for_step(
    d: Path,
    base_names: list[str],
    step: int,
) -> Path | None:

    label4 = f"{step:04d}"
    candidate_stems = [f"{b}{label4}" for b in base_names]

    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if p.stem in candidate_stems:
            return p

    return None


class MeniscusFullDataset(Dataset):
    """
    Dataset for the full meniscus U-Net.

    Expects per case:
      - One CSV with columns:
          step, t, F, F_norm, theta_rad, theta_norm

      - Images named like:
          z_displacement0001.png
          effective_strain0001.png
          effective_stress0001.png
          contact_pressure0001.png
          mask0001.png
          height0001.png

        

    Returns:
      x_img:     (2, H, W)   [mask, height]
      x_scalars: (3,)        [t, F_norm, theta_norm]
      y:         (4, H, W)   [uz, strain_eq, stress_vm, contact_pressure]
      mask_out:  (1, H, W)   binary mask
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.samples: list[dict] = []

        print("Scanning dataset root:", self.root_dir)

        # Treat each subdirectory containing a CSV as one simulation case
        for case_dir in sorted(self.root_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            print(f"\n--- Case folder: {case_dir.name} ---")

            # Prefer the "jobs" subfolder if present
            candidates = []

            jobs_dir = case_dir / "jobs"
            if jobs_dir.exists():
                candidates.append(jobs_dir)

            candidates.append(case_dir)

            data_dir = None
            scalars_path = None

            for d in candidates:
                # Prefer CSV files beginning with "scalars"
                scalars_candidates = sorted(d.glob("scalars*.csv"))

                if scalars_candidates:
                    data_dir = d
                    scalars_path = scalars_candidates[0]
                    break

                # Fallback to any CSV file
                generic_csvs = sorted(d.glob("*.csv"))

                if generic_csvs:
                    data_dir = d
                    scalars_path = generic_csvs[0]
                    break

            if scalars_path is None or data_dir is None:
                print("  -> No CSV found, skipping this case.")
                continue

            print("  Using data dir:", data_dir)
            print("  Scalars file :", scalars_path.name)

            n_before = len(self.samples)

            with scalars_path.open("r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        step = int(row["step"])
                        t = float(row["t"])
                        F_norm = float(row["F_norm"])
                        theta_norm = float(row["theta_norm"])

                    except KeyError as e:
                        print("  !! Missing column in scalars CSV:", e)
                        continue

                    # Find all required images for this step
                    uz_path = _find_image_for_step(
                        data_dir,
                        base_names=["z_displacement", "z-displacement"],
                        step=step,
                    )

                    strain_path = _find_image_for_step(
                        data_dir,
                        base_names=["effective_strain", "strain_eq"],
                        step=step,
                    )

                    stress_path = _find_image_for_step(
                        data_dir,
                        base_names=["effective_stress", "stress_vm"],
                        step=step,
                    )

                    cp_path = _find_image_for_step(
                        data_dir,
                        base_names=["contact_pressure", "contact-pressure"],
                        step=step,
                    )

                    mask_path = _find_image_for_step(
                        data_dir,
                        base_names=["mask"],
                        step=step,
                    )

                    # Height can be step-specific or a single reference image
                    height_path = _find_image_for_step(
                        data_dir,
                        base_names=["height"],
                        step=step,
                    )

                    if height_path is None:
                        for name in ["height_t0", "height"]:
                            p = data_dir / f"{name}.png"

                            if p.exists():
                                height_path = p
                                break

                    if any(
                        p is None
                        for p in [
                            uz_path,
                            strain_path,
                            stress_path,
                            cp_path,
                            mask_path,
                            height_path,
                        ]
                    ):
                        print(
                            f"  Missing images at step {step}:",
                            f"uz={uz_path}, strain={strain_path}, stress={stress_path},",
                            f"cp={cp_path}, mask={mask_path}, height={height_path}",
                        )
                        continue

                    self.samples.append(
                        dict(
                            case_dir=data_dir,
                            step=step,
                            t=t,
                            F_norm=F_norm,
                            theta_norm=theta_norm,
                            uz_path=uz_path,
                            strain_path=strain_path,
                            stress_path=stress_path,
                            cp_path=cp_path,
                            mask_path=mask_path,
                            height_path=height_path,
                        )
                    )

            n_after = len(self.samples)
            print(f"  -> Added {n_after - n_before} samples from this case.")

        print("\nTotal samples collected:", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Load grayscale images
        def load_gray(path: Path) -> np.ndarray:
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.float32)

        uz_img = load_gray(s["uz_path"])
        strain_img = load_gray(s["strain_path"])
        stress_img = load_gray(s["stress_path"])
        cp_img = load_gray(s["cp_path"])
        mask_img = load_gray(s["mask_path"])
        height_img = load_gray(s["height_path"])

        # Convert mask to binary 0/1
        mask_bin = (mask_img > 127).astype(np.float32)

        # Normalize height to [0,1]
        height_norm = height_img / 255.0

        # Image input: mask and height
        x_img = np.stack(
            [mask_bin, height_norm],
            axis=0,
        )

        # Scalar inputs
        t = float(s["t"])
        F_norm = float(s["F_norm"])
        theta_norm = float(s["theta_norm"])

        x_scalars = np.array(
            [t, F_norm, theta_norm],
            dtype=np.float32,
        )

        # Normalize target images to [0,1]
        uz_norm = uz_img / 255.0
        strain_norm = strain_img / 255.0
        stress_norm = stress_img / 255.0
        cp_norm = cp_img / 255.0

        y = np.stack(
            [
                uz_norm,
                strain_norm,
                stress_norm,
                cp_norm,
            ],
            axis=0,
        )

        mask_out = mask_bin[None, ...]

        return (
            torch.from_numpy(x_img),
            torch.from_numpy(x_scalars),
            torch.from_numpy(y),
            torch.from_numpy(mask_out),
        )
