import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """
    Two Conv-BN-ReLU blocks.

    in_ch -> out_ch -> out_ch
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool(2) -> DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d(2x) -> concat skip -> DoubleConv.

    Args:
        in_ch: channels coming from the previous decoder level (before upsample)
        out_ch: output channels after DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # up-convolution halves the channel count: in_ch -> in_ch//2
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # after concat: (in_ch//2 from up) + (in_ch//2 from skip) = in_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # handle odd sizes with padding
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class MeniscusUNet(nn.Module):
    """
    U-Net for meniscus surrogate with scalar conditioning at the bottleneck.

    Default thesis setup
    --------------------
    Image input:
        x_img: (B, 2, H, W)
            channel 0: mask           (0 or 1)
            channel 1: height_t0      (normalized to [0,1])

    Scalars from CSV:
        x_scalars: (B, 3)
            [t, F_norm, theta_norm]
        where:
            - t           ∈ [0,1]
            - F_norm      ∈ [0,1]    (your force shape)
            - theta_norm  ∈ [0,1]    (your angle shape)

    Output:
        y: (B, 4, H, W)
            channel 0: uz
            channel 1: strain_eq          (effective Lagrange strain)
            channel 2: stress_vm          (effective Cauchy/von Mises stress)
            channel 3: contact_pressure

    Notes:
        - No final activation: treat this as regression.
        - Apply your own normalization/denormalization outside the model.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,
        scalar_dim: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scalar_dim = scalar_dim

        # ----- Encoder -----
        self.inc   = DoubleConv(in_channels, base_channels)            # 32
        self.down1 = Down(base_channels, base_channels * 2)            # 64
        self.down2 = Down(base_channels * 2, base_channels * 4)        # 128
        self.down3 = Down(base_channels * 4, base_channels * 8)        # 256
        self.down4 = Down(base_channels * 8, base_channels * 16)       # 512

        self.bottleneck_channels = base_channels * 16

        # ----- Scalar embedding MLP -----
        # maps scalars (B, scalar_dim) -> (B, bottleneck_channels)
        if self.scalar_dim > 0:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(self.scalar_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.bottleneck_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.scalar_mlp = None

        # ----- Decoder -----
        self.up1 = Up(self.bottleneck_channels, base_channels * 8)     # 512 -> 256
        self.up2 = Up(base_channels * 8, base_channels * 4)            # 256 -> 128
        self.up3 = Up(base_channels * 4, base_channels * 2)            # 128 -> 64
        self.up4 = Up(base_channels * 2, base_channels)                # 64  -> 32

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x_img: torch.Tensor,
        x_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ----- Encoder -----
        x1 = self.inc(x_img)   # (B, 32, H,   W)
        x2 = self.down1(x1)    # (B, 64, H/2, W/2)
        x3 = self.down2(x2)    # (B, 128, ...)
        x4 = self.down3(x3)    # (B, 256, ...)
        x5 = self.down4(x4)    # (B, 512, H/16, W/16)

        # ----- Scalar conditioning at bottleneck -----
        if self.scalar_dim > 0:
            if x_scalars is None:
                raise ValueError("scalar_dim > 0 but x_scalars is None.")
            if x_scalars.dim() == 1:
                x_scalars = x_scalars.unsqueeze(0)

            # (B, scalar_dim) -> (B, bottleneck_channels, 1, 1)
            s = self.scalar_mlp(x_scalars)
            s = s.unsqueeze(-1).unsqueeze(-1)

            # broadcast add
            x5 = x5 + s

        # ----- Decoder with skip connections -----
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)  # (B, out_channels, H, W)
        return out


if __name__ == "__main__":
    # quick smoke test
    B, H, W = 2, 256, 256
    model = MeniscusUNet(
        in_channels=2,   # [mask, height_t0]
        out_channels=4,  # [uz, strain_eq, stress_vm, contact_pressure]
        scalar_dim=3,    # [t, F_norm, theta_norm]
        base_channels=32,
    )

    x_img = torch.randn(B, 2, H, W)
    x_scalars = torch.randn(B, 3)

    y = model(x_img, x_scalars)
    print("Output shape:", y.shape)   # -> (2, 4, 256, 256)
