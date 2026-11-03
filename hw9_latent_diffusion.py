"""
DATA 266 HW9 — Latent Diffusion and Stable Diffusion
SJSU ID 019129357. Run top-to-bottom; writes figures/, checkpoints/, METRICS.md, RUN_LOG.txt.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 0. Standing requirements — personal parameters (first thing we report)
# ---------------------------------------------------------------------------
SJSU_ID = "019129357"
SID4 = int(SJSU_ID[-4:])  # 9357
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10

NULL_LABEL = 10  # 10 Fashion-MNIST classes + 1 null label for CFG
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CKPT_DIR = ROOT / "checkpoints"
ART_DIR = ROOT / "artifacts"
for p in (DATA_DIR, FIG_DIR, CKPT_DIR, ART_DIR):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Config — assignment-specified values only (no extra HP_ID mapping in HW9)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    image_size: int = 128  # 128 / 8 = 16 latent, so self-attention is at 16x16
    latent_channels: int = 4
    latent_size: int = 16
    vae_scale: float = 0.18215
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4)
    epochs: int = 50
    lr: float = 2e-5
    batch_size: int = 256
    drop_label_prob: float = 0.1
    guidance_w: float = 3.0
    num_classes: int = 10
    vae_id: str = "stabilityai/sd-vae-ft-mse"


CFG = Config()


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 3. U-Net denoiser (operates on VAE latents)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        k = k.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        v = v.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class LatentUNet(nn.Module):
    """U-Net on 4x16x16 latents: base 64, multipliers (1,2,4), attn at 16x16 + bottleneck."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        multipliers: tuple = (1, 2, 4),
        num_classes: int = 10,
        time_dim: int = 256,
        latent_size: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = time_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # 10 classes + 1 null label
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim)

        channels = [base_channels * m for m in multipliers]
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Down path
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = channels[0]
        res = latent_size
        self.down_resolutions = []
        for i, ch in enumerate(channels):
            self.down_res.append(
                nn.ModuleList(
                    [
                        ResidualBlock(ch_in, ch, emb_dim),
                        ResidualBlock(ch, ch, emb_dim),
                    ]
                )
            )
            use_attn = (res == 16) or (i == len(channels) - 1)
            self.down_attn.append(SelfAttention(ch) if use_attn else nn.Identity())
            self.down_resolutions.append(res)
            self.downsamples.append(
                Downsample(ch) if i < len(channels) - 1 else nn.Identity()
            )
            ch_in = ch
            if i < len(channels) - 1:
                res //= 2

        # Bottleneck
        mid_ch = channels[-1]
        self.mid_res1 = ResidualBlock(mid_ch, mid_ch, emb_dim)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_res2 = ResidualBlock(mid_ch, mid_ch, emb_dim)

        # Up path
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            skip_ch = ch
            self.up_res.append(
                nn.ModuleList(
                    [
                        ResidualBlock(ch_in + skip_ch, ch, emb_dim),
                        ResidualBlock(ch, ch, emb_dim),
                    ]
                )
            )
            res_here = self.down_resolutions[i]
            use_attn = (res_here == 16) or (i == len(channels) - 1)
            self.up_attn.append(SelfAttention(ch) if use_attn else nn.Identity())
            self.upsamples.append(
                Upsample(ch) if i > 0 else nn.Identity()
            )
            ch_in = ch

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.conv_out = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.label_emb(y)

        h = self.conv_in(z)
        skips = []
        for res_blocks, attn, down in zip(self.down_res, self.down_attn, self.downsamples):
            for block in res_blocks:
                h = block(h, emb)
            h = attn(h)
            skips.append(h)
            h = down(h)

        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        for res_blocks, attn, up in zip(self.up_res, self.up_attn, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in res_blocks:
                h = block(h, emb)
            h = attn(h)
            h = up(h)

        return self.conv_out(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# 2. Forward Gaussian diffusion on latents
# ---------------------------------------------------------------------------
class LatentDiffusionSchedule:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """q(z_t | z_0) = N(sqrt(alpha_bar_t) z_0, (1 - alpha_bar_t) I)."""
        if noise is None:
            noise = torch.randn_like(z0)
        s = t.shape[0]
        a = self.sqrt_alpha_bars[t].view(s, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(s, 1, 1, 1)
        return a * z0 + b * noise, noise


@torch.no_grad()
def p_sample_cfg(
    model: nn.Module,
    schedule: LatentDiffusionSchedule,
    shape: tuple,
    labels: torch.Tensor,
    w: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    z = torch.randn(shape, device=device, generator=generator)
    null = torch.full_like(labels, NULL_LABEL)
    for t_int in tqdm(range(schedule.T - 1, -1, -1), desc="DDPM sample", leave=False):
        t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([labels, null], dim=0)
        with torch.amp.autocast("cuda", enabled=z.device.type == "cuda", dtype=torch.bfloat16):
            eps_cond, eps_uncond = model(z_in, t_in, y_in).float().chunk(2, dim=0)
        eps = (1.0 + w) * eps_cond - w * eps_uncond

        alpha = schedule.alphas[t_int]
        alpha_bar = schedule.alpha_bars[t_int]
        beta = schedule.betas[t_int]
        mean = (1.0 / torch.sqrt(alpha)) * (z - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if t_int > 0:
            noise = torch.randn(shape, device=device, generator=generator)
            z = mean + torch.sqrt(beta) * noise
        else:
            z = mean
    return z


# ---------------------------------------------------------------------------
# VAE helpers
# ---------------------------------------------------------------------------
def fashion_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def to_display(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] RGB -> [0,1] grayscale. Handles CHW and NCHW."""
    x = (x.clamp(-1, 1) + 1) / 2
    if x.dim() == 4:
        return x.mean(dim=1, keepdim=True)
    if x.dim() == 3:
        return x.mean(dim=0, keepdim=True)
    return x


@torch.no_grad()
def encode_images(vae, images: torch.Tensor, scale: float, sample: bool = False) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    z = posterior.sample() if sample else posterior.mode()
    return z * scale


@torch.no_grad()
def decode_latents(vae, z_scaled: torch.Tensor, scale: float) -> torch.Tensor:
    return vae.decode(z_scaled / scale).sample


def save_grid(tensor: torch.Tensor, path: Path, nrow: int, title: str | None = None):
    grid = make_grid(to_display(tensor).cpu(), nrow=nrow, padding=2)
    if title is None:
        save_image(grid, path)
        return
    fig, ax = plt.subplots(figsize=(nrow * 1.3, (tensor.size(0) // nrow + 1) * 1.3))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics: dict = {
        "SID4": SID4,
        "SEED": SEED,
        "SLICE": SLICE,
        "HP_ID": HP_ID,
        "CLS_A": CLS_A,
        "CLS_B": CLS_B,
        "device": str(device),
        "started": now_utc(),
    }

    log("=" * 72)
    log("DATA 266 HW9 — Latent Diffusion and Stable Diffusion")
    log(f"SID4={SID4}  SEED={SEED}  SLICE={SLICE}  HP_ID={HP_ID}  CLS_A={CLS_A} ({CLASS_NAMES[CLS_A]})  CLS_B={CLS_B} ({CLASS_NAMES[CLS_B]})")
    log(f"device={device}  {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    log("HP_ID 3 is Learning-rate-low in Table 0.1, but HW9 has no HP_ID mapping.")
    log("Using the assignment's fixed settings: 50 epochs, LR=2e-5. No second model.")
    log("=" * 72)

    # ------------------------------------------------------------------
    # 1. VAE compression
    # ------------------------------------------------------------------
    log("Section 1: loading pretrained VAE " + CFG.vae_id)
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(CFG.vae_id)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    reported_scale = float(getattr(vae.config, "scaling_factor", CFG.vae_scale))
    log(f"VAE scaling_factor from config={reported_scale}; assignment s={CFG.vae_scale}")
    metrics["vae_config_scale"] = reported_scale
    metrics["vae_scale_used"] = CFG.vae_scale

    tfm = fashion_transform(CFG.image_size)
    train_set = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    raw_set = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    log(f"FashionMNIST train size={len(train_set)}; visualization start index SLICE={SLICE}")

    vis_idx = list(range(SLICE, SLICE + 8))
    vis_imgs = torch.stack([train_set[i][0] for i in vis_idx]).to(device)
    vis_labels = torch.tensor([train_set[i][1] for i in vis_idx], device=device)

    with torch.no_grad():
        z0 = encode_images(vae, vis_imgs, CFG.vae_scale, sample=False)
        recon = decode_latents(vae, z0, CFG.vae_scale)
    mse_128 = F.mse_loss(recon, vis_imgs).item()

    recon_gray_28 = F.interpolate(to_display(recon), size=28, mode="bilinear", align_corners=False)
    orig_gray_28 = torch.stack([raw_set[i][0] for i in vis_idx]).to(device)
    mse_28 = F.mse_loss(recon_gray_28, orig_gray_28).item()
    metrics["recon_mse_128_rgb"] = mse_128
    metrics["recon_mse_28_gray"] = mse_28
    log(f"Reconstruction MSE on 128x128 RGB ([-1,1] space) = {mse_128:.6f}")
    log(f"Reconstruction MSE on 28x28 grayscale [0,1]     = {mse_28:.6f}")
    log(f"Latent tensor shape z0 = {tuple(z0.shape)}  (C,H,W should be 4x16x16)")

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for i in range(8):
        axes[0, i].imshow(to_display(vis_imgs[i]).cpu().squeeze(), cmap="gray")
        axes[0, i].set_title(CLASS_NAMES[int(vis_labels[i])], fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(to_display(recon[i]).cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(f"VAE reconstructions (indices {SLICE}..{SLICE+7})  MSE_28={mse_28:.4f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_vae_reconstructions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Forward diffusion visualizations
    # ------------------------------------------------------------------
    log("Section 2: forward diffusion in latent space")
    schedule = LatentDiffusionSchedule(CFG.timesteps, CFG.beta_start, CFG.beta_end, device)
    viz_ts = [0, 200, 500, 800, 1000]
    n_show = 4
    z_clean = z0[:n_show]
    fig, axes = plt.subplots(n_show, len(viz_ts), figsize=(12, 8))
    set_seed(SEED)
    for col, t_val in enumerate(viz_ts):
        t_idx = min(t_val, CFG.timesteps - 1)
        t = torch.full((n_show,), t_idx, device=device, dtype=torch.long)
        if t_val == 0:
            z_t = z_clean
        elif t_val >= CFG.timesteps:
            z_t = torch.randn_like(z_clean)
        else:
            z_t, _ = schedule.q_sample(z_clean, t)
        decoded = decode_latents(vae, z_t, CFG.vae_scale)
        for row in range(n_show):
            axes[row, col].imshow(to_display(decoded[row]).cpu().squeeze(), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t={t_val}")
    fig.suptitle("Decoded noisy latents z_t  (linear beta schedule, T=1000)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_forward_diffusion_latents.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log("Saved forward-diffusion visualization.")

    # ------------------------------------------------------------------
    # Encode full training set into latent cache
    # ------------------------------------------------------------------
    cache_path = ART_DIR / "fashionmnist_latents.pt"
    if cache_path.exists():
        log(f"Loading cached latents from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_z = blob["z"]
        all_y = blob["y"]
    else:
        log("Encoding full FashionMNIST train set into latent space (one-time).")
        loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0)
        zs, ys = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="VAE encode"):
                imgs = imgs.to(device)
                z = encode_images(vae, imgs, CFG.vae_scale, sample=True)
                zs.append(z.cpu())
                ys.append(labels)
        all_z = torch.cat(zs, dim=0)
        all_y = torch.cat(ys, dim=0)
        torch.save({"z": all_z, "y": all_y, "scale": CFG.vae_scale}, cache_path)
        log(f"Cached {tuple(all_z.shape)} latents to {cache_path}")
    metrics["latent_mean"] = float(all_z.mean())
    metrics["latent_std"] = float(all_z.std())
    metrics["n_train"] = int(all_z.size(0))
    log(f"Scaled latent stats: mean={metrics['latent_mean']:.4f}  std={metrics['latent_std']:.4f}")

    # ------------------------------------------------------------------
    # 3–4. Build U-Net and train noise predictor
    # ------------------------------------------------------------------
    set_seed(SEED)
    unet = LatentUNet(
        in_channels=CFG.latent_channels,
        base_channels=CFG.base_channels,
        multipliers=CFG.channel_mults,
        num_classes=CFG.num_classes,
        latent_size=CFG.latent_size,
    ).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    metrics["unet_params"] = n_params
    log(f"U-Net parameters: {n_params:,}")
    log("Architecture: ResBlocks + GroupNorm + SiLU; sinusoidal time + MLP; label+time sum; "
        "self-attention at 16x16 and bottleneck; skip connections.")

    ds = TensorDataset(all_z, all_y)
    train_loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(unet.parameters(), lr=CFG.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = CKPT_DIR / "ldm_unet.pt"
    history = {"epoch": [], "loss": []}
    t0 = time.time()
    log(f"Section 4: training {CFG.epochs} epochs, lr={CFG.lr}, batch={CFG.batch_size}, "
        f"label-drop p={CFG.drop_label_prob}")

    unet.train()
    for epoch in range(1, CFG.epochs + 1):
        running = 0.0
        n_seen = 0
        for z_batch, y_batch in train_loader:
            z_batch = z_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            drop = torch.rand(y_batch.size(0), device=device) < CFG.drop_label_prob
            y_in = y_batch.clone()
            y_in[drop] = NULL_LABEL
            t = torch.randint(0, CFG.timesteps, (z_batch.size(0),), device=device)
            noise = torch.randn_like(z_batch)
            z_t, _ = schedule.q_sample(z_batch, t, noise)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                pred = unet(z_t, t, y_in)
                loss = F.mse_loss(pred.float(), noise)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * z_batch.size(0)
            n_seen += z_batch.size(0)
        epoch_loss = running / max(n_seen, 1)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == CFG.epochs:
            log(f"epoch {epoch:03d}/{CFG.epochs}  noise-pred MSE={epoch_loss:.6f}")

    train_seconds = time.time() - t0
    metrics["train_seconds"] = train_seconds
    metrics["final_train_loss"] = history["loss"][-1]
    metrics["first_epoch_loss"] = history["loss"][0]
    log(f"Training finished in {train_seconds/60:.2f} min. Final loss={history['loss'][-1]:.6f}")

    torch.save(
        {
            "unet": unet.state_dict(),
            "history": history,
            "config": asdict(CFG),
            "personal": {
                "SID4": SID4,
                "SEED": SEED,
                "SLICE": SLICE,
                "HP_ID": HP_ID,
                "CLS_A": CLS_A,
                "CLS_B": CLS_B,
            },
        },
        ckpt_path,
    )
    log(f"Wrote checkpoint {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["loss"], color="#1f4e79")
    ax.set_xlabel("epoch")
    ax.set_ylabel("noise-prediction MSE")
    ax.set_title("HW9 LDM training loss (FashionMNIST latents)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_train_loss.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Sampling with classifier-free guidance
    # ------------------------------------------------------------------
    log(f"Section 5: sampling 64 images with CFG w={CFG.guidance_w}")
    unet.eval()
    set_seed(SEED)
    labels_64 = torch.arange(64, device=device) % 10
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    t_samp = time.time()
    z_samp = p_sample_cfg(
        unet,
        schedule,
        shape=(64, CFG.latent_channels, CFG.latent_size, CFG.latent_size),
        labels=labels_64,
        w=CFG.guidance_w,
        device=device,
        generator=gen,
    )
    with torch.no_grad():
        imgs = decode_latents(vae, z_samp, CFG.vae_scale)
    metrics["sample_seconds"] = time.time() - t_samp
    log(f"Sampling+decode took {metrics['sample_seconds']:.1f}s")

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_display(imgs[i]).cpu().squeeze(), cmap="gray")
        ax.set_title(f"{int(labels_64[i])} {CLASS_NAMES[int(labels_64[i])][:8]}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"CFG samples w={CFG.guidance_w}, labels 0-9 repeated, seed={SEED}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_cfg_grid_64.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    save_image(to_display(imgs).cpu(), FIG_DIR / "04_cfg_grid_64_raw.png", nrow=8, padding=2)

    # Extra: two rows of CLS_A / CLS_B for the standing focus classes
    set_seed(SEED + 1)
    focus = torch.tensor([CLS_A] * 8 + [CLS_B] * 8, device=device)
    gen.manual_seed(SEED + 1)
    z_focus = p_sample_cfg(
        unet,
        schedule,
        shape=(16, CFG.latent_channels, CFG.latent_size, CFG.latent_size),
        labels=focus,
        w=CFG.guidance_w,
        device=device,
        generator=gen,
    )
    with torch.no_grad():
        focus_imgs = decode_latents(vae, z_focus, CFG.vae_scale)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for i in range(8):
        axes[0, i].imshow(to_display(focus_imgs[i]).cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(to_display(focus_imgs[i + 8]).cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel(f"CLS_A={CLS_A}\n{CLASS_NAMES[CLS_A]}", fontsize=8)
    axes[1, 0].set_ylabel(f"CLS_B={CLS_B}\n{CLASS_NAMES[CLS_B]}", fontsize=8)
    fig.suptitle("Focus-class CFG samples")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_focus_classes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Unconditional vs conditional comparison (small)
    set_seed(SEED + 2)
    y_cond = torch.arange(10, device=device)
    gen.manual_seed(SEED + 2)
    z_cond = p_sample_cfg(unet, schedule, (10, 4, 16, 16), y_cond, CFG.guidance_w, device, gen)
    z_uncond = p_sample_cfg(unet, schedule, (10, 4, 16, 16), torch.full((10,), NULL_LABEL, device=device), 0.0, device, gen)
    with torch.no_grad():
        cond_imgs = decode_latents(vae, z_cond, CFG.vae_scale)
        uncond_imgs = decode_latents(vae, z_uncond, CFG.vae_scale)
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
    for i in range(10):
        axes[0, i].imshow(to_display(cond_imgs[i]).cpu().squeeze(), cmap="gray")
        axes[0, i].set_title(CLASS_NAMES[i][:8], fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(to_display(uncond_imgs[i]).cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("cond w=3")
    axes[1, 0].set_ylabel("uncond")
    fig.suptitle("Conditional (top) vs unconditional (bottom) DDPM samples")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_cond_vs_uncond.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    metrics["finished"] = now_utc()
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ART_DIR / "loss_history.json").write_text(json.dumps(history), encoding="utf-8")
    log(f"Metrics: {json.dumps(metrics, indent=2)}")
    return metrics, history


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    main()
