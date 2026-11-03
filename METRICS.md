# HW9 metrics

Personal parameters from SJSU ID `019129357`:

| Parameter | Value | Notes |
|---|---:|---|
| SID4 | 9357 | last four digits |
| SEED | 9357 | Python / NumPy / PyTorch |
| SLICE | 357 | visualization start index |
| HP_ID | 3 | Table 0.1 = Learning-rate-low; **HW9 has no mapping, unused** |
| CLS_A | 7 | Sneaker |
| CLS_B | 6 | Shirt |

HW9 specifies a single diffusion training run (50 epochs, lr \(2\times 10^{-5}\)) plus a pretrained VAE. No second hyperparameter model.

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (32 GB), CUDA 12.8, PyTorch 2.11.0+cu128
- VAE: `stabilityai/sd-vae-ft-mse`, `scaling_factor = 0.18215`
- U-Net: 14,828,484 parameters, 4×16×16 latent input

## 1. VAE reconstruction

Measured on Fashion-MNIST train indices `SLICE .. SLICE+7` = 357–364.

| Quantity | Value |
|---|---:|
| Latent shape | `(8, 4, 16, 16)` |
| MSE, 128×128 RGB in \([-1,1]\) | 0.000813 |
| MSE, 28×28 grayscale in \([0,1]\) | 0.001013 |
| Scaled latent mean (full train) | -0.0451 |
| Scaled latent std (full train) | 1.260 |

Reconstructions preserve class identity (sneaker, bag, trouser, ankle boot, plaid shirt, coat, sandal). The pretrained SD VAE is not the quality bottleneck on this dataset.

## 2. Forward diffusion

Linear \(\beta_t \in [10^{-4}, 0.02]\), \(T=1000\). Decoded \(z_t\) at \(t\in\{0,200,500,800,1000\}\):

- \(t=0\): matches the reconstruction
- \(t=200\): structure still readable
- \(t=500\): class identity mostly gone
- \(t=800,1000\): unstructured fields, as \(\bar\alpha_t \to 0\)

## 3. Training (noise prediction)

| Quantity | Value |
|---|---:|
| Epochs | 50 |
| Learning rate | \(2\times 10^{-5}\) |
| Batch size | 256 |
| Label-drop probability | 0.1 |
| Epoch 1 MSE | 0.5006 |
| Epoch 5 MSE | 0.1546 |
| Epoch 25 MSE | 0.1119 |
| Epoch 50 MSE | 0.1033 |
| Wall time | 426 s (7.10 min) |

Loss dropped sharply in the first five epochs and then flattened near 0.10. No second model was trained.

## 4. Sampling

Classifier-free guidance \(\varepsilon=(1+w)\varepsilon_{\mathrm{cond}}-w\varepsilon_{\mathrm{uncond}}\) with \(w=3\), 1000-step DDPM.

| Quantity | Value |
|---|---:|
| 64-image grid | labels \(0\)–\(9\) repeated (`i % 10`), seed 9357 |
| 64-grid sample + decode time | 11.6 s |
| CLS_A / CLS_B | sneakers consistent; shirts overlap coats |
| Unconditional (\(w=0\), null label) | mixed classes, as expected |

Checkpoint: `checkpoints/ldm_unet.pt`  
Training log: `RUN_LOG.txt`  
Figures: `figures/`
