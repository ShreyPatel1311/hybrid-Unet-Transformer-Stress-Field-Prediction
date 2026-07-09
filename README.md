# Hybrid U-Net + Transformer for Stress Field Prediction

A hybrid CNN–Transformer autoencoder that predicts 2D stress/strain fields directly from part geometry images, trained to approximate FEA (finite element analysis) simulation results.

Given a rendered geometry image, the model outputs a per-pixel field map (e.g. Mises stress, S11/S12, PE11/PE12) — a fast, differentiable surrogate for traditional FEM solvers.

## Model

`ViTAutoencoder` — a U-Net-style encoder/decoder with a Transformer bottleneck:

- **Encoder**: convolutional stem + 3 stages of residual blocks, downsampling `128×128 → 8×8`, producing skip connections at 64×64, 32×32, and 16×16.
- **Bottleneck**: 6-layer Transformer (8 heads, dim 256) over the 8×8 feature grid, with 2D sinusoidal positional embeddings.
- **Decoder**: bilinear upsampling + skip-connection fusion (ResBlocks) back to `128×128`, with auxiliary prediction heads at 32×32 and 64×64 for multi-scale supervision.

~17.0M parameters total.

**Loss**: `(1 − SSIM) + L1` on the full-resolution output, plus weighted auxiliary losses at the 64×64 and 32×32 decoder heads.

## Dataset

Trained on [`Godseye1311/geometry-stress-strain-fea`](https://huggingface.co/datasets/Godseye1311/geometry-stress-strain-fea), a set of paired geometry/field PNGs (left half = input geometry, right half = target field) across 10 subdatasets:

`BC_scale`, `FIELD2GEO`, `HEXAGON`, `MISES`, `PE11_0.2`, `PE12`, `S11`, `S12`, `STITCH(GEO_MISES)`, `TRIANGLE`

~16k training images and ~4k test images in total, resized to 128×128.

## Results

Trained for 25 epochs (AdamW, lr 1e-4, batch size 4) on all 10 subdatasets combined:

| Metric | Value |
|---|---|
| Val loss | 0.2375 |
| Val SSIM | 0.910 |
| Val R² | 0.908 |

## Repo contents

- [`Main.ipynb`](Main.ipynb) — end-to-end notebook: dataset download, data pipeline, model definition, training loop, evaluation, and Grad-CAM visualization of what the model attends to when predicting a field.
- `stress_vit_ALL.pth` — trained weights (best checkpoint by validation loss).

## Usage

The notebook is designed to run top-to-bottom on Colab/Jupyter with a CUDA GPU:

1. Install dependencies (`torchmetrics`, `grad-cam`).
2. Clone the dataset from Hugging Face into `/content/dataset`.
3. Run the training cells, or load `stress_vit_ALL.pth` directly into `ViTAutoencoder` for inference.
4. The Grad-CAM section visualizes encoder/decoder attention for a chosen subdataset and layer.
