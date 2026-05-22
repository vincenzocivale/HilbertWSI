# Vendored 2DMambaMIL reference code

The four Python files in this directory are **lightly patched copies** of the
official 2DMambaMIL implementation, kept for provenance.

Source:
- `MambaMIL_2D.py` — 2DMambaMIL model class.
- `mamba_simple.py` — Mamba / MambaConfig with 2D-scan support.
- `pscan.py` — parallel scan (1D + 2D, pure-PyTorch autograd).
- `pscan_2d.py` — CUDA-kernel selective-scan binding (only used when
  `MambaConfig.use_cuda=True` AND `mamba_2d=True`).

Upstream paths (in `AtlasAnalyticsLab/2DMamba`):
- `2DMambaMIL/models/MambaMIL_2D.py`
- `2DMambaMIL/models/mamba_simple.py`
- `2DMambaMIL/models/pscan.py`
- `2DMambaMIL/models/pscan_2d.py`

Paper: Zhang et al., *2DMamba: Efficient State Space Model for Image
Representation with Applications on Giga-Pixel Whole Slide Image
Classification*, CVPR 2025 (arXiv 2412.00678).

## Patches applied

The vendored files differ from upstream **only** in their import paths
(`from models.X import ...` → `from .X import ...`) so the package can live
under `hilbert_wsi.models.vendor.twodmamba`. No logic changes.

## Runtime baseline lives elsewhere

The Patho-Bench-compatible encoder used in the killer experiment is in
`hilbert_wsi/models/twodmamba.py`. It wraps `MambaMIL_2D`, swaps the
classifier head for `nn.Identity` so the network returns a slide embedding
(per the Trident slide-encoder contract), and exposes an `nn.Module`-style
config instead of the upstream argparse `args` namespace.

## Conda env

The vendored code runs end-to-end **without** rebuilding the CUDA kernel
provided we keep `use_cuda=False` (pure-Python parallel scan, the upstream
default for MIL experiments). Therefore the existing `hilbert-wsi-clean`
env is sufficient: no new conda env required.

Upstream `cuda_kernel/`, `v2dmamba_scan/`, and `2DVMamba/` (the ImageNet
classification / MMDetection / MMSegmentation reference) are **not vendored**
as they are unrelated to the slide-level MIL pipeline.
