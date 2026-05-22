# Vendored upstream reference code

The two files in this directory are **unmodified copies** of the official
MambaMIL and TransMIL implementations, kept for reference and provenance.

- `MambaMIL.py` — `MambaMIL: Enhancing Long Sequence Modeling with Sequence
  Reordering in Computational Pathology` (Yang et al., MICCAI 2024).
  Source: https://github.com/isyangshu/MambaMIL/blob/main/models/MambaMIL.py

- `TransMIL.py` — `TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification` (Shao et al., NeurIPS 2021).
  Source: https://github.com/isyangshu/MambaMIL/blob/main/models/TransMIL.py
  (originally from https://github.com/szc19990412/TransMIL)

The runtime baselines used in HilbertWSI's killer experiment live in
`hilbert_wsi/models/mambamil.py` and `hilbert_wsi/models/transmil.py`. Those
are clean re-implementations that follow the original architectures but:

1. Use the kernels we already depend on (`mamba_ssm.Mamba2`,
   `nystrom_attention.NystromAttention`) instead of the upstream custom
   `mamba` fork / inlined Nystrom attention.
2. Return a slide embedding instead of classifier logits, as required by the
   Patho-Bench / Trident slide-encoder contract.
3. Are device-agnostic (no hardcoded `.cuda()` calls).

Upstream repos have no LICENSE file; their code is used here only as an
architectural reference. Please cite the original MICCAI 2024 / NeurIPS 2021
papers when reporting results from these baselines.
