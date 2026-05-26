"""No-ordering and classical-MIL baseline models.

These are kept for baseline comparison experiments but are not part of
the core 1D-ordering vs 2D contribution experiments.

Models:
    transmil   — TransMIL (Shao et al., NeurIPS 2021), Nystromformer + PPEG, no ordering
    mambamil   — MambaMIL (Yang et al., MICCAI 2024), Mamba2 + attn pool, no ordering
    clam       — CLAM-SB/MB (Lu et al., Nature Biomed Eng 2021), gated attention MIL
    dsmil      — DSMIL (Li et al., CVPR 2021), critical-instance bag attention
    twodmamba  — 2DMambaMIL (2D-native SSM, no 1D ordering)
    vendor/    — pristine upstream code for provenance
"""
