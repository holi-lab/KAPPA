# Geometric Analysis (Figures 8–10)

Reproduces the paper's analysis of the knowledge and prediction subspaces (Appendix B).

Run each script from the repository root, e.g. `python geometry/principal_angles.py`
(see the top-level README's "Geometric Analysis" section for the full command list).

| Script | Figure | Produces |
|---|---|---|
| `principal_angles.py` | Figure 8 | Layer-wise mean principal angles (knowledge↔prediction, prediction↔`W_U`) with random baseline and Top-1/3/6 intervention markers |
| `cka.py` | Figure 9 | Layer-wise orthogonal linear CKA (knowledge↔prediction) with random-baseline controls |
| `geometry_gap_correlation.py` | Figure 10 | Cross-benchmark scatter of mean principal angle vs the `1−AGR` gap (Spearman ρ) |

## Prerequisites

Run Steps 1–2 first (`run_activation_collection.py`, `run_probe.py`) so probe weights and
base-model metadata exist under `kappa_core/outputs/`. The subspace math lives in
`kappa_core/geometry.py`.

## Common options

`--models` (default: Llama-3.1 8B, Qwen 2.5 7B) · `--datasets` · `--config-dir` (default
`configs/`) · `--output-root` (default `kappa_core/outputs/`) · `--out-dir`
(default `geometry/results/<analysis>/`) · `--allow-partial` · `--skip-wu`.
