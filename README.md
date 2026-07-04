# KAPPA

[![arXiv](https://img.shields.io/badge/arXiv-2509.23782-b31b1b.svg)](https://arxiv.org/abs/2509.23782)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://arxiv.org/abs/2509.23782)

Code for the ICML 2026 accepted paper:

**Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions**

Yoonah Park\*,
Haesung Pyun\*,
Yohan Jo

📄 [Paper](https://arxiv.org/abs/2509.23782)

## Overview

Large language models (LLMs) often fail on multiple-choice questions (MCQs) even when their hidden representations encode the correct answer, revealing a misalignment between internal knowledge and output behavior: the **knowledge-prediction gap**.

This repository provides code to (1) probe the residual stream for distinct knowledge and prediction subspaces, and (2) mitigate the gap with **KAPPA** (Knowledge-Aligned Prediction through Projection-based Adjustment), a lightweight inference-time intervention. KAPPA extracts the residual-stream activation at a given layer, locates its coordinates within the knowledge and prediction subspaces identified by two linear probes, and minimally adjusts the hidden state within the prediction subspace so it aligns with the knowledge subspace.


## Scripts

This repository is organized around three main scripts, run in sequence.

### 1. `run_activation_collection.py`

Collects and saves hidden-state activations from LLMs.

Given an activation config, this script runs a model on MCQ data, extracts residual-stream and related hidden representations at the configured layers and token positions, and saves the activations and metadata for later probing.

```bash
python run_activation_collection.py --config configs/llama3.1-8B/activation/bbq_religion.yaml
```

### 2. `run_probe.py`

Trains probing models for knowledge and prediction.

This script loads saved activations and trains probes that estimate whether the model's hidden representation contains the correct answer knowledge and/or supports the model's actual prediction. These probes are used to analyze the knowledge-prediction gap and to build the KAPPA intervention.

```bash
python run_probe.py --config configs/llama3.1-8B/probe/bbq_religion_lr_2e-4_know.yaml
python run_probe.py --config configs/llama3.1-8B/probe/bbq_religion_lr_2e-4_pred.yaml
```

### 3. `run_KAPPA.py`

Runs the KAPPA intervention.

This script applies the proposed inference-time intervention using the trained knowledge and prediction probes. KAPPA adjusts the residual stream to better align the model's internal knowledge with its output behavior, reducing the knowledge-prediction gap on MCQs.

```bash
python run_KAPPA.py --config configs/llama3.1-8B/KAPPA/bbq_religion.yaml
```

## Geometric Analysis (Figures 8–10)

After collecting activations and training the knowledge/prediction probes (Steps 1–2),
these scripts reproduce the paper's geometric analysis of the two subspaces (Appendix B).
They read the public `configs/<model>/KAPPA/*.yaml` and the outputs under `kappa_core/outputs/`.

```bash
python geometry/principal_angles.py          # Figure 8: mean principal angles (knowledge↔prediction, prediction↔W_U)
python geometry/cka.py                        # Figure 9: orthogonal linear CKA (knowledge↔prediction)
python geometry/geometry_gap_correlation.py   # Figure 10: subspace angle vs the knowledge-prediction gap
```

Figures and tables are written under `geometry/results/` (git-ignored).

## Repository Structure

```text
KAPPA/
├── run_activation_collection.py   # Step 1: collect hidden-state activations
├── run_probe.py                   # Step 2: train knowledge / prediction probes
├── run_KAPPA.py                   # Step 3: run the KAPPA intervention
├── kappa_core/                    # Shared implementation used by all three scripts
│   ├── activation_loader.py       # Load collected activations for probing / intervention
│   ├── collector.py               # Activation collection utilities
│   ├── data.py                    # Dataset loading and MCQ example handling
│   ├── data_path.py               # Dataset path helpers
│   ├── exp_config.py              # Experiment config parsing
│   └── geometry.py                # Subspace-geometry math (principal angles, CKA)
├── geometry/                      # Reproduce the geometric analysis (Figures 8–10)
│   ├── principal_angles.py        # Figure 8: principal-angle analysis
│   ├── cka.py                     # Figure 9: orthogonal linear CKA analysis
│   ├── geometry_gap_correlation.py # Figure 10: geometry-gap correlation analysis
│   ├── _common.py                 # Shared geometry-analysis helpers
│   └── _manifest.py               # Config / output manifest discovery
├── configs/                       # Example experiment configs
│   ├── llama3.1-8B/               # Llama-3.1-8B configs
│   │   ├── activation/            # Activation collection configs
│   │   ├── probe/                 # Knowledge / prediction probe configs
│   │   └── KAPPA/                 # KAPPA intervention configs
│   └── qwen2.5-7B/                # Qwen2.5-7B configs
│       ├── activation/
│       ├── probe/
│       └── KAPPA/
├── data/                          # MCQ datasets
└── prompts/                       # Prompt templates and dataset metadata
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{park2026bridging,
  title     = {Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions},
  author    = {Park, Yoonah and Pyun, Haesung and Jo, Yohan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Uvutpgvpmi}
}
```

## License

This project is released under the [MIT License](LICENSE).