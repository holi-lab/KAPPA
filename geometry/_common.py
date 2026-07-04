#!/usr/bin/env python3
"""Shared plumbing for the geometry analysis: name maps, read-only path manager,
public-config iteration, and result-directory helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import kappa_core
from kappa_core.data_path import PathManager
from kappa_core.exp_config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    ProbeConfig,
    SteeringConfig,
)
from kappa_core.geometry import PairingLogicError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_OUTPUT_ROOT = Path(kappa_core.__file__).resolve().parent / "outputs"

# Model key (model_name_model_size) -> configs/<dir>
MODEL_DIR_MAP = {
    "Llama-3.1_8B": "llama3.1-8B",
    "Qwen2.5_7B": "qwen2.5-7B",
}

DEFAULT_PLOT_MODEL_KEYS = ("Llama-3.1_8B", "Qwen2.5_7B")

# Public config save_names for the three largest-gap datasets used in Fig 8/9.
DEFAULT_PLOT_DATASET_KEYS = ("truthfulqa", "gsm8k-mc", "bbh_algo")

# All eight benchmarks used in the Fig 10 correlation.
DATASET_CORRELATION_DATASET_KEYS = (
    "arc-challenge",
    "gsm8k-mc",
    "bbh_algo",
    "bbh_nlp",
    "mmlu_stem",
    "truthfulqa",
    "bbq_age",
    "bbq_religion",
)

MODEL_NAME_MAP = {
    "Llama-3.1_8B": "Llama 3.1 8B",
    "Qwen2.5_7B": "Qwen 2.5 7B",
}

# Keyed on public config save_names (configs/<model>/*/<dataset>.yaml).
DATASET_NAME_MAP = {
    "arc-challenge": "ARC",
    "gsm8k-mc": "GSM8k",
    "bbh_algo": "BBH-Algo",
    "bbh_nlp": "BBH-NLP",
    "mmlu_stem": "MMLU-STEM",
    "mmlu_humanities": "MMLU-Humanities",
    "mmlu_social_sciences": "MMLU-Social",
    "pubmedqa": "PubMedQA",
    "truthfulqa": "TruthfulQA",
    "bbq_age": "BBQ-Age",
    "bbq_religion": "BBQ-Religion",
}

DISPLAY_TO_MODEL = {display: raw for raw, display in MODEL_NAME_MAP.items()}
DISPLAY_TO_SAVE_NAME = {display: raw for raw, display in DATASET_NAME_MAP.items()}


class ExactPathValidationError(PairingLogicError):
    """Raised when the config-derived path contract cannot be satisfied."""


class ReadOnlyPathManager(PathManager):
    """Reuse PathManager naming logic without creating directories."""

    def setup_configs(
        self,
        dataset_config: DatasetConfig = None,
        generation_config: GenerationConfig = None,
        probe_config: Optional[ProbeConfig] = None,
        steering_config: Optional[SteeringConfig] = None,
        icl_config: Optional[object] = None,
        finetuning_config: Optional[object] = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.generation_config = generation_config
        self.probe_config = probe_config
        self.steering_config = steering_config
        self.icl_config = icl_config
        self.finetuning_config = finetuning_config

    def _make_path(self, *parts: str, mkdir: bool = True) -> Path:
        return self.base_path.joinpath(*parts)


def iter_kappa_config_paths(
    config_dir: Path,
    model_keys: Optional[Sequence[str]],
) -> Iterator[Tuple[str, Path]]:
    selected = model_keys or MODEL_DIR_MAP.keys()
    for model_key in selected:
        if model_key not in MODEL_DIR_MAP:
            raise ExactPathValidationError(
                f"Unknown model key '{model_key}'. Known keys: {sorted(MODEL_DIR_MAP)}"
            )
        kappa_dir = Path(config_dir) / MODEL_DIR_MAP[model_key] / "KAPPA"
        if not kappa_dir.exists():
            continue
        for yaml_path in sorted(kappa_dir.glob("*.yaml")):
            yield model_key, yaml_path


def model_key_from_cfg(cfg: ExperimentConfig) -> str:
    llm = cfg.llm_config
    if llm is None:
        raise ExactPathValidationError("Config is missing llm_config.")
    return f"{llm.model_name}_{llm.model_size}"


def extract_probe_save_name(probe_exp_config: Optional[ExperimentConfig], context: str) -> str:
    if probe_exp_config is None:
        raise ExactPathValidationError(f"{context}: config is missing probe_exp_config.")
    probe_list = probe_exp_config.probe_config_list or []
    save_names = {probe_cfg.save_name for probe_cfg in probe_list}
    if not save_names:
        raise ExactPathValidationError(f"{context}: probe_exp_config.probe_config_list is empty.")
    if len(save_names) != 1:
        raise ExactPathValidationError(
            f"{context}: expected exactly one probe save_name, found {sorted(save_names)}."
        )
    return next(iter(save_names))


def result_dirs(base: Path) -> Dict[str, Path]:
    base = Path(base)
    dirs = {
        "root": base,
        "data": base / "data",
        "plots": base / "plots",
        "tables": base / "tables",
        "reports": base / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
