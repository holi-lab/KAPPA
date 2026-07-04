#!/usr/bin/env python3
"""Config-driven per-(model,dataset) bundles + ACC/AGR/KLD recompute for the geometry figures."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from kappa_core.activation_loader import ProbeActivationLoader
from kappa_core.exp_config import (
    ExperimentConfig,
    GenerationConfig,
    ProbeConfig,
)
from kappa_core.geometry import (
    MODEL_CONFIGS,
    PairingLogicError,
    center_weights,
    compute_principal_angles,
    get_random_baseline,
    linear_centered_cka,
    load_W_U_for_options,
    load_probe_weights,
    _orthonormal_row_space_basis,
    _random_baseline_cka,
    RANDOM_BASELINE_SAMPLES,
    RANDOM_BASELINE_SEED,
)
from geometry._common import (
    DATASET_NAME_MAP,
    DEFAULT_CONFIG_DIR,
    DEFAULT_OUTPUT_ROOT,
    ExactPathValidationError,
    ReadOnlyPathManager,
    extract_probe_save_name,
    iter_kappa_config_paths,
    model_key_from_cfg,
)


LOG = logging.getLogger("geometry.manifest")

_ITEMS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_INDEX_CACHE: Dict[str, Dict[Tuple[int, int, str], Dict[str, Any]]] = {}

ActivationView = Dict[str, Any]

# Public repo option-symbol table used to reconstruct option_info.
OPTION_TYPES_YAML = DEFAULT_CONFIG_DIR.parent / "prompts" / "option_types.yaml"
_OPTION_TYPES_CACHE: Optional[Dict[str, List[str]]] = None
PROBE_LAYER_DIR_RE = re.compile(r"res_layer_(?P<layer>-?\d+)_token_(?P<token>.+)")


# ---------------------------------------------------------------------------
# Normalization helpers (verbatim from the original geometry analysis code).
# ---------------------------------------------------------------------------
def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return stripped
        try:
            if "." in stripped or "e" in stripped.lower():
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped
    return value


def _token_key(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, str):
        return tuple(_normalize_scalar(part) for part in value.split("_"))
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return _token_key(value[0])
        return tuple(_normalize_scalar(part) for part in value)
    if value is None:
        return tuple()
    return (_normalize_scalar(value),)


def _token_text(value: Any) -> str:
    parts = _token_key(value)
    return "_".join(str(part) for part in parts)


def _normalize_generation_cfg(generation_cfg: Any) -> Dict[str, Any]:
    if isinstance(generation_cfg, dict):
        data = generation_cfg
    else:
        data = {
            "decoding_mode": getattr(generation_cfg, "decoding_mode", None),
            "polarity": getattr(generation_cfg, "polarity", None),
            "is_positive": getattr(generation_cfg, "is_positive", None),
            "temperature": getattr(generation_cfg, "temperature", None),
            "top_p": getattr(generation_cfg, "top_p", None),
            "do_sample": getattr(generation_cfg, "do_sample", None),
            "max_new_tokens": getattr(generation_cfg, "max_new_tokens", None),
        }
    return {
        "decoding_mode": data.get("decoding_mode"),
        "polarity": data.get("polarity"),
        "is_positive": data.get("is_positive"),
        "temperature": data.get("temperature"),
        "top_p": data.get("top_p"),
        "do_sample": data.get("do_sample"),
        "max_new_tokens": data.get("max_new_tokens"),
    }


def _normalize_probe_entry(probe_entry: Any) -> Dict[str, Any]:
    if isinstance(probe_entry, dict):
        training_cfg = probe_entry.get("training_config") or {}
        data = probe_entry
    else:
        training_cfg = probe_entry.training_config or {}
        data = {
            "method": probe_entry.method,
            "model_name": probe_entry.model_name,
            "objective": probe_entry.objective,
            "component": probe_entry.component,
            "save_name": probe_entry.save_name,
            "layer": probe_entry.layer,
            "token_positions": probe_entry.token_positions,
        }
    return {
        "method": data.get("method"),
        "model_name": data.get("model_name"),
        "objective": data.get("objective"),
        "component": data.get("component"),
        "save_name": data.get("save_name"),
        "layer": _normalize_scalar(data.get("layer")),
        "token_positions": _token_text(data.get("token_positions")),
        "training_config": {
            "learning_rate": _normalize_scalar(getattr(training_cfg, "learning_rate", None) if not isinstance(training_cfg, dict) else training_cfg.get("learning_rate")),
            "batch_size": _normalize_scalar(getattr(training_cfg, "batch_size", None) if not isinstance(training_cfg, dict) else training_cfg.get("batch_size")),
            "epochs": _normalize_scalar(getattr(training_cfg, "epochs", None) if not isinstance(training_cfg, dict) else training_cfg.get("epochs")),
            "loss_function": getattr(training_cfg, "loss_function", None) if not isinstance(training_cfg, dict) else training_cfg.get("loss_function"),
            "metric": getattr(training_cfg, "metric", None) if not isinstance(training_cfg, dict) else training_cfg.get("metric"),
        },
    }


def _probe_group_key(probe_cfg: ProbeConfig) -> Tuple[Any, ...]:
    norm = _normalize_probe_entry(probe_cfg)
    train = norm["training_config"]
    return (
        norm["save_name"],
        norm["method"],
        norm["model_name"],
        norm["component"],
        norm["layer"],
        norm["token_positions"],
        train["learning_rate"],
        train["batch_size"],
        train["epochs"],
    )


def _probe_template_key(probe_cfg: ProbeConfig) -> Tuple[Any, ...]:
    norm = _normalize_probe_entry(probe_cfg)
    train = norm["training_config"]
    return (
        norm["save_name"],
        norm["method"],
        norm["model_name"],
        norm["component"],
        norm["token_positions"],
        train["learning_rate"],
        train["batch_size"],
        train["epochs"],
    )


# ---------------------------------------------------------------------------
# Option info (verbatim from the original geometry analysis code).
# ---------------------------------------------------------------------------
def _load_option_types_yaml() -> Dict[str, List[str]]:
    global _OPTION_TYPES_CACHE
    if _OPTION_TYPES_CACHE is None:
        with open(OPTION_TYPES_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _OPTION_TYPES_CACHE = data.get("option_symbol", {})
    return _OPTION_TYPES_CACHE


def _option_info_from_dataset_configs(ds_cfgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    symbol_names = {
        cfg.get("prompt_config", {}).get("option_symbol")
        for cfg in ds_cfgs
        if cfg.get("prompt_config", {}).get("option_symbol")
    }
    k_values = {cfg.get("num_option") for cfg in ds_cfgs if cfg.get("num_option") is not None}
    if not k_values:
        raise ExactPathValidationError("Canonical dataset config is missing num_option.")
    if len(k_values) != 1:
        raise ExactPathValidationError(f"Canonical dataset config has inconsistent num_option values: {sorted(k_values)}")

    k = int(next(iter(k_values)))
    option_map = _load_option_types_yaml()
    mixed_symbols = len(symbol_names) > 1

    if mixed_symbols:
        symbol_sets = {}
        for symbol_name in sorted(symbol_names):
            chars = option_map.get(symbol_name, [])
            symbol_sets[symbol_name] = [str(option) for option in chars[:k]]
        primary_name = sorted(symbol_names)[0]
        return {
            "k": k,
            "option_symbol_name": ", ".join(sorted(symbol_names)),
            "options": symbol_sets.get(primary_name, []),
            "mixed_symbols": True,
            "all_symbol_sets": symbol_sets,
        }

    symbol_name = next(iter(symbol_names)) if symbol_names else None
    if symbol_name is None:
        raise ExactPathValidationError("Canonical dataset config is missing prompt_config.option_symbol.")

    chars = option_map.get(symbol_name, [])
    if not chars:
        raise ExactPathValidationError(f"Unknown option_symbol '{symbol_name}' in {OPTION_TYPES_YAML}.")

    return {
        "k": k,
        "option_symbol_name": symbol_name,
        "options": [str(option) for option in chars[:k]],
        "mixed_symbols": False,
    }


# ---------------------------------------------------------------------------
# Probe-pair building (verbatim from the original geometry analysis code).
# ---------------------------------------------------------------------------
def _build_probe_pairs(
    cfg: ExperimentConfig,
    output_root: Path,
    yaml_path: Path,
) -> List[Dict[str, Any]]:
    if cfg.probe_exp_config is None:
        raise ExactPathValidationError(f"{yaml_path}: steering YAML is missing probe_exp_config.")

    probe_exp = cfg.probe_exp_config
    activation_exp = probe_exp.activation_exp_config
    if activation_exp is None:
        raise ExactPathValidationError(f"{yaml_path}: nested probe_exp_config is missing activation_exp_config.")

    generation_cfgs = activation_exp.generation_config_list or []
    if len(generation_cfgs) != 1:
        raise ExactPathValidationError(
            f"{yaml_path}: expected exactly one generation_config in nested probe_exp_config, "
            f"found {len(generation_cfgs)}."
        )
    generation_cfg = generation_cfgs[0]

    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for probe_cfg in probe_exp.probe_config_list or []:
        key = _probe_group_key(probe_cfg)
        grouped.setdefault(key, {})
        grouped[key][probe_cfg.objective] = probe_cfg

    probe_pm = ReadOnlyPathManager(probe_exp)
    probe_pm.base_path = output_root

    pairs: List[Dict[str, Any]] = []
    for objective_map in grouped.values():
        if set(objective_map.keys()) != {"answer", "pred"}:
            continue

        answer_cfg = objective_map["answer"]
        pred_cfg = objective_map["pred"]

        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=answer_cfg)
        answer_model_path = probe_pm.get_probe_output(
            layer=answer_cfg.layer,
            token_position=answer_cfg.token_positions,
            output_type="model",
        )
        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=answer_cfg)
        answer_metadata_path = probe_pm.get_probe_output(
            layer=answer_cfg.layer,
            token_position=answer_cfg.token_positions,
            output_type="metadata",
        )
        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=pred_cfg)
        pred_model_path = probe_pm.get_probe_output(
            layer=pred_cfg.layer,
            token_position=pred_cfg.token_positions,
            output_type="model",
        )
        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=pred_cfg)
        pred_metadata_path = probe_pm.get_probe_output(
            layer=pred_cfg.layer,
            token_position=pred_cfg.token_positions,
            output_type="metadata",
        )

        pairs.append({
            "source_yaml": str(yaml_path),
            "layer": int(_normalize_probe_entry(answer_cfg)["layer"]),
            "ans_path": str(answer_model_path),
            "ans_metadata_path": str(answer_metadata_path),
            "pred_path": str(pred_model_path),
            "pred_metadata_path": str(pred_metadata_path),
            "token_positions": _normalize_probe_entry(answer_cfg)["token_positions"],
        })

    return pairs


def _build_steering_runs(
    cfg: ExperimentConfig,
    output_root: Path,
    yaml_path: Path,
) -> List[Dict[str, Any]]:
    if not cfg.steering_config_list:
        raise ExactPathValidationError(f"{yaml_path}: steering YAML is missing steering_config entries.")

    steering_pm = ReadOnlyPathManager(cfg)
    steering_pm.base_path = output_root

    runs: List[Dict[str, Any]] = []
    for steering_cfg, dataset_cfg, generation_cfg in product(
        cfg.steering_config_list or [],
        cfg.dataset_config_list or [],
        cfg.generation_config_list or [],
    ):
        steering_pm.setup_configs(
            steering_config=steering_cfg,
            dataset_config=dataset_cfg,
            generation_config=generation_cfg,
        )
        metadata_path = steering_pm.get_steering_output("metadata")
        runs.append({
            "source_yaml": str(yaml_path),
            "source_subdir": yaml_path.parent.name,
            "dataset_name": dataset_cfg.dataset,
            "dataset_split": dataset_cfg.split,
            "metadata_path": str(metadata_path),
            "intervention_layers": list(steering_cfg.intervention_layers.layers),
            "multiplier": steering_cfg.multiplier,
            "w": steering_cfg.w,
            "beta": steering_cfg.beta,
            "component": steering_cfg.component,
        })

    return runs


def _build_base_runs(
    cfg: ExperimentConfig,
    output_root: Path,
    yaml_path: Path,
) -> List[Dict[str, Any]]:
    activation_exp = cfg.activation_exp_config
    if activation_exp is None:
        raise ExactPathValidationError(f"{yaml_path}: steering YAML is missing activation_exp_config.")

    generation_cfgs = activation_exp.generation_config_list or []
    dataset_cfgs = activation_exp.dataset_config_list or []
    if not generation_cfgs:
        raise ExactPathValidationError(f"{yaml_path}: activation_exp_config has no generation_config entries.")
    if not dataset_cfgs:
        raise ExactPathValidationError(f"{yaml_path}: activation_exp_config has no dataset_config entries.")

    activation_pm = ReadOnlyPathManager(activation_exp)
    activation_pm.base_path = output_root

    runs: List[Dict[str, Any]] = []
    for dataset_cfg, generation_cfg in product(dataset_cfgs, generation_cfgs):
        if getattr(dataset_cfg, "split", None) != "test":
            continue
        activation_pm.setup_configs(
            dataset_config=dataset_cfg,
            generation_config=generation_cfg,
        )
        metadata_path = activation_pm.get_activation_output("metadata")
        runs.append({
            "source_yaml": str(yaml_path),
            "dataset_name": dataset_cfg.dataset,
            "dataset_split": dataset_cfg.split,
            "metadata_path": str(metadata_path),
        })

    if not runs:
        raise ExactPathValidationError(
            f"{yaml_path}: activation_exp_config did not yield any test activation metadata path."
        )

    return runs


def _parse_probe_layer_dir(path: Path, expected_token_positions: str, context: str) -> int:
    match = PROBE_LAYER_DIR_RE.fullmatch(path.name)
    if not match:
        raise ExactPathValidationError(f"{context}: could not parse probe layer directory name: {path.name}")
    observed_token = match.group("token")
    if observed_token != expected_token_positions:
        raise ExactPathValidationError(
            f"{context}: token-position mismatch while scanning sibling probe dirs. "
            f"Expected token={expected_token_positions}, observed token={observed_token} in {path}"
        )
    return int(match.group("layer"))


def _discover_available_probe_layers(
    probe_pm: ReadOnlyPathManager,
    generation_cfg: GenerationConfig,
    answer_cfg: ProbeConfig,
    pred_cfg: ProbeConfig,
    requested_layers: Optional[set[int]],
    context: str,
) -> List[int]:
    probe_pm.setup_configs(generation_config=generation_cfg, probe_config=answer_cfg)
    answer_seed_path = probe_pm.get_probe_output(
        layer=answer_cfg.layer,
        token_position=answer_cfg.token_positions,
        output_type="model",
    )

    if not answer_seed_path.exists():
        LOG.warning(
            f"{context}: representative answer probe weight is missing at {answer_seed_path}."
        )

    answer_objective_root = answer_seed_path.parent.parent.parent
    layer_dir_name = answer_seed_path.parent.name
    token_positions = _token_text(answer_cfg.token_positions)
    pattern = f"res_layer_*_token_{token_positions}/{layer_dir_name}/model_SoftmaxClassifier.pt"

    discovered: List[int] = []
    for answer_model_path in sorted(answer_objective_root.glob(pattern)):
        layer_dir = answer_model_path.parent.parent
        layer = _parse_probe_layer_dir(layer_dir, token_positions, context)
        if requested_layers is not None and layer not in requested_layers:
            continue

        candidate_answer_cfg = replace(answer_cfg, layer=layer)
        candidate_pred_cfg = replace(pred_cfg, layer=layer)

        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=candidate_answer_cfg)
        canonical_answer_path = probe_pm.get_probe_output(
            layer=layer,
            token_position=candidate_answer_cfg.token_positions,
            output_type="model",
        )
        probe_pm.setup_configs(generation_config=generation_cfg, probe_config=candidate_pred_cfg)
        canonical_pred_path = probe_pm.get_probe_output(
            layer=layer,
            token_position=candidate_pred_cfg.token_positions,
            output_type="model",
        )

        if canonical_answer_path.exists() and canonical_pred_path.exists():
            discovered.append(layer)

    return sorted(dict.fromkeys(discovered))


def _expand_probe_pairs_from_final_config(
    source_yamls: Sequence[str],
    output_root: Path,
    requested_layers: Optional[Sequence[int]],
    context: str,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    requested_layer_set = set(int(layer) for layer in requested_layers) if requested_layers is not None else None
    template_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for source_yaml in source_yamls:
        cfg = ExperimentConfig.from_sources({}, cfg_path=Path(source_yaml))
        probe_exp = cfg.probe_exp_config
        if probe_exp is None:
            raise ExactPathValidationError(f"{context}: source YAML is missing probe_exp_config: {source_yaml}")
        activation_exp = probe_exp.activation_exp_config
        if activation_exp is None:
            raise ExactPathValidationError(f"{context}: nested probe_exp_config is missing activation_exp_config: {source_yaml}")

        generation_cfgs = activation_exp.generation_config_list or []
        if len(generation_cfgs) != 1:
            raise ExactPathValidationError(
                f"{context}: expected exactly one generation_config in nested probe_exp_config, "
                f"found {len(generation_cfgs)} in {source_yaml}."
            )
        generation_cfg = generation_cfgs[0]

        for probe_cfg in probe_exp.probe_config_list or []:
            key = _probe_template_key(probe_cfg)
            entry = template_map.setdefault(
                key,
                {
                    "probe_exp": probe_exp,
                    "generation_cfg": generation_cfg,
                    "answer_cfg": None,
                    "pred_cfg": None,
                    "canonical_layers": set(),
                },
            )
            if _normalize_generation_cfg(entry["generation_cfg"]) != _normalize_generation_cfg(generation_cfg):
                raise ExactPathValidationError(
                    f"{context}: source YAMLs disagree on nested generation_config for the same probe template."
                )

            entry["canonical_layers"].add(int(probe_cfg.layer))
            if probe_cfg.objective == "answer" and entry["answer_cfg"] is None:
                entry["answer_cfg"] = probe_cfg
            elif probe_cfg.objective == "pred" and entry["pred_cfg"] is None:
                entry["pred_cfg"] = probe_cfg

    pairs_by_path: Dict[Tuple[str, str], Dict[str, Any]] = {}
    canonical_layers: set[int] = set()

    for template in template_map.values():
        answer_cfg = template["answer_cfg"]
        pred_cfg = template["pred_cfg"]
        if answer_cfg is None or pred_cfg is None:
            raise ExactPathValidationError(
                f"{context}: final-config probe template is missing answer/pred representative configs."
            )

        canonical_layers.update(int(layer) for layer in template["canonical_layers"])
        probe_pm = ReadOnlyPathManager(template["probe_exp"])
        probe_pm.base_path = output_root

        available_layers = _discover_available_probe_layers(
            probe_pm=probe_pm,
            generation_cfg=template["generation_cfg"],
            answer_cfg=answer_cfg,
            pred_cfg=pred_cfg,
            requested_layers=requested_layer_set,
            context=context,
        )

        for layer in available_layers:
            candidate_answer_cfg = replace(answer_cfg, layer=layer)
            candidate_pred_cfg = replace(pred_cfg, layer=layer)

            probe_pm.setup_configs(generation_config=template["generation_cfg"], probe_config=candidate_answer_cfg)
            answer_path = probe_pm.get_probe_output(
                layer=layer,
                token_position=candidate_answer_cfg.token_positions,
                output_type="model",
            )
            probe_pm.setup_configs(generation_config=template["generation_cfg"], probe_config=candidate_pred_cfg)
            pred_path = probe_pm.get_probe_output(
                layer=layer,
                token_position=candidate_pred_cfg.token_positions,
                output_type="model",
            )

            if not answer_path.exists() or not pred_path.exists():
                continue

            norm = _normalize_probe_entry(candidate_answer_cfg)
            train = norm["training_config"]
            pair = {
                "layer": int(layer),
                "component": norm["component"],
                "token_positions": norm["token_positions"],
                "learning_rate": train["learning_rate"],
                "batch_size": train["batch_size"],
                "epochs": train["epochs"],
                "ans_path": str(answer_path),
                "pred_path": str(pred_path),
                "ans_config_path": "",
                "pred_config_path": "",
                "probe_save_name": norm["save_name"],
                "is_explicit_final_config_probe_layer": int(layer) in template["canonical_layers"],
            }
            pairs_by_path[(pair["ans_path"], pair["pred_path"])] = pair

    pairs = sorted(pairs_by_path.values(), key=lambda pair: (pair["layer"], pair["component"], pair["token_positions"]))
    if not pairs:
        LOG.warning(
            "%s: no same-template probe pairs were found when expanding final-config probe settings across layers. "
            "Skipping this bundle.",
            context,
        )
        return [], sorted(canonical_layers)

    return pairs, sorted(canonical_layers)


def select_run_by_size(steering_runs: List[Dict[str, Any]], size: int) -> Optional[Dict[str, Any]]:
    candidates = [run for run in steering_runs if len(run["intervention_layers"]) == size]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda run: (
            run.get("source_subdir", ""),
            run.get("metadata_path", ""),
        ),
    )[0]


# ---------------------------------------------------------------------------
# Lenient manifest (ported from build_final_config_manifest; eval-CSV dropped).
# ---------------------------------------------------------------------------
def _build_lenient_manifest(
    config_dir: Path,
    output_root: Path,
    models: Optional[Sequence[str]],
    datasets: Optional[Sequence[str]],
    allow_partial: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_filter = set(datasets) if datasets else None

    bundles: Dict[Tuple[str, str], Dict[str, Any]] = {}
    error_rows: List[Dict[str, Any]] = []

    for _model_key, yaml_path in iter_kappa_config_paths(config_dir, models):
        try:
            cfg = ExperimentConfig.from_sources({}, cfg_path=yaml_path)
            model_key = model_key_from_cfg(cfg)
            dataset_key = extract_probe_save_name(cfg.probe_exp_config, str(yaml_path))
            if dataset_filter and dataset_key not in dataset_filter:
                continue

            bundle = bundles.setdefault(
                (model_key, dataset_key),
                {
                    "model": model_key,
                    "dataset": dataset_key,
                    "source_yamls": [],
                    "probe_pairs": [],
                    "steering_runs": [],
                    "base_runs": [],
                },
            )
            bundle["source_yamls"].append(str(yaml_path))
            bundle["probe_pairs"].extend(_build_probe_pairs(cfg, output_root, yaml_path))
            bundle["steering_runs"].extend(_build_steering_runs(cfg, output_root, yaml_path))
            bundle["base_runs"].extend(_build_base_runs(cfg, output_root, yaml_path))
        except Exception as exc:
            if not allow_partial:
                raise
            error_rows.append({
                "raw_model": "",
                "raw_dataset": "",
                "display_model": "",
                "display_dataset": "",
                "method_kind": "bundle_build",
                "candidate_path": str(yaml_path),
                "error": str(exc),
            })

    manifest: List[Dict[str, Any]] = []
    for (model_key, dataset_key), bundle in sorted(bundles.items()):
        probe_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for pair in bundle["probe_pairs"]:
            probe_map.setdefault((pair["ans_path"], pair["pred_path"]), dict(pair))

        run_map: Dict[str, Dict[str, Any]] = {}
        for run in bundle["steering_runs"]:
            run_map.setdefault(run["metadata_path"], dict(run))

        base_map: Dict[str, Dict[str, Any]] = {}
        for run in bundle["base_runs"]:
            key = run["metadata_path"]
            if key in base_map:
                prior = base_map[key]
                prior["source_yaml"] = ";".join(sorted(set((prior["source_yaml"] + ";" + run["source_yaml"]).split(";"))))
            else:
                base_map[key] = dict(run)

        manifest.append({
            "model": model_key,
            "dataset": dataset_key,
            "probe_pairs": sorted(probe_map.values(), key=lambda row: (int(row["layer"]), row["ans_path"])),
            "audit": {
                "source_yamls": sorted(set(bundle["source_yamls"])),
                "base_runs": sorted(base_map.values(), key=lambda row: row["metadata_path"]),
                "steering_runs": sorted(
                    run_map.values(),
                    key=lambda row: (len(row["intervention_layers"]), row["metadata_path"]),
                ),
            },
        })

    return manifest, error_rows


# ---------------------------------------------------------------------------
# Public bundle manifest (ported from build_config_driven_manifest).
# ---------------------------------------------------------------------------
def build_manifest(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    models: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    allow_partial: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lenient_manifest, manifest_errors = _build_lenient_manifest(
        config_dir=config_dir,
        output_root=output_root,
        models=models,
        datasets=datasets,
        allow_partial=allow_partial,
    )

    manifest: List[Dict[str, Any]] = []
    for bundle in lenient_manifest:
        try:
            source_yamls = bundle["audit"]["source_yamls"]
            if not source_yamls:
                raise ExactPathValidationError(
                    f"{bundle['model']}/{bundle['dataset']}: no source YAMLs were recorded in the final-config manifest."
                )

            reference_cfg = ExperimentConfig.from_sources({}, cfg_path=Path(source_yamls[0]))
            dataset_cfg_payloads = reference_cfg.probe_exp_config.activation_exp_config.to_dict().get("dataset_config_list") or []
            option_info = _option_info_from_dataset_configs(dataset_cfg_payloads)

            intervention_layers: Dict[int, List[int]] = {}
            grouped_layers: Dict[int, set[int]] = defaultdict(set)
            for run in bundle["audit"]["steering_runs"]:
                grouped_layers[len(run["intervention_layers"])].update(int(layer) for layer in run["intervention_layers"])
            for size, layers_for_size in grouped_layers.items():
                intervention_layers[size] = sorted(layers_for_size)

            probe_pairs, canonical_probe_layers = _expand_probe_pairs_from_final_config(
                source_yamls=source_yamls,
                output_root=output_root,
                requested_layers=None,
                context=f"{bundle['model']}/{bundle['dataset']}",
            )

            if not probe_pairs:
                LOG.warning(
                    "%s/%s: no probe pairs were produced from the public KAPPA configs. Skipping this bundle.",
                    bundle["model"],
                    bundle["dataset"],
                )
                continue

            manifest.append({
                "model": bundle["model"],
                "dataset": bundle["dataset"],
                "layers": [pair["layer"] for pair in probe_pairs],
                "option_info": option_info,
                "probe_pairs": probe_pairs,
                "canonical_probe_layers": canonical_probe_layers,
                "intervention_layers": intervention_layers,
                "audit": {
                    "source_yamls": source_yamls,
                    "steering_runs": bundle["audit"]["steering_runs"],
                },
            })
        except ExactPathValidationError as exc:
            if allow_partial:
                LOG.error("%s", exc)
                continue
            raise

    if not manifest and not allow_partial:
        raise ExactPathValidationError(
            "No config-driven experiment bundles survived manifest construction. "
            "Next step: inspect the public KAPPA configs and the derived output paths. "
            "Run run_activation_collection.py then run_probe.py first to produce probe outputs under kappa_core/outputs/."
        )
    return manifest, manifest_errors


# ---------------------------------------------------------------------------
# Metric recompute (verbatim from the original geometry analysis code).
# ---------------------------------------------------------------------------
def _sniff_json_format(path: Path, max_bytes: int = 1 << 20) -> str:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(max_bytes)

    stripped = head.lstrip()
    if not stripped:
        return "json"

    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except Exception:
            pass

    ok = 0
    total = 0
    for line in head.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            json.loads(line)
            ok += 1
        except Exception:
            break
        if total >= 5:
            break

    if ok >= 1 and ok == total:
        return "jsonl"
    return "json"


def load_items(path: Path) -> List[Dict[str, Any]]:
    cache_key = str(path)
    if cache_key in _ITEMS_CACHE:
        return _ITEMS_CACHE[cache_key]

    fmt = _sniff_json_format(path)
    if fmt == "jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
        _ITEMS_CACHE[cache_key] = items
        return items

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        out = [item for item in obj if isinstance(item, dict)]
        _ITEMS_CACHE[cache_key] = out
        return out
    if isinstance(obj, dict):
        for key in ["data", "items", "examples", "records", "rows", "metadata"]:
            value = obj.get(key)
            if isinstance(value, list):
                out = [item for item in value if isinstance(item, dict)]
                _ITEMS_CACHE[cache_key] = out
                return out
        out = [obj]
        _ITEMS_CACHE[cache_key] = out
        return out

    _ITEMS_CACHE[cache_key] = []
    return []


def extract_key(item: Dict[str, Any]) -> Optional[Tuple[int, int, str]]:
    try:
        return (int(item["item_idx"]), int(item["perm_idx"]), item["question_no_option"])
    except Exception:
        return None


def index_items_by_key(items: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for item in items:
        key = extract_key(item)
        if key is not None:
            out[key] = item
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    ex = np.exp(x)
    denom = ex.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(x, 1.0 / len(x), dtype=np.float64)
    return ex / denom


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def summarize_file_against_kp(file_path: Path, kp_path: Path) -> Optional[Dict[str, float]]:
    items = load_items(file_path)
    kp_cache_key = str(kp_path)
    if kp_cache_key not in _INDEX_CACHE:
        _INDEX_CACHE[kp_cache_key] = index_items_by_key(load_items(kp_path))
    kp_map = _INDEX_CACHE[kp_cache_key]

    rows: List[Tuple[int, int, float]] = []
    miss_key = 0
    miss_kp = 0
    bad_logits = 0

    for item in items:
        item_key = extract_key(item)
        if item_key is None:
            miss_key += 1
            continue

        kp_item = kp_map.get(item_key)
        if kp_item is None:
            miss_kp += 1
            continue

        if "logits" not in item or "logits" not in kp_item:
            bad_logits += 1
            continue

        y = kp_item.get("answer")
        logits_method = np.asarray(item["logits"], dtype=np.float64).reshape(-1)
        logits_kp = np.asarray(kp_item["logits"], dtype=np.float64).reshape(-1)
        p_method = softmax(logits_method)
        p_kp = softmax(logits_kp)

        rows.append(
            (
                int(int(np.argmax(p_method)) == int(y)),
                int(int(np.argmax(p_method)) == int(np.argmax(p_kp))),
                kl_div(p_method, p_kp),
            )
        )

    if not rows:
        return None

    arr = np.asarray(rows, dtype=np.float64)
    return {
        "n": int(arr.shape[0]),
        "ACC_mean": float(arr[:, 0].mean()),
        "AGR_mean": float(arr[:, 1].mean()),
        "KLD_mean": float(arr[:, 2].mean()),
        "miss_key": miss_key,
        "miss_kp": miss_kp,
        "bad_logits": bad_logits,
    }


def summarize_knowprobe_self(kp_path: Path) -> Optional[Dict[str, float]]:
    return summarize_file_against_kp(kp_path, kp_path)


def infer_num_options_from_metadata(metadata_path: Path) -> int:
    items = load_items(metadata_path)
    if not items:
        raise PairingLogicError(f"Could not infer num_options because metadata is empty: {metadata_path}")
    first = items[0]
    if "logits" not in first:
        raise PairingLogicError(f"Could not infer num_options because logits are missing: {metadata_path}")
    return int(len(first["logits"]))


def _select_acc_best_probe_pair(bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for pair in bundle["probe_pairs"]:
        kp_path = Path(pair["ans_metadata_path"])
        ans_model_path = Path(pair["ans_path"])
        pred_model_path = Path(pair["pred_path"])
        if not (kp_path.exists() and ans_model_path.exists() and pred_model_path.exists()):
            continue

        summary = summarize_knowprobe_self(kp_path)
        if summary is None:
            continue

        candidates.append({
            "pair": pair,
            "summary": summary,
        })

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda row: (
            -float(row["summary"]["ACC_mean"]),
            -float(row["summary"]["AGR_mean"]),
            float(row["summary"]["KLD_mean"]),
            int(row["pair"]["layer"]),
        ),
    )[0]


# ---------------------------------------------------------------------------
# Per-(model,dataset) rows (ported from build_final_config_rows; gain dropped).
# ---------------------------------------------------------------------------
def build_rows(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    models: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    allow_partial: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    manifest, manifest_errors = _build_lenient_manifest(
        config_dir=config_dir,
        output_root=output_root,
        models=models,
        datasets=datasets,
        allow_partial=allow_partial,
    )

    selected_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for bundle in manifest:
        model_key = bundle["model"]
        dataset_key = bundle["dataset"]
        display_model = model_key
        display_dataset = DATASET_NAME_MAP.get(dataset_key, dataset_key)
        base_runs = bundle["audit"].get("base_runs", [])

        if len(base_runs) != 1:
            message = (
                f"Expected exactly one canonical base test activation path from activation_exp_config, "
                f"found {len(base_runs)}."
            )
            audit_rows.append({
                "model": display_model,
                "dataset": display_dataset,
                "raw_model": model_key,
                "raw_dataset": dataset_key,
                "status": "excluded",
                "message": message,
                "selected_layer": "",
                "best_knowprobe_layer": "",
                "path_base": "",
                "path_knowprobe": "",
            })
            if not allow_partial:
                raise PairingLogicError(f"{model_key}/{dataset_key}: {message}")
            continue

        base_run = base_runs[0]
        base_path = Path(base_run["metadata_path"])
        best_probe = _select_acc_best_probe_pair(bundle)
        if best_probe is None:
            message = "No scoreable paired knowprobe layer was found for ACC-best selection."
            audit_rows.append({
                "model": display_model,
                "dataset": display_dataset,
                "raw_model": model_key,
                "raw_dataset": dataset_key,
                "status": "excluded",
                "message": message,
                "selected_layer": "",
                "best_knowprobe_layer": "",
                "path_base": str(base_path),
                "path_knowprobe": "",
            })
            if not allow_partial:
                raise PairingLogicError(f"{model_key}/{dataset_key}: {message}")
            continue

        selected_pair = best_probe["pair"]
        selected_layer = int(selected_pair["layer"])
        kp_path = Path(selected_pair["ans_metadata_path"])

        kp_has_path = kp_path.exists()
        base_has_path = base_path.exists()
        status = "verified" if kp_has_path and base_has_path else "excluded"
        message = (
            f"config_source=public_kappa | best_knowprobe_layer={selected_layer} | "
            f"base_path={'ok' if base_has_path else 'missing'} | "
            f"knowprobe_path={'ok' if kp_has_path else 'missing'}"
        )

        audit_rows.append({
            "model": display_model,
            "dataset": display_dataset,
            "raw_model": model_key,
            "raw_dataset": dataset_key,
            "status": status,
            "message": message,
            "selected_layer": selected_layer,
            "best_knowprobe_layer": selected_layer,
            "path_base": str(base_path),
            "path_knowprobe": str(kp_path),
        })

        manifest_rows.append({
            "model": display_model,
            "dataset": display_dataset,
            "raw_model": model_key,
            "raw_dataset": dataset_key,
            "selected_layer": selected_layer,
            "best_knowprobe_layer": selected_layer,
            "path_base": str(base_path),
            "path_knowprobe": str(kp_path),
            "source_yamls": ";".join(bundle["audit"]["source_yamls"]),
        })

        if not (kp_has_path and base_has_path):
            if not allow_partial:
                raise PairingLogicError(
                    f"{model_key}/{dataset_key}: excluded because a config-derived path is missing. {message}"
                )
            continue

        selected_rows.append({
            "model": display_model,
            "dataset": display_dataset,
            "raw_model": model_key,
            "raw_dataset": dataset_key,
            "num_options": infer_num_options_from_metadata(kp_path),
            "selected_layer": selected_layer,
            "best_knowprobe_layer": selected_layer,
            "path_base": str(base_path),
            "path_knowprobe": str(kp_path),
            "answer_model_path": str(Path(selected_pair["ans_path"])),
            "pred_model_path": str(Path(selected_pair["pred_path"])),
            "base_source_yaml": str(base_run["source_yaml"]),
            "source_yamls": ";".join(bundle["audit"]["source_yamls"]),
        })

    for error in manifest_errors:
        audit_rows.append({
            "model": "",
            "dataset": "",
            "raw_model": error.get("raw_model", ""),
            "raw_dataset": error.get("raw_dataset", ""),
            "status": "excluded",
            "message": error.get("error", ""),
            "selected_layer": "",
            "best_knowprobe_layer": "",
            "path_base": "",
            "path_knowprobe": "",
        })

    return selected_rows, audit_rows, pd.DataFrame(manifest_rows)


def _summary_to_prefixed_columns(summary: Optional[Dict[str, float]], prefix: str) -> Dict[str, Any]:
    if summary is None:
        return {
            f"n_{prefix}": None,
            f"acc_{prefix}": None,
            f"agr_{prefix}": None,
            f"kld_{prefix}": None,
        }
    return {
        f"n_{prefix}": int(summary["n"]),
        f"acc_{prefix}": float(summary["ACC_mean"]),
        f"agr_{prefix}": float(summary["AGR_mean"]),
        f"kld_{prefix}": float(summary["KLD_mean"]),
    }


def _safe_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def compute_method_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    kp_path = Path(row["path_knowprobe"])
    base_path = Path(row["path_base"])

    if not kp_path.exists():
        raise PairingLogicError(f"{row['raw_model']}/{row['raw_dataset']}: knowprobe metadata is missing: {kp_path}")
    if not base_path.exists():
        raise PairingLogicError(f"{row['raw_model']}/{row['raw_dataset']}: base metadata is missing: {base_path}")

    knowprobe_summary = summarize_knowprobe_self(kp_path)
    base_summary = summarize_file_against_kp(base_path, kp_path)

    if knowprobe_summary is None:
        raise PairingLogicError(
            f"{row['raw_model']}/{row['raw_dataset']}: failed to compute knowprobe metrics from {kp_path}"
        )
    if base_summary is None:
        raise PairingLogicError(
            f"{row['raw_model']}/{row['raw_dataset']}: failed to compute base metrics from {base_path}"
        )

    metrics = {}
    metrics.update(_summary_to_prefixed_columns(base_summary, "base"))
    metrics.update(_summary_to_prefixed_columns(knowprobe_summary, "knowprobe"))

    metrics["gap_acc_knowprobe_vs_base"] = _safe_delta(metrics["acc_knowprobe"], metrics["acc_base"])
    metrics["gap_agr_knowprobe_vs_base"] = _safe_delta(metrics["agr_knowprobe"], metrics["agr_base"])
    metrics["gap_kld_knowprobe_vs_base"] = _safe_delta(metrics["kld_knowprobe"], metrics["kld_base"])
    return metrics


def attach_gap_metrics(rows: List[Dict[str, Any]], show_progress: bool = False) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    row_iter = rows
    for row in row_iter:
        merged = dict(row)
        merged.update(compute_method_metrics(row))
        enriched.append(merged)
    return enriched


# ---------------------------------------------------------------------------
# Activation views (verbatim from the original geometry analysis code).
# ---------------------------------------------------------------------------
def _activation_file_exists(base_path: Path) -> bool:
    return base_path.exists() or any(base_path.parent.glob(f"{base_path.stem}.shard_*.pt"))


def _normalize_token_positions(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(part) for part in value.split("_") if str(part).strip() != ""]
    if isinstance(value, (list, tuple)):
        normalized: List[int] = []
        for item in value:
            normalized.extend(_normalize_token_positions(item))
        return normalized
    return [int(value)]


def _resolve_position(seq_len: int, pos: int) -> int:
    return seq_len + pos if pos < 0 else pos


def _validate_saved_token_alignment(
    key: Tuple[str, int, int],
    items: List[Dict[str, Any]],
    context: str,
) -> None:
    _component, _layer, pos = key
    for idx, item in enumerate(items):
        tokens = item.get("tokens")
        saved_positions = item.get("saved_token_positions")
        saved_tokens = item.get("saved_tokens")

        if not isinstance(tokens, list) or not tokens:
            continue
        if not isinstance(saved_positions, list) or not saved_positions:
            continue

        resolved = _resolve_position(len(tokens), int(pos))
        if resolved not in saved_positions:
            raise ExactPathValidationError(
                f"{context}: token-position mismatch for activation key={key} at item_idx={item.get('item_idx')} "
                f"perm_idx={item.get('perm_idx')}. Configured token_position={pos} resolves to {resolved}, "
                f"but metadata saved_token_positions={saved_positions}."
            )

        if isinstance(saved_tokens, list) and saved_tokens:
            if resolved < 0 or resolved >= len(tokens):
                raise ExactPathValidationError(
                    f"{context}: resolved token position {resolved} is out of range for key={key} "
                    f"at item index {idx}."
                )
            observed_token = tokens[resolved]
            if observed_token not in saved_tokens:
                raise ExactPathValidationError(
                    f"{context}: saved token text mismatch for activation key={key} at item_idx={item.get('item_idx')} "
                    f"perm_idx={item.get('perm_idx')}. Observed token={observed_token!r}, "
                    f"metadata saved_tokens={saved_tokens}."
                )


def load_bundle_activation_views(
    bundle: Dict[str, Any],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[Tuple[str, int, int], ActivationView]:
    source_yamls = bundle["audit"]["source_yamls"]
    if not source_yamls:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: no source YAMLs were recorded for activation loading."
        )

    reference_cfg = ExperimentConfig.from_sources({}, cfg_path=Path(source_yamls[0]))
    probe_exp = reference_cfg.probe_exp_config
    if probe_exp is None or probe_exp.activation_exp_config is None:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: nested activation_exp_config is missing."
        )
    activation_exp = probe_exp.activation_exp_config
    activation_cfg = activation_exp.activation_config
    if activation_cfg is None:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: activation_config is missing in nested activation_exp_config."
        )

    configured_positions = sorted(dict.fromkeys(_normalize_token_positions(activation_cfg.token_positions)))
    expected_positions = sorted(
        {
            pos
            for pair in bundle["probe_pairs"]
            for pos in _normalize_token_positions(pair.get("token_positions"))
        }
    )
    if configured_positions != expected_positions:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: activation token_positions do not match probe token_positions. "
            f"activation_exp_config={configured_positions}, probe_pairs={expected_positions}"
        )

    dataset_cfgs = activation_exp.dataset_config_list or []
    generation_cfgs = activation_exp.generation_config_list or []
    if not dataset_cfgs:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: dataset_config_list is empty in activation_exp_config."
        )
    if not generation_cfgs:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: generation_config_list is empty in activation_exp_config."
        )

    split_names = sorted({str(ds_cfg.split) for ds_cfg in dataset_cfgs})
    if len(split_names) != 1:
        raise ExactPathValidationError(
            f"{bundle['model']}/{bundle['dataset']}: expected exactly one activation split, found {split_names}."
        )

    expanded_activation_cfg = replace(
        activation_cfg,
        layers=sorted(int(layer) for layer in bundle["layers"]),
    )
    local_manager = ReadOnlyPathManager(activation_exp)
    local_manager.base_path = output_root

    activations: Dict[Tuple[str, int, int], List[torch.Tensor]] = {}
    items: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    metadata_paths: Dict[Tuple[str, int, int], set[str]] = {}

    for ds_cfg, gen_cfg in product(dataset_cfgs, generation_cfgs):
        local_manager.setup_configs(dataset_config=ds_cfg, generation_config=gen_cfg)
        activation_path = Path(local_manager.get_activation_output(output_type="activations"))
        metadata_path = Path(local_manager.get_activation_output(output_type="metadata"))

        if not _activation_file_exists(activation_path):
            raise ExactPathValidationError(
                f"{bundle['model']}/{bundle['dataset']}: activation shard(s) are missing at the canonical path root "
                f"{activation_path.parent}."
            )
        if not metadata_path.exists():
            raise ExactPathValidationError(
                f"{bundle['model']}/{bundle['dataset']}: activation metadata is missing: {metadata_path}."
            )

        loader = ProbeActivationLoader(
            generation_config=gen_cfg,
            dataset_config=ds_cfg,
            activation_config=expanded_activation_cfg,
            path_manager=local_manager,
        )
        loaded_acts, loaded_items = loader.load_activation()
        if not loaded_acts:
            raise ExactPathValidationError(
                f"{bundle['model']}/{bundle['dataset']}: activation loader returned no activations for {metadata_path}."
            )

        for key, tensors in loaded_acts.items():
            activations.setdefault(key, [])
            items.setdefault(key, [])
            metadata_paths.setdefault(key, set())
            activations[key].extend(tensors)
            items[key].extend(loaded_items[key])
            metadata_paths[key].add(str(metadata_path))

    stacked: Dict[Tuple[str, int, int], ActivationView] = {}
    for key, tensor_list in activations.items():
        if not tensor_list:
            raise PairingLogicError(
                f"{bundle['model']}/{bundle['dataset']}: activation list for key={key} is unexpectedly empty."
            )
        matrix = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in tensor_list], dim=0)
        key_items = items[key]
        if matrix.shape[0] != len(key_items):
            raise PairingLogicError(
                f"{bundle['model']}/{bundle['dataset']}: activation/item count mismatch for key={key}: "
                f"{matrix.shape[0]} activations vs {len(key_items)} items."
            )
        _validate_saved_token_alignment(
            key,
            key_items,
            context=f"{bundle['model']}/{bundle['dataset']}",
        )
        stacked[key] = {
            "activations": matrix,
            "items": key_items,
            "metadata_paths": sorted(metadata_paths[key]),
            "split": split_names[0],
        }

    return stacked


def materialize_probe_activation_matrix(
    activation_views: Dict[Tuple[str, int, int], ActivationView],
    pair: Dict[str, Any],
    context: str,
) -> ActivationView:
    component = str(pair["component"])
    layer = int(pair["layer"])
    token_positions = str(pair["token_positions"])
    positions = [int(pos) for pos in token_positions.split("_")]

    tensors: List[torch.Tensor] = []
    items: List[Dict[str, Any]] = []
    metadata_paths: set[str] = set()
    split_name: str | None = None

    for pos in positions:
        key = (component, layer, pos)
        if key not in activation_views:
            raise ExactPathValidationError(
                f"{context}: missing activation view for component={component}, layer={layer}, token_position={pos}."
            )
        view = activation_views[key]
        tensors.append(view["activations"])
        items.extend(view["items"])
        metadata_paths.update(view["metadata_paths"])
        split_name = view["split"]

    matrix = torch.cat(tensors, dim=0)
    if matrix.shape[0] != len(items):
        raise PairingLogicError(
            f"{context}: concatenated activation/item count mismatch: {matrix.shape[0]} vs {len(items)}."
        )

    return {
        "activations": matrix,
        "items": items,
        "metadata_paths": sorted(metadata_paths),
        "split": split_name or "",
    }


# ---------------------------------------------------------------------------
# Within-task CKA rows (verbatim body of compute_cka_artifacts_for_bundle,
# summaries construction removed).
# ---------------------------------------------------------------------------
def compute_cka_rows_for_bundle(
    bundle: Dict[str, Any],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_wu: bool = False,
    random_baseline_samples: int = RANDOM_BASELINE_SAMPLES,
    random_baseline_seed: int = RANDOM_BASELINE_SEED,
) -> List[Dict[str, Any]]:
    model = bundle["model"]
    dataset = bundle["dataset"]
    option_info = bundle["option_info"]
    probe_pairs = bundle["probe_pairs"]
    intervention_layers = bundle["intervention_layers"]
    canonical_probe_layers = set(int(layer) for layer in bundle.get("canonical_probe_layers", []))

    if model not in MODEL_CONFIGS:
        raise ExactPathValidationError(f"{model}/{dataset}: MODEL_CONFIGS does not define hidden size for this model.")

    d = int(MODEL_CONFIGS[model]["d"])
    k = int(option_info["k"])
    activation_views = load_bundle_activation_views(bundle=bundle, output_root=output_root)

    W_U = None
    if not skip_wu and not option_info["mixed_symbols"]:
        W_U = load_W_U_for_options(model, option_info["options"])

    rows: List[Dict[str, Any]] = []
    for pair in probe_pairs:
        layer = int(pair["layer"])
        context = f"{model}/{dataset}: layer={layer}"
        activation_view = materialize_probe_activation_matrix(activation_views, pair, context=context)
        raw_acts = activation_view["activations"]

        W_answer = load_probe_weights(Path(pair["ans_path"]), d, k)
        W_pred = load_probe_weights(Path(pair["pred_path"]), d, k)
        Q_answer = _orthonormal_row_space_basis(W_answer, context=f"{context}: answer probe")
        Q_pred = _orthonormal_row_space_basis(W_pred, context=f"{context}: prediction probe")

        features: Dict[str, torch.Tensor] = {
            "answer": torch.as_tensor(raw_acts, dtype=torch.float32) @ Q_answer,
            "pred": torch.as_tensor(raw_acts, dtype=torch.float32) @ Q_pred,
        }
        basis_map: Dict[str, torch.Tensor] = {
            "answer": Q_answer,
            "pred": Q_pred,
        }
        if W_U is not None:
            Q_U = _orthonormal_row_space_basis(W_U, context=f"{context}: W_U option span")
            features["U"] = torch.as_tensor(raw_acts, dtype=torch.float32) @ Q_U
            basis_map["U"] = Q_U

        pair_defs = [
            ("answer_vs_pred", "answer", "pred"),
        ]
        if W_U is not None:
            pair_defs.extend(
                [
                    ("answer_vs_U", "answer", "U"),
                    ("pred_vs_U", "pred", "U"),
                ]
            )

        base_row = {
            "model": model,
            "dataset": dataset,
            "layer": layer,
            "component": pair.get("component", ""),
            "token_pos": pair.get("token_positions", ""),
            "learning_rate": pair.get("learning_rate", ""),
            "batch_size": pair.get("batch_size", ""),
            "epochs": pair.get("epochs", ""),
            "n_examples": int(raw_acts.shape[0]),
            "is_intervention_1": layer in intervention_layers.get(1, []),
            "is_intervention_3": layer in intervention_layers.get(3, []),
            "is_intervention_6": layer in intervention_layers.get(6, []),
            "is_explicit_final_config_probe_layer": bool(
                pair.get("is_explicit_final_config_probe_layer", layer in canonical_probe_layers)
            ),
            "activation_metadata_path": ";".join(activation_view["metadata_paths"]),
            "activation_split": activation_view["split"],
            "ans_path": pair["ans_path"],
            "pred_path": pair["pred_path"],
            "projection_basis": "orthonormal_centered_row_span",
            "random_baseline_samples": int(random_baseline_samples),
        }

        for pair_name, left_name, right_name in pair_defs:
            left = features[left_name]
            right = features[right_name]
            rows.append(
                {
                    **base_row,
                    "pair": pair_name,
                    "cka": linear_centered_cka(left, right),
                    "feature_dim_left": int(left.shape[1]),
                    "feature_dim_right": int(right.shape[1]),
                    "random_baseline_std": float("nan"),
                }
            )

        for feature_name, pair_name in (
            ("answer", "answer_vs_random"),
            ("pred", "pred_vs_random"),
        ):
            baseline = _random_baseline_cka(
                raw_acts=raw_acts,
                features=features[feature_name],
                d=d,
                rank=int(features[feature_name].shape[1]),
                num_samples=int(random_baseline_samples),
                seed=int(random_baseline_seed),
                context=f"{context}: {pair_name}",
            )
            rows.append(
                {
                    **base_row,
                    "pair": pair_name,
                    "cka": baseline["mean"],
                    "feature_dim_left": int(features[feature_name].shape[1]),
                    "feature_dim_right": int(features[feature_name].shape[1]),
                    "random_baseline_std": baseline["std"],
                    "random_baseline_samples": baseline["num_samples"],
                }
            )
        if W_U is not None:
            baseline = _random_baseline_cka(
                raw_acts=raw_acts,
                features=features["U"],
                d=d,
                rank=int(features["U"].shape[1]),
                num_samples=int(random_baseline_samples),
                seed=int(random_baseline_seed),
                context=f"{context}: U_vs_random",
            )
            rows.append(
                {
                    **base_row,
                    "pair": "U_vs_random",
                    "cka": baseline["mean"],
                    "feature_dim_left": int(features["U"].shape[1]),
                    "feature_dim_right": int(features["U"].shape[1]),
                    "random_baseline_std": baseline["std"],
                    "random_baseline_samples": baseline["num_samples"],
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Within-task principal angle rows (verbatim body of _compute_bundle_results;
# only the unreachable post-`continue` raises are dropped).
# ---------------------------------------------------------------------------
def compute_principal_angle_rows_for_bundle(
    bundle: Dict[str, Any],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_wu: bool = False,
) -> List[Dict[str, Any]]:
    model = bundle["model"]
    dataset = bundle["dataset"]
    option_info = bundle["option_info"]
    probe_pairs = bundle["probe_pairs"]
    intervention_layers = bundle["intervention_layers"]
    canonical_probe_layers = set(int(layer) for layer in bundle.get("canonical_probe_layers", []))

    if model not in MODEL_CONFIGS:
        raise ExactPathValidationError(
            f"{model}/{dataset}: MODEL_CONFIGS does not define hidden size for this model."
        )

    d = MODEL_CONFIGS[model]["d"]
    k = option_info["k"]
    random_baseline = get_random_baseline(d, k, num_samples=100)

    W_U = None
    if skip_wu:
        LOG.info("%s/%s: skipping W_U because --skip-wu was requested.", model, dataset)
    elif option_info["mixed_symbols"]:
        LOG.warning(
            "%s/%s: mixed option symbols detected (%s); skipping W_U-based pairs.",
            model,
            dataset,
            option_info["option_symbol_name"],
        )
    else:
        try:
            W_U = load_W_U_for_options(model, option_info["options"])
        except Exception as exc:  # pragma: no cover - depends on local HF access
            raise exc

    rows: List[Dict[str, Any]] = []
    for pair in probe_pairs:
        layer = pair["layer"]
        try:
            W_know = load_probe_weights(Path(pair["ans_path"]), d, k)
        except Exception as exc:
            continue

        try:
            W_pred = load_probe_weights(Path(pair["pred_path"]), d, k)
        except Exception as exc:
            continue

        W_know_c = center_weights(W_know)
        W_pred_c = center_weights(W_pred)
        W_U_c = center_weights(W_U) if W_U is not None else None

        pa_kp = compute_principal_angles(W_know_c, W_pred_c)
        pa_pu = compute_principal_angles(W_pred_c, W_U_c) if W_U_c is not None else None
        pa_ku = compute_principal_angles(W_know_c, W_U_c) if W_U_c is not None else None

        base = {
            **random_baseline,
            "model": model,
            "dataset": dataset,
            "layer": layer,
            "component": pair.get("component", ""),
            "token_pos": pair.get("token_positions", ""),
            "learning_rate": pair.get("learning_rate", ""),
            "batch_size": pair.get("batch_size", ""),
            "epochs": pair.get("epochs", ""),
            "is_intervention_1": layer in intervention_layers.get(1, []),
            "is_intervention_3": layer in intervention_layers.get(3, []),
            "is_intervention_6": layer in intervention_layers.get(6, []),
            "is_explicit_final_config_probe_layer": bool(
                pair.get("is_explicit_final_config_probe_layer", layer in canonical_probe_layers)
            ),
            "k": k,
            "d": d,
            "option_symbol_name": option_info["option_symbol_name"],
            "ans_path": pair["ans_path"],
            "pred_path": pair["pred_path"],
            "ans_config_path": pair.get("ans_config_path", ""),
            "pred_config_path": pair.get("pred_config_path", ""),
        }

        rows.append({
            **base,
            "pair": "know_vs_pred",
            "mean_angle_deg": pa_kp["mean_angle_deg"],
            "min_angle_deg": pa_kp["min_angle_deg"],
            "max_angle_deg": pa_kp["max_angle_deg"],
            "frob_dist": pa_kp["proj_frob_dist"],
        })
        rows.append({
            **base,
            "pair": "pred_vs_U",
            "mean_angle_deg": pa_pu["mean_angle_deg"] if pa_pu is not None else float("nan"),
            "min_angle_deg": pa_pu["min_angle_deg"] if pa_pu is not None else float("nan"),
            "max_angle_deg": pa_pu["max_angle_deg"] if pa_pu is not None else float("nan"),
            "frob_dist": pa_pu["proj_frob_dist"] if pa_pu is not None else float("nan"),
        })
        rows.append({
            **base,
            "pair": "know_vs_U",
            "mean_angle_deg": pa_ku["mean_angle_deg"] if pa_ku is not None else float("nan"),
            "min_angle_deg": pa_ku["min_angle_deg"] if pa_ku is not None else float("nan"),
            "max_angle_deg": pa_ku["max_angle_deg"] if pa_ku is not None else float("nan"),
            "frob_dist": pa_ku["proj_frob_dist"] if pa_ku is not None else float("nan"),
        })

    return rows


__all__ = [
    "attach_gap_metrics",
    "build_manifest",
    "build_rows",
    "compute_cka_rows_for_bundle",
    "compute_method_metrics",
    "compute_principal_angle_rows_for_bundle",
    "extract_key",
    "index_items_by_key",
    "infer_num_options_from_metadata",
    "load_bundle_activation_views",
    "load_items",
    "materialize_probe_activation_matrix",
    "select_run_by_size",
    "summarize_file_against_kp",
    "summarize_knowprobe_self",
]
