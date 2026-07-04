#!/usr/bin/env python3
"""Reproduce Figure 9 (orthogonal CKA).

1. Build per-(model,dataset) bundles from configs/<model>/KAPPA/*.yaml.
2. For each layer, compute within-task orthogonal CKA between the knowledge
   and prediction probe spans (plus W_U and matched random baselines).
3. Render layer-wise CKA line plots with Top-1/3/6 intervention-layer markers.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from geometry._common import (
    DATASET_NAME_MAP, DEFAULT_CONFIG_DIR, DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLOT_DATASET_KEYS, DEFAULT_PLOT_MODEL_KEYS, MODEL_NAME_MAP,
    result_dirs,
)
from geometry._manifest import build_manifest, compute_cka_rows_for_bundle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("cka")

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CKA_PAIR_LABELS = {
    "answer_vs_pred": "Knowledge vs Prediction",
    "answer_vs_U": "Knowledge vs W_U",
    "pred_vs_U": "Prediction vs W_U",
    "answer_vs_random": "Knowledge vs Random",
    "pred_vs_random": "Prediction vs Random",
    "U_vs_random": "W_U vs Random",
}
PLOT_DATASET_NAME_MAP = {
    **DATASET_NAME_MAP,
    "bbh_algo_all": "BBH Algorithm",
}
INTERVENTION_STYLES = {
    "I6": {"label": "Top-6 Intv. Layer", "color": "#78c6b0", "half_width": 0.21, "alpha": 0.20},
    "I3": {"label": "Top-3 Intv. Layer", "color": "#f2c57c", "half_width": 0.13, "alpha": 0.30},
    "I1": {"label": "Top-1 Intv. Layer", "color": "#e6a5a5", "half_width": 0.05, "alpha": 0.40},
}
INTERVENTION_ORDER = ["I6", "I3", "I1"]


def _safe_name(text: str) -> str:
    return SAFE_NAME_RE.sub("_", text).strip("_") or "unknown"


def _display_model_name(model: str) -> str:
    return MODEL_NAME_MAP.get(model, model)


def _display_dataset_name(dataset: str) -> str:
    return PLOT_DATASET_NAME_MAP.get(dataset, dataset)


def _cka_title(model: str, dataset: str) -> str:
    return f"Orthogonal CKA: {_display_model_name(model)} / {_display_dataset_name(dataset)}"


def _save_plot_outputs(fig: plt.Figure, output_path: Path) -> None:
    base_path = output_path.with_suffix("")
    for suffix in (".png", ".pdf"):
        target_path = base_path.with_suffix(suffix)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=300, bbox_inches="tight")


def _dataset_plot_file_stem(dataset: str) -> str:
    stem = str(dataset)
    if stem.endswith("_all"):
        stem = stem[: -len("_all")]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "cka"
    return f"{stem}_cka"


def _build_intervention_layer_frame(group: pd.DataFrame) -> pd.DataFrame:
    return (
        group[["layer", "is_intervention_1", "is_intervention_3", "is_intervention_6"]]
        .drop_duplicates()
        .assign(
            intervention_tag=lambda frame: frame.apply(
                lambda r: "_".join(
                    [
                        name
                        for name, flag in zip(
                            ["I1", "I3", "I6"],
                            [r["is_intervention_1"], r["is_intervention_3"], r["is_intervention_6"]],
                        )
                        if bool(flag)
                    ]
                )
                or "none",
                axis=1,
            )
        )
    )


def _intervention_keys_from_tag(tag: str) -> List[str]:
    return [key for key in INTERVENTION_ORDER if key in str(tag)]


def _intervention_legend_handles(keys: Sequence[str]) -> List[Patch]:
    handles: List[Patch] = []
    for key in INTERVENTION_ORDER:
        if key not in keys:
            continue
        style = INTERVENTION_STYLES[key]
        handles.append(
            Patch(
                facecolor=style["color"],
                edgecolor="none",
                alpha=style["alpha"],
                label=style["label"],
            )
        )
    return handles


def _shade_intervention_layers(ax, layer_flags: pd.DataFrame) -> List[Patch]:
    if layer_flags.empty:
        return []

    present_keys = set()
    for _, row in layer_flags.iterrows():
        if str(row.get("intervention_tag", "none")) == "none":
            continue
        layer = float(row["layer"])
        for key in _intervention_keys_from_tag(str(row["intervention_tag"])):
            style = INTERVENTION_STYLES[key]
            ax.axvspan(
                layer - style["half_width"],
                layer + style["half_width"],
                color=style["color"],
                alpha=style["alpha"],
                linewidth=0,
                zorder=0.1,
            )
            present_keys.add(key)
    return _intervention_legend_handles(sorted(present_keys, key=INTERVENTION_ORDER.index))


def _configure_layer_axis(ax, layers: Sequence[int]) -> None:
    ordered_layers = sorted({int(layer) for layer in layers})
    if not ordered_layers:
        return
    ax.set_xlim(min(ordered_layers) - 0.5, max(ordered_layers) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="x", which="major", alpha=0.14, linewidth=0.8)
    ax.grid(axis="y", which="major", alpha=0.12, linewidth=0.8)
    ax.set_axisbelow(True)


def _filter_plot_rows(
    rows: List[Dict[str, Any]],
    requested_models: Optional[Sequence[str]],
    requested_datasets: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    model_targets = set(requested_models) if requested_models else set(DEFAULT_PLOT_MODEL_KEYS)
    dataset_targets = set(requested_datasets) if requested_datasets else set(DEFAULT_PLOT_DATASET_KEYS)
    return [
        row for row in rows
        if row.get("model") in model_targets
        and row.get("dataset") in dataset_targets
        and ("target_dataset" not in row or row.get("target_dataset") in dataset_targets)
    ]


def _save_payload(payload: Dict[str, Any], tag: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"cka{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    LOG.info("Saved CKA payload to %s", out_path)
    return out_path


def _load_payload_from_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {json_path}, got {type(payload).__name__}")
    payload.setdefault("meta", {})
    payload.setdefault("bundle_audit", [])
    payload.setdefault("cka", [])
    return payload


def _load_payload_from_csv(csv_path: Path) -> Dict[str, Any]:
    cka_df = pd.read_csv(csv_path)
    return {
        "meta": {"source_cka_csv": str(csv_path)},
        "bundle_audit": [],
        "cka": cka_df.to_dict(orient="records"),
    }


def save_cka_plots(rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not rows:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for (model, dataset), group in df.groupby(["model", "dataset"], sort=True):
        model_dir = plots_dir / _safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        layer_flags = _build_intervention_layer_frame(group)

        answer_pred_df = group[group["pair"] == "answer_vs_pred"].sort_values("layer")
        answer_random_df = group[group["pair"] == "answer_vs_random"].sort_values("layer")
        pred_random_df = group[group["pair"] == "pred_vs_random"].sort_values("layer")

        if not answer_pred_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(
                answer_pred_df["layer"],
                answer_pred_df["cka"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color="#1f77b4",
                label=CKA_PAIR_LABELS["answer_vs_pred"],
            )

            if not answer_random_df.empty:
                plt.plot(
                    answer_random_df["layer"],
                    answer_random_df["cka"],
                    linestyle="--",
                    linewidth=1.8,
                    color="#2ca02c",
                    label=CKA_PAIR_LABELS["answer_vs_random"],
                )
                if "random_baseline_std" in answer_random_df.columns:
                    random_std = answer_random_df["random_baseline_std"].fillna(0.0).to_numpy(dtype=float)
                    random_mean = answer_random_df["cka"].to_numpy(dtype=float)
                    plt.fill_between(
                        answer_random_df["layer"].to_numpy(dtype=float),
                        np.clip(random_mean - random_std, 0.0, 1.0),
                        np.clip(random_mean + random_std, 0.0, 1.0),
                        color="#2ca02c",
                        alpha=0.12,
                    )

            if not pred_random_df.empty:
                plt.plot(
                    pred_random_df["layer"],
                    pred_random_df["cka"],
                    linestyle="--",
                    linewidth=1.8,
                    color="#9467bd",
                    label=CKA_PAIR_LABELS["pred_vs_random"],
                )
                if "random_baseline_std" in pred_random_df.columns:
                    random_std = pred_random_df["random_baseline_std"].fillna(0.0).to_numpy(dtype=float)
                    random_mean = pred_random_df["cka"].to_numpy(dtype=float)
                    plt.fill_between(
                        pred_random_df["layer"].to_numpy(dtype=float),
                        np.clip(random_mean - random_std, 0.0, 1.0),
                        np.clip(random_mean + random_std, 0.0, 1.0),
                        color="#9467bd",
                        alpha=0.12,
                    )

            intervention_handles = _shade_intervention_layers(plt.gca(), layer_flags)
            plt.ylim(-0.02, 1.02)
            plt.xlabel("Layer")
            plt.ylabel("Orthogonal Linear CKA")
            plt.title(_cka_title(model, dataset))
            _configure_layer_axis(plt.gca(), answer_pred_df["layer"].tolist())
            line_handles, _ = plt.gca().get_legend_handles_labels()
            plt.legend(handles=line_handles + intervention_handles, frameon=False, ncol=2)
            plt.tight_layout()

            out_path = model_dir / f"{_dataset_plot_file_stem(dataset)}.png"
            _save_plot_outputs(plt.gcf(), out_path)
            plt.close()
            LOG.info("Saved CKA plot to %s", out_path)


def build_cka_payload(
    models: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_wu: bool = False,
    dry_run: bool = False,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    manifest, _manifest_errors = build_manifest(
        config_dir=config_dir,
        output_root=output_root,
        models=models,
        datasets=datasets,
        allow_partial=allow_partial,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "config_dir": str(config_dir),
            "output_root": str(output_root),
            "dry_run": dry_run,
            "skip_wu": skip_wu,
            "allow_partial": allow_partial,
        },
        "bundle_audit": manifest,
        "cka": [],
    }

    if dry_run:
        return payload

    all_rows: List[Dict[str, Any]] = []
    for bundle in tqdm(manifest, desc="Computing CKA"):
        all_rows.extend(
            compute_cka_rows_for_bundle(bundle, output_root=output_root, skip_wu=skip_wu)
        )

    if layers is not None:
        layer_set = {int(layer) for layer in layers}
        all_rows = [row for row in all_rows if int(row["layer"]) in layer_set]

    payload["cka"] = all_rows
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven orthogonal CKA using activations projected into orthonormal probe spans."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model keys to analyse (see geometry/_common.py MODEL_DIR_MAP for supported keys).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Probe save_name keys to analyse, e.g. truthfulqa gsm8k.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional layer filter applied to the config-expanded probe layer sweep.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory of public KAPPA configs (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Outputs root written by collector/run_probe.py (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--skip-wu",
        action="store_true",
        help="Skip W_U-based CKA pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config-driven bundle loading without computing CKA.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep valid bundles and log bundle-level failures instead of stopping at the first error.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for saved CKA outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "geometry" / "results" / "cka",
        help="Output directory root (default: geometry/results/cka)",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Load a previously saved CKA JSON payload and only regenerate plots from it.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Load a previously saved within-task CKA CSV and regenerate plots without recomputing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dirs = result_dirs(args.out_dir)

    if args.json_path is not None:
        payload = _load_payload_from_json(args.json_path)
        LOG.info("Loaded saved CKA payload from %s", args.json_path)
    elif args.csv_path is not None:
        payload = _load_payload_from_csv(args.csv_path)
        LOG.info("Loaded saved CKA rows from %s", args.csv_path)
    else:
        payload = build_cka_payload(
            models=args.models,
            datasets=args.datasets,
            layers=args.layers,
            config_dir=args.config_dir,
            output_root=args.output_root,
            skip_wu=args.skip_wu,
            dry_run=args.dry_run,
            allow_partial=args.allow_partial,
        )

        _save_payload(payload, args.tag, out_dirs["data"])

        cka_rows = payload["cka"]
        cka_df = pd.DataFrame(cka_rows)
        csv_suffix = f"_{args.tag}" if args.tag else ""
        csv_path = out_dirs["data"] / f"cka{csv_suffix}.csv"
        cka_df.to_csv(csv_path, index=False)
        LOG.info("Saved CKA rows to %s", csv_path)

    cka_rows = payload["cka"]
    plot_rows = _filter_plot_rows(cka_rows, args.models, args.datasets)
    save_cka_plots(plot_rows, out_dirs["plots"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
