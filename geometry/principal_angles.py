#!/usr/bin/env python3
"""Reproduce Figure 8 (principal angles).

1. Build per-(model,dataset) bundles from configs/<model>/KAPPA/*.yaml.
2. For each layer, compute knowledge-vs-prediction and prediction-vs-W_U mean
   principal angles plus a matched random baseline.
3. Render layer-wise line plots with Top-1/3/6 intervention-layer markers.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
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
from geometry._manifest import build_manifest, compute_principal_angle_rows_for_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("principal_angles")


PAIR_ORDER = ["know_vs_pred", "pred_vs_U"]
PAIR_LABELS = {
    "know_vs_pred": "Knowledge vs Prediction",
    "pred_vs_U": "Prediction vs W_U",
}
PAIR_STYLES = {
    "know_vs_pred": {"color": "#1f77b4"},
    "pred_vs_U": {"color": "#2ca02c"},
}
RANDOM_BASELINE_STYLE = {"color": "#9467bd", "band_alpha": 0.10}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
PRINCIPAL_ANGLE_YLIM = (55, 90)
DATASET_SPECIFIC_PRINCIPAL_ANGLE_YLIMS = {
    "tqa_4c_all": (45, 90),
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


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(text: str) -> str:
    return SAFE_NAME_RE.sub("_", text).strip("_") or "unknown"


def _display_model_name(model: str) -> str:
    return MODEL_NAME_MAP.get(model, model)


def _display_dataset_name(dataset: str) -> str:
    return PLOT_DATASET_NAME_MAP.get(dataset, dataset)


def _principal_angles_title(*parts: str) -> str:
    visible_parts = [part for part in parts if part]
    if not visible_parts:
        return "Principal Angles"
    return "Principal Angles: " + " / ".join(visible_parts)


def _principal_angle_ylim_for_dataset(dataset: str) -> Tuple[float, float]:
    return DATASET_SPECIFIC_PRINCIPAL_ANGLE_YLIMS.get(str(dataset), PRINCIPAL_ANGLE_YLIM)


def _save_plot_outputs(fig: plt.Figure, output_path: Path) -> None:
    base_path = output_path.with_suffix("")
    for suffix in (".png", ".pdf"):
        target_path = base_path.with_suffix(suffix)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=220, bbox_inches="tight")


def _dataset_plot_file_stem(dataset: str) -> str:
    stem = str(dataset)
    if stem.endswith("_all"):
        stem = stem[: -len("_all")]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "principal_angle"
    return f"{stem}_principal_angle"


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


def _configure_layer_axis(ax, layers: Sequence[int]) -> None:
    ordered_layers = sorted({int(layer) for layer in layers})
    if not ordered_layers:
        return
    ax.set_xlim(min(ordered_layers) - 0.5, max(ordered_layers) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="x", which="major", alpha=0.16, linewidth=0.8)
    ax.grid(axis="y", which="major", alpha=0.12, linewidth=0.8)
    ax.set_axisbelow(True)


def _resolve_plot_focus(
    available_models: Sequence[str],
    available_datasets: Sequence[str],
    requested_models: Optional[Sequence[str]],
    requested_datasets: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    model_pool = sorted(dict.fromkeys(str(model) for model in available_models))
    dataset_pool = sorted(dict.fromkeys(str(dataset) for dataset in available_datasets))

    model_targets = list(requested_models) if requested_models else list(DEFAULT_PLOT_MODEL_KEYS)
    dataset_targets = list(requested_datasets) if requested_datasets else list(DEFAULT_PLOT_DATASET_KEYS)

    selected_models = [model for model in model_pool if model in model_targets]
    selected_datasets = [dataset for dataset in dataset_pool if dataset in dataset_targets]
    return selected_models, selected_datasets


def build_principal_angle_payload(
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
        "principal_angles": [],
    }

    _print_manifest_summary(manifest, dry_run=dry_run)

    if not dry_run:
        all_results: List[Dict[str, Any]] = []
        for bundle in tqdm(manifest, desc="Computing principal angles"):
            all_results.extend(
                compute_principal_angle_rows_for_bundle(
                    bundle, output_root=output_root, skip_wu=skip_wu
                )
            )
        if layers is not None:
            layer_set = {int(layer) for layer in layers}
            all_results = [row for row in all_results if int(row["layer"]) in layer_set]
        payload["principal_angles"] = all_results
    return payload


def _print_manifest_summary(manifest: List[Dict[str, Any]], dry_run: bool) -> None:
    prefix = "[DRY RUN]" if dry_run else "[VALIDATED]"
    for bundle in manifest:
        LOG.info(
            "%s %s/%s | probe_layers=%s | final_config_probe_layers=%s | intervention_layers=%s",
            prefix,
            bundle["model"],
            bundle["dataset"],
            bundle["layers"],
            bundle.get("canonical_probe_layers", []),
            bundle["intervention_layers"],
        )
        for source_yaml in bundle["audit"]["source_yamls"]:
            LOG.info("  YAML: %s", source_yaml)
        for pair in bundle["probe_pairs"]:
            LOG.info(
                "  Probe layer=%s | explicit_final_config=%s | answer=%s | pred=%s",
                pair["layer"],
                pair.get("is_explicit_final_config_probe_layer", False),
                pair["ans_path"],
                pair["pred_path"],
            )


def _save_payload(payload: Dict[str, Any], tag: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"principal_angles{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    LOG.info("Saved results to %s", out_path)
    return out_path


def _save_payload_tables(payload: Dict[str, Any], tag: str, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    outputs: List[Path] = []

    principal_df = pd.DataFrame(payload.get("principal_angles", []))
    principal_path = out_dir / f"principal_angles{suffix}.csv"
    principal_df.to_csv(principal_path, index=False)
    outputs.append(principal_path)

    for path in outputs:
        LOG.info("Saved table to %s", path)
    return outputs


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df = df[df["pair"].isin(PAIR_ORDER)].copy()
    df["intervention_tag"] = df.apply(
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
    if "random_mean_angle_deg" in df.columns:
        df["angle_gap_to_random"] = df["random_mean_angle_deg"] - df["mean_angle_deg"]
    return df


def get_intervention_layers(sub: pd.DataFrame) -> pd.DataFrame:
    layer_df = (
        sub[["layer", "intervention_tag"]]
        .drop_duplicates()
        .sort_values("layer")
        .reset_index(drop=True)
    )
    return layer_df[layer_df["intervention_tag"] != "none"]


def shade_intervention_layers(ax, intervention_layers: pd.DataFrame) -> List[Patch]:
    if intervention_layers.empty:
        return []

    present_keys = set()
    for _, row in intervention_layers.iterrows():
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


def plot_dataset_pair_lines(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    output_path: Path,
) -> None:
    sub = df[(df["model"] == model) & (df["dataset"] == dataset)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for model={model}, dataset={dataset}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for pair in PAIR_ORDER:
        pair_df = sub[sub["pair"] == pair].sort_values("layer")
        if pair_df.empty:
            continue
        line_style = PAIR_STYLES.get(pair, {})
        ax.plot(
            pair_df["layer"],
            pair_df["mean_angle_deg"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=PAIR_LABELS.get(pair, pair),
            **line_style,
        )

    know_df = sub[sub["pair"] == "know_vs_pred"]
    if not know_df.empty and "random_mean_angle_deg" in know_df.columns:
        random_mean = float(know_df["random_mean_angle_deg"].iloc[0])
        random_std = float(know_df["random_std_mean_angle_deg"].iloc[0])
        ax.axhline(
            random_mean,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            color=RANDOM_BASELINE_STYLE["color"],
            label="Random baseline",
        )
        ax.fill_between(
            [know_df["layer"].min(), know_df["layer"].max()],
            [random_mean - random_std, random_mean - random_std],
            [random_mean + random_std, random_mean + random_std],
            color=RANDOM_BASELINE_STYLE["color"],
            alpha=RANDOM_BASELINE_STYLE["band_alpha"],
        )

    intervention_handles = shade_intervention_layers(ax, get_intervention_layers(sub))
    ax.set_title(_principal_angles_title(_display_model_name(model), _display_dataset_name(dataset)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean principal angle (deg)")
    ax.set_ylim(*_principal_angle_ylim_for_dataset(dataset))
    _configure_layer_axis(ax, sub["layer"].unique().tolist())
    line_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=line_handles + intervention_handles, frameon=False, ncol=2)
    fig.tight_layout()
    _save_plot_outputs(fig, output_path)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Optional list of models.")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional subset of datasets.")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Optional probe-layer filter.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="Directory of public KAPPA configs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "geometry" / "results" / "principal_angles",
    )
    parser.add_argument("--skip-wu", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Only compute/save tables, do not render plots.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="", help="Optional suffix for saved filenames.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dirs = result_dirs(args.out_dir)
    selected_models = args.models if args.models else None

    payload = build_principal_angle_payload(
        models=selected_models,
        datasets=args.datasets,
        layers=args.layers,
        config_dir=args.config_dir,
        output_root=args.output_root,
        skip_wu=args.skip_wu,
        dry_run=args.dry_run,
        allow_partial=args.allow_partial,
    )

    if args.dry_run:
        print("Dry run complete. Skipping plot rendering.")
        return 0

    _save_payload(payload, args.tag, dirs["data"])
    _save_payload_tables(payload, args.tag, dirs["data"])

    df = prepare_df(pd.DataFrame(payload["principal_angles"]).copy())

    if args.skip_plots:
        print(f"Skipped plotting. Output directory reserved for plots: {dirs['plots']}")
        return 0

    if df.empty:
        raise ValueError("No principal-angle rows are available to plot.")

    plot_models, plot_datasets = _resolve_plot_focus(
        available_models=sorted(df["model"].unique().tolist()),
        available_datasets=sorted(df["dataset"].unique().tolist()),
        requested_models=selected_models,
        requested_datasets=args.datasets,
    )
    working_df = df[df["model"].isin(plot_models) & df["dataset"].isin(plot_datasets)].copy()

    plot_jobs: List[Tuple[str, str]] = []
    for model in plot_models:
        model_df = working_df[working_df["model"] == model].copy()
        for dataset in sorted(model_df["dataset"].unique().tolist()):
            plot_jobs.append((model, dataset))

    for model, dataset in tqdm(plot_jobs, desc="Saving principal-angle plots"):
        model_dir = ensure_dir(dirs["plots"] / _safe_name(model))
        plot_dataset_pair_lines(
            working_df,
            model=model,
            dataset=dataset,
            output_path=model_dir / f"{_dataset_plot_file_stem(dataset)}.png",
        )
        tqdm.write(f"Saved plots for {model} / {dataset} -> {model_dir}")

    print(f"Done. Output directory: {dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
