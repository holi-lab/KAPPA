#!/usr/bin/env python3
"""Reproduce Figure 10 (geometry vs. knowledge-prediction gap correlation).

1. Build verified per-(model,dataset) rows from configs/<model>/KAPPA/*.yaml.
2. Recompute ACC/AGR/KLD for the base and knowprobe result logs and derive the
   knowprobe-vs-base gap (the paper's 1-AGR gap).
3. Load the best-knowprobe-layer probe weights and compute the mean principal
   angle between the knowledge and prediction probes.
4. Correlate the mean principal angle against the AGR gap (Spearman/Pearson)
   and render the Figure 10 scatter plot.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kappa_core.geometry import (
    center_weights,
    compute_principal_angles,
    load_probe_weights,
    MODEL_CONFIGS,
    PairingLogicError,
)
from geometry._common import (
    DATASET_CORRELATION_DATASET_KEYS, DEFAULT_CONFIG_DIR,
    DEFAULT_OUTPUT_ROOT, DEFAULT_PLOT_MODEL_KEYS,
    result_dirs,
)
from geometry._manifest import build_rows, attach_gap_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("geometry_gap_correlation")

PLOT_METRIC_FILENAME_MAP = {
    "d_proj_norm": "dproj",
    "mean_principal_angle_deg": "mean_angle",
    "gap_acc": "acc_gap",
    "gap_agr": "agr_gap",
    "gap_kld": "kld_gap",
}


def torch_rank(matrix) -> int:
    import torch

    return int(torch.linalg.matrix_rank(matrix).item())


def compute_geometry_metrics(W_know, W_pred) -> Dict[str, float]:
    W_know_c = center_weights(W_know)
    W_pred_c = center_weights(W_pred)
    result = compute_principal_angles(W_know_c, W_pred_c)

    rank_know = torch_rank(W_know_c)
    rank_pred = torch_rank(W_pred_c)
    if rank_know == 0 or rank_pred == 0:
        return {
            "rank_know": rank_know,
            "rank_pred": rank_pred,
            "d_proj": np.nan,
            "d_proj_norm": np.nan,
            "mean_principal_angle_deg": np.nan,
            "max_principal_angle_deg": np.nan,
        }

    d_proj = result["proj_frob_dist"]
    d_proj_norm = d_proj / math.sqrt(max(1, rank_know + rank_pred))
    return {
        "rank_know": rank_know,
        "rank_pred": rank_pred,
        "d_proj": d_proj,
        "d_proj_norm": round(d_proj_norm, 6),
        "mean_principal_angle_deg": result["mean_angle_deg"],
        "max_principal_angle_deg": result["max_angle_deg"],
    }


def replace_path_part(path: Path, old: str, new: str) -> Path:
    parts = list(path.parts)
    replaced = False
    for index, part in enumerate(parts):
        if part == old:
            parts[index] = new
            replaced = True
            break
    if not replaced:
        raise PairingLogicError(f"Could not replace path part '{old}' -> '{new}' in {path}")
    return Path(*parts)


def build_verified_rows_from_final_config(
    config_dir: Path,
    output_root: Path,
    models: Optional[Sequence[str]],
    datasets: Optional[Sequence[str]],
    allow_partial: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    verified_rows, audit_rows, manifest_df = build_rows(
        config_dir=config_dir,
        output_root=output_root,
        models=models,
        datasets=datasets,
        allow_partial=allow_partial,
    )
    return verified_rows, audit_rows, manifest_df


def build_geometry_rows(verified_rows: List[Dict[str, Any]], show_progress: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    row_iter = tqdm(verified_rows, desc="Computing geometry metrics") if show_progress else verified_rows
    for verified in row_iter:
        model_key = verified["raw_model"]
        dataset_key = verified["raw_dataset"]
        if model_key not in MODEL_CONFIGS:
            raise PairingLogicError(f"{model_key}/{dataset_key}: MODEL_CONFIGS is missing this model.")

        d = MODEL_CONFIGS[model_key]["d"]
        k = int(verified["num_options"])

        knowprobe_metadata_path = Path(verified["path_knowprobe"])
        answer_model_path = knowprobe_metadata_path.parent / "model_SoftmaxClassifier.pt"
        pred_metadata_path = replace_path_part(knowprobe_metadata_path, "answer", "pred")
        pred_model_path = pred_metadata_path.parent / "model_SoftmaxClassifier.pt"

        answer_config_paths = sorted(answer_model_path.parent.glob("config_*.json"))
        pred_config_paths = sorted(pred_model_path.parent.glob("config_*.json"))

        if not answer_model_path.exists():
            raise PairingLogicError(f"{model_key}/{dataset_key}: knowprobe answer model is missing: {answer_model_path}")
        if not pred_model_path.exists():
            raise PairingLogicError(f"{model_key}/{dataset_key}: knowprobe pred model is missing: {pred_model_path}")
        if not answer_config_paths:
            raise PairingLogicError(
                f"{model_key}/{dataset_key}: knowprobe answer directory has no config_*.json: {answer_model_path.parent}"
            )
        if not pred_config_paths:
            raise PairingLogicError(
                f"{model_key}/{dataset_key}: knowprobe pred directory has no config_*.json: {pred_model_path.parent}"
            )

        W_know = load_probe_weights(answer_model_path, d, k)
        W_pred = load_probe_weights(pred_model_path, d, k)
        geom = compute_geometry_metrics(W_know, W_pred)

        row = dict(verified)
        row["answer_model_path"] = str(answer_model_path)
        row["pred_model_path"] = str(pred_model_path)
        row["answer_config_count"] = len(answer_config_paths)
        row["pred_config_count"] = len(pred_config_paths)
        row["rank_know"] = geom["rank_know"]
        row["rank_pred"] = geom["rank_pred"]
        row["d_proj"] = geom["d_proj"]
        row["d_proj_norm"] = geom["d_proj_norm"]
        row["mean_principal_angle_deg"] = geom["mean_principal_angle_deg"]
        row["max_principal_angle_deg"] = geom["max_principal_angle_deg"]
        rows.append(row)

    return rows


def get_task_family(dataset_name: str) -> str:
    dataset_lower = dataset_name.lower()
    if any(key in dataset_lower for key in ["gsm8k", "math", "aqua", "svamp", "asdiv", "bbh"]):
        return "reasoning"
    if any(key in dataset_lower for key in ["truthfulqa", "bbq", "crows", "winobias"]):
        return "truthfulness/bias"
    if any(key in dataset_lower for key in ["mmlu", "triviaqa", "nq", "openbookqa", "arc"]):
        return "knowledge"
    return "other"


def build_final_dataframe(geometry_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = geometry_df.copy()
    merged_df["task_family"] = merged_df["dataset"].apply(get_task_family)
    return merged_df


def build_scatter_plot_filename(x_metric: str, y_metric: str) -> str:
    x_name = PLOT_METRIC_FILENAME_MAP.get(x_metric, x_metric)
    y_name = PLOT_METRIC_FILENAME_MAP.get(y_metric, y_metric)
    return f"{x_name}_vs_{y_name}_scatter.png"


def _save_plot_outputs(fig: plt.Figure, output_path: Path) -> None:
    base_path = output_path.with_suffix("")
    for suffix in (".png", ".pdf"):
        target_path = base_path.with_suffix(suffix)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=300, bbox_inches="tight")


def build_knowprobe_gap_dataframe(merged_df: pd.DataFrame) -> pd.DataFrame:
    gap_df = merged_df.copy()
    gap_df["analysis"] = "knowprobe_gap"
    gap_df["gap_acc"] = gap_df.get("gap_acc_knowprobe_vs_base")
    gap_df["gap_agr"] = gap_df.get("gap_agr_knowprobe_vs_base")
    gap_df["gap_kld"] = gap_df.get("gap_kld_knowprobe_vs_base")
    return gap_df


def compute_correlations(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, n_perm: int = 1000) -> Dict[str, float]:
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 3:
        return {
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "spearman_ci_low": np.nan,
            "spearman_ci_high": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "pearson_ci_low": np.nan,
            "pearson_ci_high": np.nan,
        }

    spearman_obs, _ = stats.spearmanr(x, y)
    pearson_obs, _ = stats.pearsonr(x, y)
    if np.isnan(spearman_obs):
        spearman_obs = 0.0
    if np.isnan(pearson_obs):
        pearson_obs = 0.0

    spearman_boot = []
    pearson_boot = []
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)
        if len(np.unique(x[indices])) > 1 and len(np.unique(y[indices])) > 1:
            spearman_value = stats.spearmanr(x[indices], y[indices])[0]
            pearson_value = stats.pearsonr(x[indices], y[indices])[0]
            if not np.isnan(spearman_value):
                spearman_boot.append(spearman_value)
            if not np.isnan(pearson_value):
                pearson_boot.append(pearson_value)

    spearman_ci = (
        np.percentile(spearman_boot, 2.5),
        np.percentile(spearman_boot, 97.5),
    ) if spearman_boot else (np.nan, np.nan)
    pearson_ci = (
        np.percentile(pearson_boot, 2.5),
        np.percentile(pearson_boot, 97.5),
    ) if pearson_boot else (np.nan, np.nan)

    spearman_perm = []
    pearson_perm = []
    y_perm = y.copy()
    for _ in range(n_perm):
        np.random.shuffle(y_perm)
        spearman_value = stats.spearmanr(x, y_perm)[0]
        pearson_value = stats.pearsonr(x, y_perm)[0]
        if not np.isnan(spearman_value):
            spearman_perm.append(spearman_value)
        if not np.isnan(pearson_value):
            pearson_perm.append(pearson_value)

    spearman_p = np.mean(np.abs(spearman_perm) >= np.abs(spearman_obs)) if spearman_perm else np.nan
    pearson_p = np.mean(np.abs(pearson_perm) >= np.abs(pearson_obs)) if pearson_perm else np.nan

    return {
        "spearman_rho": round(float(spearman_obs), 4),
        "spearman_p": round(float(spearman_p), 4),
        "spearman_ci_low": round(float(spearman_ci[0]), 4),
        "spearman_ci_high": round(float(spearman_ci[1]), 4),
        "pearson_r": round(float(pearson_obs), 4),
        "pearson_p": round(float(pearson_p), 4),
        "pearson_ci_low": round(float(pearson_ci[0]), 4),
        "pearson_ci_high": round(float(pearson_ci[1]), 4),
    }


def _build_group_label_map(group_cols: Sequence[str], group_key: Any) -> Dict[str, Any]:
    if len(group_cols) == 1:
        return {group_cols[0]: group_key}
    return dict(zip(group_cols, group_key))


def _save_grouped_scatter_plots(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    plots_dir: Path,
    group_cols: Sequence[str],
    x_vars: Sequence[str],
    y_vars: Sequence[str],
    path_builder,
    title_builder,
) -> None:
    if df.empty or corr_df.empty:
        return

    model_colors = {
        "Llama-3.1_8B": sns.color_palette("Set2", n_colors=2)[0],
        "Qwen2.5_7B": sns.color_palette("Set2", n_colors=2)[1],
    }
    model_markers = {
        "Llama-3.1_8B": "o",
        "Qwen2.5_7B": "s",
    }

    sns.set_theme(style="whitegrid")
    raw_model_col = "raw_model" if "raw_model" in df.columns else "model"
    display_model_col = "model" if "model" in df.columns else raw_model_col

    for group_key, subset in df.groupby(list(group_cols), sort=True):
        labels = _build_group_label_map(group_cols, group_key)
        raw_model = str(subset[raw_model_col].iloc[0])
        display_model = str(subset[display_model_col].iloc[0])
        plot_dir = path_builder(plots_dir, labels)
        plot_dir.mkdir(parents=True, exist_ok=True)
        color = model_colors.get(raw_model, sns.color_palette("Set2", n_colors=1)[0])
        marker = model_markers.get(raw_model, "o")

        for x_metric in x_vars:
            for y_metric in y_vars:
                if y_metric not in subset.columns:
                    continue

                mask = pd.Series(True, index=corr_df.index)
                for col_name, value in labels.items():
                    mask &= corr_df[col_name] == value
                mask &= corr_df["x_metric"] == x_metric
                mask &= corr_df["y_metric"] == y_metric
                matched = corr_df[mask]
                if matched.empty:
                    continue

                row = matched.iloc[0]
                rho = row["spearman_rho"]
                p_value = row["spearman_p"]

                plt.figure(figsize=(8, 6))
                ax = sns.scatterplot(
                    data=subset,
                    x=x_metric,
                    y=y_metric,
                    color=color,
                    marker=marker,
                    s=100,
                    alpha=0.8,
                )

                for _, plot_row in subset.iterrows():
                    if pd.notna(plot_row[x_metric]) and pd.notna(plot_row[y_metric]):
                        plt.annotate(
                            plot_row["dataset"],
                            (plot_row[x_metric], plot_row[y_metric]),
                            fontsize=8,
                            alpha=0.6,
                            xytext=(5, 5),
                            textcoords="offset points",
                        )

                valid_mask = ~(subset[x_metric].isna() | subset[y_metric].isna())
                if valid_mask.any():
                    sns.regplot(
                        data=subset[valid_mask],
                        x=x_metric,
                        y=y_metric,
                        scatter=False,
                        line_kws={"color": "gray", "linestyle": "--", "alpha": 0.5},
                        ax=ax,
                    )

                plt.title(title_builder(display_model, labels, x_metric, y_metric, rho, p_value))
                plt.tight_layout()
                plot_path = plot_dir / build_scatter_plot_filename(x_metric, y_metric)
                _save_plot_outputs(plt.gcf(), plot_path)
                plt.close()
                LOG.info("Saved scatter plot: %s", plot_path)


def save_gap_scatter_plots(gap_df: pd.DataFrame, corr_df: pd.DataFrame, plots_dir: Path) -> None:
    # Keep only the single scatter plot we currently use for fast preview iterations.
    x_vars = ["mean_principal_angle_deg"]
    y_vars = ["gap_agr"]
    _save_grouped_scatter_plots(
        gap_df,
        corr_df,
        plots_dir,
        group_cols=["analysis", "raw_model", "model"],
        x_vars=x_vars,
        y_vars=y_vars,
        path_builder=lambda root, labels: root / str(labels["analysis"]) / str(labels["raw_model"]),
        title_builder=lambda display_model, labels, x_metric, y_metric, rho, p_value: (
            f"{display_model} | knowprobe gap: {y_metric} vs {x_metric}\n"
            f"Spearman rho = {rho:.3f} (p={p_value:.3f})"
        ),
    )


def build_correlation_summary(
    method_df: pd.DataFrame,
    group_cols: Sequence[str],
    x_vars: Sequence[str],
    y_vars: Sequence[str],
    show_progress: bool = False,
) -> pd.DataFrame:
    corr_rows: List[Dict[str, Any]] = []
    grouped = method_df.groupby(list(group_cols), sort=True)
    group_iter = (
        tqdm(grouped, total=grouped.ngroups, desc="Computing correlation summaries")
        if show_progress else grouped
    )
    for group_key, method_subset in group_iter:
        labels = _build_group_label_map(group_cols, group_key)
        for x_metric in x_vars:
            for y_metric in y_vars:
                if y_metric not in method_subset.columns:
                    continue
                corr = compute_correlations(method_subset[x_metric].values, method_subset[y_metric].values)
                corr.update(labels)
                corr["x_metric"] = x_metric
                corr["y_metric"] = y_metric
                corr_rows.append(corr)

    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        return corr_df

    return corr_df[
        list(group_cols)
        + [
            "x_metric",
            "y_metric",
            "spearman_rho",
            "spearman_p",
            "spearman_ci_low",
            "spearman_ci_high",
            "pearson_r",
            "pearson_p",
            "pearson_ci_low",
            "pearson_ci_high",
        ]
    ].copy()


def filter_plot_method_df(
    method_df: pd.DataFrame,
    requested_models: Optional[Sequence[str]],
    requested_datasets: Optional[Sequence[str]],
) -> pd.DataFrame:
    if method_df.empty:
        return method_df.copy()

    model_targets = set(requested_models) if requested_models else set(DEFAULT_PLOT_MODEL_KEYS)
    dataset_targets = set(requested_datasets) if requested_datasets else set(DATASET_CORRELATION_DATASET_KEYS)

    model_col = "raw_model" if "raw_model" in method_df.columns else "model"
    dataset_col = "raw_dataset" if "raw_dataset" in method_df.columns else "dataset"
    return method_df[
        method_df[model_col].isin(model_targets) & method_df[dataset_col].isin(dataset_targets)
    ].copy()


def write_audit_summary(audit_df: pd.DataFrame, out_path: Path) -> None:
    verified_df = audit_df[audit_df["status"] == "verified"].copy()
    error_df = audit_df[audit_df["status"] != "verified"].copy()

    lines = [
        "# Row Verification Summary",
        "",
        f"- total_rows: {len(audit_df)}",
        f"- verified_rows: {len(verified_df)}",
        f"- excluded_rows: {len(error_df)}",
        "",
        "## Verified Rows",
    ]

    if verified_df.empty:
        lines.append("- none")
    else:
        for _, row in verified_df.sort_values(["raw_model", "raw_dataset"]).iterrows():
            lines.append(
                f"- {row['raw_model']} / {row['raw_dataset']} | best_layer={row.get('best_knowprobe_layer', row.get('selected_layer', ''))} | "
                f"base={row.get('path_base', '')} | knowprobe={row.get('path_knowprobe', '')} | kappa={row.get('path_kappa', '')}"
            )

    lines.extend(["", "## Excluded Rows"])
    if error_df.empty:
        lines.append("- none")
    else:
        for _, row in error_df.sort_values(["raw_model", "raw_dataset"]).iterrows():
            lines.append(f"- {row['raw_model']} / {row['raw_dataset']} | {row['message']}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven dataset-level geometry vs. knowledge-prediction gap correlation (Figure 10)."
    )
    parser.set_defaults(allow_partial=True)
    parser.add_argument("--models", nargs="*", default=None, help="Optional model filter.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter.")
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
        help=f"Outputs root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "geometry" / "results" / "dataset_correlation",
        help="Output directory (default: geometry/results/dataset_correlation)",
    )
    parser.add_argument(
        "--allow-partial",
        dest="allow_partial",
        action="store_true",
        help="Skip incomplete bundles and keep validated rows (default).",
    )
    parser.add_argument(
        "--strict",
        dest="allow_partial",
        action="store_false",
        help="Abort after writing audit outputs if any config-driven bundle is missing or invalid.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dirs = result_dirs(args.out_dir)
    effective_datasets = list(args.datasets) if args.datasets else list(DATASET_CORRELATION_DATASET_KEYS)
    LOG.info("Using dataset-correlation dataset set: %s", ", ".join(effective_datasets))

    verified_rows, audit_rows, manifest_df = build_verified_rows_from_final_config(
        config_dir=args.config_dir,
        output_root=args.output_root,
        models=args.models,
        datasets=effective_datasets,
        allow_partial=args.allow_partial,
    )

    manifest_path = out_dirs["tables"] / "final_config_manifest_paths.csv"
    manifest_df.to_csv(manifest_path, index=False)
    LOG.info("Saved final-config manifest paths to %s", manifest_path)

    audit_df = pd.DataFrame(audit_rows)
    audit_path = out_dirs["tables"] / "row_verification_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    LOG.info("Saved row verification audit to %s", audit_path)

    failure_path = out_dirs["tables"] / "row_verification_failures.csv"
    excluded_df = audit_df[audit_df["status"] != "verified"].copy()
    excluded_df.to_csv(failure_path, index=False)
    LOG.info("Saved row verification failures to %s", failure_path)

    summary_path = out_dirs["reports"] / "row_verification_summary.md"
    write_audit_summary(audit_df, summary_path)
    LOG.info("Saved row verification summary to %s", summary_path)

    if not excluded_df.empty:
        LOG.warning(
            "Skipping %d final-config bundle(s) that failed path verification. See %s",
            len(excluded_df),
            failure_path,
        )
        if not args.allow_partial:
            raise PairingLogicError(
                f"{len(excluded_df)} final-config bundle(s) failed path verification. "
                f"Inspect {failure_path} for the exact missing-path or manifest errors."
            )

    if not verified_rows:
        raise PairingLogicError(
            "No final-config-derived rows survived path loading. "
            f"Inspect {failure_path} for the exact missing-path or manifest errors."
        )

    verified_rows = attach_gap_metrics(verified_rows)

    verified_df = pd.DataFrame(verified_rows)
    verified_path = out_dirs["tables"] / "verified_result_paths.csv"
    verified_df.to_csv(verified_path, index=False)
    LOG.info("Saved verified result paths to %s", verified_path)

    geometry_rows = build_geometry_rows(verified_rows, show_progress=True)
    geometry_df = pd.DataFrame(geometry_rows)
    geometry_path = out_dirs["tables"] / "geometry_metrics_audited.csv"
    geometry_df.to_csv(geometry_path, index=False)
    LOG.info("Saved audited geometry metrics to %s", geometry_path)

    performance_cols = [
        "model",
        "dataset",
        "raw_model",
        "raw_dataset",
        "best_knowprobe_layer",
        "selected_layer",
        "path_base",
        "path_knowprobe",
        "acc_base",
        "agr_base",
        "kld_base",
        "acc_knowprobe",
        "agr_knowprobe",
        "kld_knowprobe",
        "gap_acc_knowprobe_vs_base",
        "gap_agr_knowprobe_vs_base",
        "gap_kld_knowprobe_vs_base",
    ]
    available_performance_cols = [column for column in performance_cols if column in geometry_df.columns]
    performance_path = out_dirs["tables"] / "computed_performance_from_paths.csv"
    geometry_df.loc[:, available_performance_cols].to_csv(performance_path, index=False)
    LOG.info("Saved config-derived performance metrics to %s", performance_path)

    merged_df = build_final_dataframe(geometry_df)
    gap_df = build_knowprobe_gap_dataframe(merged_df)
    merged_path = out_dirs["tables"] / "geometry_gap_gain_correlation.csv"
    merged_df.to_csv(merged_path, index=False)
    LOG.info("Saved wide geometry/gap table to %s", merged_path)

    x_vars = ["mean_principal_angle_deg"]
    gap_y_vars = ["gap_agr"]
    gap_corr_df = build_correlation_summary(
        gap_df,
        group_cols=["analysis", "raw_model", "model"],
        x_vars=x_vars,
        y_vars=gap_y_vars,
        show_progress=True,
    )

    corr_path = out_dirs["tables"] / "geometry_gap_gain_correlation_summary.csv"
    gap_corr_df.to_csv(corr_path, index=False)
    LOG.info("Saved knowprobe-gap correlation summary to %s", corr_path)

    plot_gap_df = filter_plot_method_df(gap_df, requested_models=args.models, requested_datasets=effective_datasets)
    plot_gap_corr_df = build_correlation_summary(
        plot_gap_df,
        group_cols=["analysis", "raw_model", "model"],
        x_vars=x_vars,
        y_vars=gap_y_vars,
        show_progress=True,
    )
    save_gap_scatter_plots(plot_gap_df, plot_gap_corr_df, out_dirs["plots"])

    print(f"Done. Output directory: {out_dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
