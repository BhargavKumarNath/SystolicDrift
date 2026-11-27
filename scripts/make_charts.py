"""Generate publication-ready Matplotlib charts from saved ablation results."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "slate": "#4D4D4D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts from SystolicDrift CSV results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=REPO_ROOT / "results" / "ablation_summary.csv",
        help="Path to ablation_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory to save generated PNG figures.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    int_cols = {
        "m_dim",
        "k_dim",
        "n_dim",
        "array_rows",
        "array_cols",
        "top1p_count",
        "non_finite_count",
        "total_cycles",
        "wavefront_cycles",
        "active_cycles",
        "preload_cycles",
        "drain_cycles",
        "num_tiles",
        "num_macs",
        "weight_loads",
        "output_drains",
    }
    float_cols = {
        "cosine_sim",
        "rel_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "rmse",
        "top1p_rel_l2",
        "top1p_max_abs",
        "top1p_mean_abs",
        "utilization",
        "e2e_cosine_sim",
        "e2e_rel_l2_error",
        "e2e_top1p_rel_l2",
        "wall_time_sec",
    }

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            parsed: dict[str, Any] = dict(raw)
            for col in int_cols:
                if col in parsed:
                    parsed[col] = int(parsed[col])
            for col in float_cols:
                if col in parsed:
                    parsed[col] = float(parsed[col])
            rows.append(parsed)
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_precision_and_dataflow(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Bar chart comparing relative L2 drift across accumulator precision and dataflow schedules."""
    layers = sorted({r["layer"] for r in rows})
    acc_order = ("fp32", "fp16", "bf16")
    schedule_specs = (
        ("WS 8x8 (chunk k=8)", "8x8", "weight_stationary", "M-N-K", PALETTE["blue"]),
        ("OS 8x8 (seq K)", "8x8", "output_stationary", "M-N-K", PALETTE["vermillion"]),
        ("OS 8x8 (chunk k=8)", "8x8", "output_stationary", "K-M-N", PALETTE["orange"]),
        ("WS 16x16 (chunk k=16)", "16x16", "weight_stationary", "M-N-K", PALETTE["sky"]),
        ("OS 16x16 (seq K)", "16x16", "output_stationary", "M-N-K", PALETTE["purple"]),
    )

    fig, axes = plt.subplots(1, len(layers), figsize=(12.5, 4.8), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    x = np.arange(len(acc_order))
    width = 0.15

    for ax, layer in zip(axes, layers):
        sub = [r for r in rows if r["layer"] == layer and r["input_dtype"] == "bf16"]
        for idx, (label, arr_sz, df, order, color) in enumerate(schedule_specs):
            vals = []
            for acc in acc_order:
                match = [
                    r
                    for r in sub
                    if r["array_size"] == arr_sz
                    and r["dataflow"] == df
                    and r["loop_order"] == order
                    and r["acc_dtype"] == acc
                ]
                val = match[0]["rel_l2_error"] if match else np.nan
                vals.append(max(val, 1e-8) if np.isfinite(val) else np.nan)

            offset = (idx - (len(schedule_specs) - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.92)
            for bar, v in zip(bars, vals):
                if np.isfinite(v) and v > 1e-4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        v * 1.12,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_yscale("log")
        ax.set_ylim(1e-7, 8e-2)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in acc_order])
        ax.set_xlabel("Accumulator Precision")
        ax.set_title(f"Layer: {layer} (BF16 Inputs)")

    axes[0].set_ylabel("Relative L2 Error vs. Quantized FP64 Ref (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(schedule_specs),
        fontsize=8.5,
    )
    fig.suptitle(
        "Accumulation Precision Dominates Drift, While Schedule Shapes Narrow-Precision Error",
        y=1.09,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = out_dir / "drift_by_precision_and_dataflow.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_drift_vs_cycles_pareto(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Scatter plot of total clock cycles vs. relative L2 drift across schedules and array sizes."""
    target_layer = "mlp_down_proj" if any(r["layer"] == "mlp_down_proj" for r in rows) else rows[0]["layer"]
    sub = [
        r
        for r in rows
        if r["layer"] == target_layer
        and r["input_dtype"] == "bf16"
        and r["acc_dtype"] in ("fp16", "bf16")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    acc_panels = ("fp16", "bf16")
    schedule_markers = {
        ("weight_stationary", "M-N-K"): ("o", PALETTE["blue"], "WS M-N-K"),
        ("weight_stationary", "K-M-N"): ("s", PALETTE["sky"], "WS K-M-N"),
        ("weight_stationary", "K-N-M"): ("^", PALETTE["green"], "WS K-N-M (weight reuse)"),
        ("output_stationary", "M-N-K"): ("D", PALETTE["vermillion"], "OS M-N-K (seq K)"),
        ("output_stationary", "K-M-N"): ("P", PALETTE["orange"], "OS K-M-N (chunk K)"),
    }

    for ax, acc in zip(axes, acc_panels):
        panel_rows = [r for r in sub if r["acc_dtype"] == acc]
        for (df, order), (marker, color, label) in schedule_markers.items():
            for arr_sz, fill_alpha, edge in (("8x8", 0.9, "black"), ("16x16", 0.55, color)):
                pts = [
                    r
                    for r in panel_rows
                    if r["dataflow"] == df and r["loop_order"] == order and r["array_size"] == arr_sz
                ]
                if not pts:
                    continue
                cyc = [p["total_cycles"] for p in pts]
                err = [p["rel_l2_error"] for p in pts]
                lbl = f"{label} [{arr_sz}]" if acc == "fp16" else None
                ax.scatter(
                    cyc,
                    err,
                    marker=marker,
                    s=95 if arr_sz == "8x8" else 115,
                    color=color,
                    alpha=fill_alpha,
                    edgecolors=edge,
                    linewidths=0.9,
                    label=lbl,
                )
                for p in pts:
                    ax.annotate(
                        arr_sz,
                        (p["total_cycles"], p["rel_l2_error"]),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7.5,
                        color=PALETTE["slate"],
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total Clock Cycles (log scale)")
        ax.set_title(f"{acc.upper()} Accumulation ({target_layer}, BF16 Inputs)")

    axes[0].set_ylabel("Relative L2 Error (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        fontsize=7.8,
    )
    fig.suptitle(
        "Accuracy vs. Latency Tradeoff Across Dataflow Schedules and Loop Orders",
        y=1.15,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = out_dir / "drift_vs_cycles_pareto.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_outlier_and_quantization_breakdown(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Two-panel view of top-1% outlier drift and input-quantization vs. accumulation drift."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left panel: Overall Rel L2 vs Top-1% Tail Rel L2 for BF16 inputs on 8x8
    sub_left = [
        r
        for r in rows
        if r["array_size"] == "8x8"
        and r["input_dtype"] == "bf16"
        and r["loop_order"] == "M-N-K"
    ]
    combos = (
        ("attn_q_proj", "weight_stationary", "o", PALETTE["blue"], "attn WS"),
        ("attn_q_proj", "output_stationary", "D", PALETTE["sky"], "attn OS-seq"),
        ("mlp_down_proj", "weight_stationary", "s", PALETTE["orange"], "mlp WS"),
        ("mlp_down_proj", "output_stationary", "^", PALETTE["vermillion"], "mlp OS-seq"),
    )
    all_vals = []
    for layer, df, marker, color, label in combos:
        pts = [r for r in sub_left if r["layer"] == layer and r["dataflow"] == df]
        if not pts:
            continue
        xs = [max(p["rel_l2_error"], 1e-8) for p in pts]
        ys = [max(p["top1p_rel_l2"], 1e-8) for p in pts]
        all_vals.extend(xs + ys)
        ax_left.scatter(xs, ys, marker=marker, s=85, color=color, edgecolors="black", linewidths=0.7, label=label)
        for p, xv, yv in zip(pts, xs, ys):
            ax_left.annotate(
                p["acc_dtype"].upper(),
                (xv, yv),
                textcoords="offset points",
                xytext=(5, -3),
                fontsize=7.5,
            )

    if all_vals:
        lo = min(all_vals) * 0.6
        hi = max(all_vals) * 1.6
        ax_left.plot([lo, hi], [lo, hi], linestyle=":", color=PALETTE["slate"], label="y = x parity")
        ax_left.set_xlim(lo, hi)
        ax_left.set_ylim(lo, hi)

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("Overall Relative L2 Error (log scale)")
    ax_left.set_ylabel("Top-1% Magnitude Tail Relative L2 Error (log scale)")
    ax_left.set_title("Tail Outlier Error vs. Overall L2 Error (8x8, M-N-K)")
    ax_left.legend(fontsize=8, loc="upper left")

    # Right panel: Accumulation-only drift vs Total End-to-End drift (quantization + accumulation)
    target_layer = "mlp_down_proj" if any(r["layer"] == "mlp_down_proj" for r in rows) else rows[0]["layer"]
    pairs = (
        ("bf16", "fp32", "BF16 in\nFP32 acc"),
        ("bf16", "fp16", "BF16 in\nFP16 acc"),
        ("bf16", "bf16", "BF16 in\nBF16 acc"),
        ("fp8_e4m3", "fp32", "FP8 in\nFP32 acc"),
        ("fp8_e4m3", "fp16", "FP8 in\nFP16 acc"),
        ("fp8_e4m3", "bf16", "FP8 in\nBF16 acc"),
    )
    acc_only_vals = []
    e2e_vals = []
    tick_labels = []
    for in_dt, acc_dt, lbl in pairs:
        match = [
            r
            for r in rows
            if r["layer"] == target_layer
            and r["array_size"] == "8x8"
            and r["dataflow"] == "output_stationary"
            and r["loop_order"] == "M-N-K"
            and r["input_dtype"] == in_dt
            and r["acc_dtype"] == acc_dt
        ]
        if not match:
            continue
        acc_only_vals.append(max(match[0]["rel_l2_error"], 1e-8))
        e2e_vals.append(max(match[0]["e2e_rel_l2_error"], 1e-8))
        tick_labels.append(lbl)

    x_pos = np.arange(len(tick_labels))
    bar_w = 0.36
    ax_right.bar(
        x_pos - bar_w / 2.0,
        acc_only_vals,
        width=bar_w,
        color=PALETTE["blue"],
        label="Accumulation Drift (vs. Quantized FP64 Ref)",
    )
    ax_right.bar(
        x_pos + bar_w / 2.0,
        e2e_vals,
        width=bar_w,
        color=PALETTE["orange"],
        label="Total End-to-End Drift (vs. Unquantized FP64 Ref)",
    )
    ax_right.set_yscale("log")
    ax_right.set_ylim(1e-7, 2e-1)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(tick_labels, fontsize=8.5)
    ax_right.set_ylabel("Relative L2 Error (log scale)")
    ax_right.set_title(f"Input Quantization vs. Accumulation Drift ({target_layer}, OS 8x8)")
    ax_right.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Tail Stability and Precision Budget Breakdown Across Layers",
        y=1.03,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = out_dir / "tail_outlier_and_quantization_breakdown.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    rows = load_rows(args.results_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.results_csv}")

    paths = [
        plot_precision_and_dataflow(rows, args.output_dir),
        plot_drift_vs_cycles_pareto(rows, args.output_dir),
        plot_outlier_and_quantization_breakdown(rows, args.output_dir),
    ]
    for path in paths:
        print(f"Generated figure: {path}")


if __name__ == "__main__":
    main()
