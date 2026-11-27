"""Run the controlled SystolicDrift ablation grid and save CSV/JSON artifacts."""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from systolic_drift import (  # noqa: E402
    LayerWorkload,
    SystolicArraySimulator,
    SystolicConfig,
    compute_fp64_reference,
    cosine_similarity,
    evaluate_simulation,
    load_workloads,
    quantize_tensor,
    relative_l2_error,
    top_k_percent_outlier_error,
)

ARRAY_SIZES: tuple[tuple[int, int], ...] = ((8, 8), (16, 16))
DATAFLOWS: tuple[str, ...] = ("weight_stationary", "output_stationary")
LOOP_ORDERS: tuple[str, ...] = ("M-N-K", "K-M-N", "K-N-M")
INPUT_DTYPES: tuple[str, ...] = ("bf16", "fp8_e4m3")
ACC_DTYPES: tuple[str, ...] = ("fp32", "fp16", "bf16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SystolicDrift controlled ablation sweep.")
    parser.add_argument(
        "--source",
        choices=("auto", "huggingface", "synthetic"),
        default="auto",
        help="Workload source (default: auto with synthetic fallback).",
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace causal LM identifier.",
    )
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length (M dimension).")
    parser.add_argument("--max-k", type=int, default=256, help="Max K slice (0 for full).")
    parser.add_argument("--max-n", type=int, default=256, help="Max N slice (0 for full).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory to write ablation_summary.csv and ablation_details.json.",
    )
    return parser.parse_args()


def summarize_workload(wl: LayerWorkload, ref_unq: np.ndarray) -> dict[str, Any]:
    act = wl.activations.astype(np.float64)
    wt = wl.weights.astype(np.float64)
    act_rms = float(np.sqrt(np.mean(np.square(act))))
    act_max = float(np.max(np.abs(act)))
    return {
        "name": wl.name,
        "model_id": wl.model_id,
        "source": wl.source,
        "shape_m": int(wl.shape[0]),
        "shape_k": int(wl.shape[1]),
        "shape_n": int(wl.shape[2]),
        "act_mean": float(np.mean(act)),
        "act_std": float(np.std(act)),
        "act_rms": act_rms,
        "act_max_abs": act_max,
        "act_crest_factor": (act_max / act_rms) if act_rms > 0.0 else 0.0,
        "weight_std": float(np.std(wt)),
        "weight_max_abs": float(np.max(np.abs(wt))),
        "ref_out_rms": float(np.sqrt(np.mean(np.square(ref_unq)))),
        "ref_out_max_abs": float(np.max(np.abs(ref_unq))),
        "metadata": wl.metadata,
    }


def build_highlights(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute concise summary statistics for downstream reporting."""
    by_acc: dict[str, list[float]] = {}
    for row in runs:
        by_acc.setdefault(row["acc_dtype"], []).append(row["rel_l2_error"])

    mean_rel_l2_by_acc = {
        acc: float(np.mean(vals)) for acc, vals in sorted(by_acc.items())
    }
    max_rel_l2_by_acc = {
        acc: float(np.max(vals)) for acc, vals in sorted(by_acc.items())
    }

    bf16_acc_runs = [r for r in runs if r["acc_dtype"] == "bf16" and r["input_dtype"] == "bf16"]
    fastest_bf16 = min(bf16_acc_runs, key=lambda r: r["total_cycles"]) if bf16_acc_runs else None
    lowest_drift_bf16 = (
        min(bf16_acc_runs, key=lambda r: r["rel_l2_error"]) if bf16_acc_runs else None
    )

    return {
        "total_runs": len(runs),
        "mean_rel_l2_by_acc_dtype": mean_rel_l2_by_acc,
        "max_rel_l2_by_acc_dtype": max_rel_l2_by_acc,
        "fastest_bf16_acc_config": fastest_bf16,
        "lowest_drift_bf16_acc_config": lowest_drift_bf16,
    }


def main() -> None:
    args = parse_args()
    max_k = None if args.max_k <= 0 else args.max_k
    max_n = None if args.max_n <= 0 else args.max_n

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start_all = time.perf_counter()

    print(
        f"Loading workloads (source={args.source}, model={args.model_id}, "
        f"seq_len={args.seq_len}, max_k={max_k}, max_n={max_n})..."
    )
    workloads = load_workloads(
        source=args.source,
        model_id=args.model_id,
        seq_len=args.seq_len,
        batch_size=1,
        max_k=max_k,
        max_n=max_n,
        seed=args.seed,
    )

    workload_summaries: dict[str, dict[str, Any]] = {}
    all_runs: list[dict[str, Any]] = []

    total_runs = (
        len(workloads)
        * len(INPUT_DTYPES)
        * len(ARRAY_SIZES)
        * len(DATAFLOWS)
        * len(LOOP_ORDERS)
        * len(ACC_DTYPES)
    )
    run_idx = 0

    for layer_name, wl in workloads.items():
        ref_unq = compute_fp64_reference(wl.activations, wl.weights, input_dtype=None)
        workload_summaries[layer_name] = summarize_workload(wl, ref_unq)
        print(
            f"Layer {layer_name}: source={wl.source}, shape={wl.shape}, "
            f"act_max={workload_summaries[layer_name]['act_max_abs']:.3f}, "
            f"crest={workload_summaries[layer_name]['act_crest_factor']:.2f}"
        )

        for in_dt in INPUT_DTYPES:
            # Quantize inputs once per input precision so all hardware schedules see identical values
            a_q = quantize_tensor(wl.activations, in_dt)
            b_q = quantize_tensor(wl.weights, in_dt)
            ref_q = compute_fp64_reference(a_q, b_q, input_dtype=None)

            for rows, cols in ARRAY_SIZES:
                for df in DATAFLOWS:
                    for order in LOOP_ORDERS:
                        for acc_dt in ACC_DTYPES:
                            run_idx += 1
                            cfg = SystolicConfig(
                                array_rows=rows,
                                array_cols=cols,
                                dataflow=df,  # type: ignore[arg-type]
                                loop_order=order,  # type: ignore[arg-type]
                                input_dtype=in_dt,
                                acc_dtype=acc_dt,
                                quantize_inputs=False,
                            )
                            t0 = time.perf_counter()
                            sim_res = SystolicArraySimulator(cfg).matmul(a_q, b_q)
                            wall_sec = time.perf_counter() - t0

                            report = evaluate_simulation(sim_res, ref_q)
                            e2e_top = top_k_percent_outlier_error(
                                sim_res.output, ref_unq, top_fraction=0.01
                            )

                            row: dict[str, Any] = {
                                "layer": layer_name,
                                "model_id": wl.model_id,
                                "workload_source": wl.source,
                                "m_dim": wl.shape[0],
                                "k_dim": wl.shape[1],
                                "n_dim": wl.shape[2],
                                "array_size": f"{rows}x{cols}",
                                "array_rows": rows,
                                "array_cols": cols,
                                "dataflow": df,
                                "loop_order": order,
                                "input_dtype": in_dt,
                                "acc_dtype": acc_dt,
                                **report.to_dict(),
                                "e2e_cosine_sim": cosine_similarity(sim_res.output, ref_unq),
                                "e2e_rel_l2_error": relative_l2_error(sim_res.output, ref_unq),
                                "e2e_top1p_rel_l2": float(e2e_top["top_rel_l2"]),
                                "wall_time_sec": round(wall_sec, 4),
                            }
                            all_runs.append(row)

                            if run_idx % 12 == 0 or run_idx == total_runs:
                                print(
                                    f"[{run_idx:3d}/{total_runs}] {layer_name:13s} "
                                    f"{rows}x{cols} {df[:2].upper()} {order} "
                                    f"in={in_dt:8s} acc={acc_dt:4s} "
                                    f"rel_l2={report.rel_l2_error:.6f} "
                                    f"cyc={report.total_cycles}"
                                )

    csv_path = args.output_dir / "ablation_summary.csv"
    fieldnames = list(all_runs[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_runs)

    total_wall = round(time.perf_counter() - t_start_all, 3)
    details_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "args": {
                "source": args.source,
                "model_id": args.model_id,
                "seq_len": args.seq_len,
                "max_k": max_k,
                "max_n": max_n,
                "seed": args.seed,
            },
            "total_wall_time_sec": total_wall,
        },
        "workloads": workload_summaries,
        "highlights": build_highlights(all_runs),
        "runs": all_runs,
    }
    json_path = args.output_dir / "ablation_details.json"
    with open(json_path, "w") as f:
        json.dump(details_payload, f, indent=2)

    print(f"Completed {len(all_runs)} runs in {total_wall:.2f}s.")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
