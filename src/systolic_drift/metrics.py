"""Numerical error metrics, outlier tail checks, and simulation evaluation."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from systolic_drift.dtypes import quantize_tensor
from systolic_drift.simulator import SimulationResult


@dataclass(frozen=True)
class DriftReport:
    """Complete numerical drift and hardware execution summary for one run."""

    cosine_sim: float
    rel_l2_error: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    top1p_rel_l2: float
    top1p_max_abs: float
    top1p_mean_abs: float
    top1p_count: int
    non_finite_count: int
    total_cycles: int
    wavefront_cycles: int
    active_cycles: int
    preload_cycles: int
    drain_cycles: int
    num_tiles: int
    num_macs: int
    utilization: float
    weight_loads: int
    output_drains: int
    addition_order_tag: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report fields into a flat dictionary for CSV/JSON export."""
        return asdict(self)


def _validate_pair(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return np.asarray(pred, dtype=np.float64).ravel(), np.asarray(ref, dtype=np.float64).ravel()


def compute_fp64_reference(
    a: np.ndarray,
    b: np.ndarray,
    input_dtype: str | None = None,
) -> np.ndarray:
    """Compute an FP64 reference matmul, optionally after one-time input quantization."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shapes {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")

    if input_dtype is not None:
        a_q = quantize_tensor(a, input_dtype).astype(np.float64)
        b_q = quantize_tensor(b, input_dtype).astype(np.float64)
    else:
        a_q = np.asarray(a, dtype=np.float64)
        b_q = np.asarray(b, dtype=np.float64)
    return a_q @ b_q


def cosine_similarity(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute flattened cosine similarity between pred and ref in float64."""
    p, r = _validate_pair(pred, ref)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return float("nan")

    norm_p = float(np.linalg.norm(p))
    norm_r = float(np.linalg.norm(r))
    if norm_p == 0.0 and norm_r == 0.0:
        return 1.0
    if norm_p == 0.0 or norm_r == 0.0:
        return 0.0

    cos = float(np.dot(p, r) / (norm_p * norm_r))
    return float(np.clip(cos, -1.0, 1.0))


def relative_l2_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute ||pred - ref||_2 / ||ref||_2 in float64."""
    p, r = _validate_pair(pred, ref)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return float("nan")

    diff_norm = float(np.linalg.norm(p - r))
    ref_norm = float(np.linalg.norm(r))
    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return diff_norm / ref_norm


def max_abs_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute max(|pred - ref|) in float64."""
    p, r = _validate_pair(pred, ref)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return float("nan")
    return float(np.max(np.abs(p - r)))


def mean_abs_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute mean(|pred - ref|) in float64."""
    p, r = _validate_pair(pred, ref)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return float("nan")
    return float(np.mean(np.abs(p - r)))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute root-mean-square error in float64."""
    p, r = _validate_pair(pred, ref)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(p - r))))


def top_k_percent_outlier_error(
    pred: np.ndarray,
    ref: np.ndarray,
    top_fraction: float = 0.01,
) -> dict[str, float | int]:
    """Measure error restricted to the top-fraction highest-magnitude reference elements."""
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    p, r = _validate_pair(pred, ref)
    k = max(1, int(np.ceil(top_fraction * r.size)))
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        return {
            "top_rel_l2": float("nan"),
            "top_max_abs": float("nan"),
            "top_mean_abs": float("nan"),
            "top_count": int(k),
        }

    idx = np.argpartition(np.abs(r), -k)[-k:]
    p_top = p[idx]
    r_top = r[idx]

    diff = p_top - r_top
    diff_norm = float(np.linalg.norm(diff))
    ref_norm = float(np.linalg.norm(r_top))
    if ref_norm == 0.0:
        rel_l2 = 0.0 if diff_norm == 0.0 else float("inf")
    else:
        rel_l2 = diff_norm / ref_norm

    return {
        "top_rel_l2": rel_l2,
        "top_max_abs": float(np.max(np.abs(diff))),
        "top_mean_abs": float(np.mean(np.abs(diff))),
        "top_count": int(k),
    }


def evaluate_simulation(result: SimulationResult, ref: np.ndarray) -> DriftReport:
    """Evaluate a SimulationResult against an FP64 reference output matrix."""
    pred = result.output
    stats = result.stats
    cfg = result.config

    non_finite_count = int(np.size(pred) - np.sum(np.isfinite(pred)))
    top_stats = top_k_percent_outlier_error(pred, ref, top_fraction=0.01)

    total_pe_cycles = stats.total_cycles * cfg.array_rows * cfg.array_cols
    utilization = float(stats.num_macs / total_pe_cycles) if total_pe_cycles > 0 else 0.0

    return DriftReport(
        cosine_sim=cosine_similarity(pred, ref),
        rel_l2_error=relative_l2_error(pred, ref),
        max_abs_error=max_abs_error(pred, ref),
        mean_abs_error=mean_abs_error(pred, ref),
        rmse=rmse(pred, ref),
        top1p_rel_l2=float(top_stats["top_rel_l2"]),
        top1p_max_abs=float(top_stats["top_max_abs"]),
        top1p_mean_abs=float(top_stats["top_mean_abs"]),
        top1p_count=int(top_stats["top_count"]),
        non_finite_count=non_finite_count,
        total_cycles=stats.total_cycles,
        wavefront_cycles=stats.wavefront_cycles,
        active_cycles=stats.active_cycles,
        preload_cycles=stats.preload_cycles,
        drain_cycles=stats.drain_cycles,
        num_tiles=stats.num_tiles,
        num_macs=stats.num_macs,
        utilization=utilization,
        weight_loads=stats.weight_loads,
        output_drains=stats.output_drains,
        addition_order_tag=stats.addition_order_tag,
    )
