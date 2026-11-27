"""Precision casting, quantization, and MAC rounding primitives."""

from typing import Callable

import numpy as np

SUPPORTED_INPUT_DTYPES = ("fp64", "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2")
SUPPORTED_ACC_DTYPES = ("fp64", "fp32", "fp16", "bf16")

_DTYPE_ALIASES = {
    "float64": "fp64",
    "fp64": "fp64",
    "double": "fp64",
    "float32": "fp32",
    "fp32": "fp32",
    "float": "fp32",
    "float16": "fp16",
    "fp16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "e5m2": "fp8_e5m2",
}


def normalize_dtype_name(name: str) -> str:
    """Normalize precision name aliases to canonical identifiers."""
    key = name.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype '{name}'. Supported: {SUPPORTED_INPUT_DTYPES}")
    return _DTYPE_ALIASES[key]


def round_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Round array to IEEE bfloat16 using round-to-nearest-even, returned as float32."""
    f32 = np.asarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded = (u32 + bias) & np.uint32(0xFFFF0000)
    if np.isnan(f32).any():
        is_nan = (u32 & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
        rounded = np.where(is_nan, (u32 & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded)
    return rounded.view(np.float32)


def round_to_fp16(arr: np.ndarray) -> np.ndarray:
    """Round array to IEEE float16 using round-to-nearest-even, returned as float32."""
    return np.asarray(arr, dtype=np.float32).astype(np.float16).astype(np.float32)


def _round_to_custom_float(
    arr: np.ndarray,
    exp_bits: int,
    mant_bits: int,
    has_inf: bool,
    saturate: bool,
) -> np.ndarray:
    """Round float32 values to a narrow IEEE-like floating format with subnormals."""
    f32 = np.asarray(arr, dtype=np.float32)
    shape = f32.shape
    flat = f32.ravel()
    u32 = flat.view(np.uint32)

    sign = (u32 >> np.uint32(31)) & np.uint32(1)
    abs_u = u32 & np.uint32(0x7FFFFFFF)

    bias_out = (1 << (exp_bits - 1)) - 1
    max_exp_field = (1 << exp_bits) - 1

    if has_inf:
        max_norm_exp = max_exp_field - 1
        max_val = (2.0 - 2.0 ** (-mant_bits)) * (2.0 ** (max_norm_exp - bias_out))
    else:
        max_norm_exp = max_exp_field
        max_mant = ((1 << mant_bits) - 2) / (1 << mant_bits)
        max_val = (1.0 + max_mant) * (2.0 ** (max_norm_exp - bias_out))

    min_normal = 2.0 ** (1 - bias_out)
    subnormal_step = 2.0 ** (1 - bias_out - mant_bits)

    abs_f = np.abs(flat)
    out = np.zeros_like(flat, dtype=np.float32)

    is_nan = abs_u > np.uint32(0x7F800000)
    is_inf = abs_u == np.uint32(0x7F800000)
    is_zero = abs_u == np.uint32(0)

    is_normal = (~is_nan) & (~is_inf) & (abs_f >= min_normal) & (abs_f <= max_val)
    is_subnormal = (~is_nan) & (~is_inf) & (~is_zero) & (abs_f < min_normal)
    is_overflow = (~is_nan) & (~is_inf) & (abs_f > max_val)

    if np.any(is_normal):
        shift = 23 - mant_bits
        mask = np.uint32(0xFFFFFFFF ^ ((1 << shift) - 1))
        u_norm = abs_u[is_normal]
        lsb = (u_norm >> np.uint32(shift)) & np.uint32(1)
        bias = np.uint32((1 << (shift - 1)) - 1) + lsb
        rounded_u = (u_norm + bias) & mask
        res_norm = rounded_u.view(np.float32)
        if saturate:
            res_norm = np.minimum(res_norm, np.float32(max_val))
        out[is_normal] = res_norm

    if np.any(is_subnormal):
        scaled = abs_f[is_subnormal] / np.float32(subnormal_step)
        rounded_int = np.rint(scaled)
        out[is_subnormal] = (rounded_int * np.float32(subnormal_step)).astype(np.float32)

    if np.any(is_overflow):
        if has_inf and not saturate:
            out[is_overflow] = np.float32(np.inf)
        else:
            out[is_overflow] = np.float32(max_val)

    if np.any(is_inf):
        if has_inf:
            out[is_inf] = np.float32(np.inf)
        else:
            out[is_inf] = np.float32(max_val) if saturate else np.float32(np.nan)

    if np.any(is_nan):
        out[is_nan] = np.float32(np.nan)

    out_u32 = out.view(np.uint32) | (sign << np.uint32(31))
    return out_u32.view(np.float32).reshape(shape)


def round_to_fp8_e4m3(arr: np.ndarray, saturate: bool = True) -> np.ndarray:
    """Quantize values to OCP FP8 E4M3FN (max finite 448.0), returned as float32."""
    return _round_to_custom_float(arr, exp_bits=4, mant_bits=3, has_inf=False, saturate=saturate)


def round_to_fp8_e5m2(arr: np.ndarray, saturate: bool = True) -> np.ndarray:
    """Quantize values to IEEE-like FP8 E5M2 (max finite 57344.0), returned as float32."""
    return _round_to_custom_float(arr, exp_bits=5, mant_bits=2, has_inf=True, saturate=saturate)


def quantize_tensor(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Quantize a tensor once to the target input precision."""
    canonical = normalize_dtype_name(dtype)
    if canonical == "fp64":
        return np.asarray(arr, dtype=np.float64)
    if canonical == "fp32":
        return np.asarray(arr, dtype=np.float32)
    if canonical == "fp16":
        return round_to_fp16(arr)
    if canonical == "bf16":
        return round_to_bf16(arr)
    if canonical == "fp8_e4m3":
        return round_to_fp8_e4m3(arr, saturate=True)
    if canonical == "fp8_e5m2":
        return round_to_fp8_e5m2(arr, saturate=True)
    raise ValueError(f"Unsupported input dtype '{dtype}'")


def _cast_fp32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _cast_fp64(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def get_acc_caster(acc_dtype: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorized rounding function for the accumulator precision."""
    canonical = normalize_dtype_name(acc_dtype)
    if canonical not in SUPPORTED_ACC_DTYPES:
        raise ValueError(
            f"Unsupported accumulator dtype '{acc_dtype}'. Supported: {SUPPORTED_ACC_DTYPES}"
        )
    if canonical == "fp64":
        return _cast_fp64
    if canonical == "fp32":
        return _cast_fp32
    if canonical == "fp16":
        return round_to_fp16
    if canonical == "bf16":
        return round_to_bf16
    raise ValueError(f"Unsupported accumulator dtype '{acc_dtype}'")


def mac_step(
    acc: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    acc_cast: Callable[[np.ndarray], np.ndarray],
    fma: bool = False,
) -> np.ndarray:
    """Execute one multiply-accumulate step with explicit accumulator rounding."""
    if fma:
        wide = acc.astype(np.float64) + a.astype(np.float64) * b.astype(np.float64)
        return acc_cast(wide)
    prod = acc_cast(a * b)
    return acc_cast(acc + prod)
