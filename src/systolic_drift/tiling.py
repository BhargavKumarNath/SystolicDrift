"""Matrix tiling, edge padding, and M/N/K loop traversal schedules."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import DTypeLike

SUPPORTED_LOOP_ORDERS = ("M-N-K", "K-M-N", "K-N-M")


@dataclass(frozen=True)
class TileSpec:
    """Coordinates and bounds for one matrix multiply tile."""

    m_idx: int
    n_idx: int
    k_idx: int
    m_start: int
    m_end: int
    n_start: int
    n_end: int
    k_start: int
    k_end: int
    is_first_k: bool
    is_last_k: bool

    @property
    def m_size(self) -> int:
        return self.m_end - self.m_start

    @property
    def n_size(self) -> int:
        return self.n_end - self.n_start

    @property
    def k_size(self) -> int:
        return self.k_end - self.k_start


def compute_tile_ranges(total: int, tile_size: int) -> list[tuple[int, int]]:
    """Split a dimension of length total into contiguous [start, end) slices."""
    if total <= 0:
        raise ValueError(f"Dimension length must be positive, got {total}")
    if tile_size <= 0:
        raise ValueError(f"Tile size must be positive, got {tile_size}")
    return [(start, min(start + tile_size, total)) for start in range(0, total, tile_size)]


def infer_tile_dims(
    array_rows: int,
    array_cols: int,
    dataflow: str,
    tile_m: int | None = None,
    tile_k: int | None = None,
    tile_n: int | None = None,
) -> tuple[int, int, int]:
    """Validate and resolve (tile_m, tile_k, tile_n) for the physical PE grid."""
    if array_rows <= 0 or array_cols <= 0:
        raise ValueError(f"Array dimensions must be positive, got ({array_rows}, {array_cols})")

    tm = array_rows if tile_m is None else tile_m
    tk = array_rows if tile_k is None else tile_k
    tn = array_cols if tile_n is None else tile_n

    if tm <= 0 or tk <= 0 or tn <= 0:
        raise ValueError(f"Tile dimensions must be positive, got ({tm}, {tk}, {tn})")

    if dataflow == "weight_stationary":
        if tk > array_rows:
            raise ValueError(
                f"Weight-stationary tile_k ({tk}) cannot exceed array_rows ({array_rows})"
            )
        if tn > array_cols:
            raise ValueError(
                f"Weight-stationary tile_n ({tn}) cannot exceed array_cols ({array_cols})"
            )
    elif dataflow == "output_stationary":
        if tm > array_rows:
            raise ValueError(
                f"Output-stationary tile_m ({tm}) cannot exceed array_rows ({array_rows})"
            )
        if tn > array_cols:
            raise ValueError(
                f"Output-stationary tile_n ({tn}) cannot exceed array_cols ({array_cols})"
            )
    else:
        raise ValueError(f"Unsupported dataflow '{dataflow}'")

    return tm, tk, tn


def generate_tile_schedule(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    loop_order: str = "M-N-K",
) -> Iterator[TileSpec]:
    """Yield TileSpec items in the requested M/N/K loop traversal order."""
    if loop_order not in SUPPORTED_LOOP_ORDERS:
        raise ValueError(
            f"Unsupported loop_order '{loop_order}'. Supported: {SUPPORTED_LOOP_ORDERS}"
        )

    m_ranges = compute_tile_ranges(m_dim, tile_m)
    k_ranges = compute_tile_ranges(k_dim, tile_k)
    n_ranges = compute_tile_ranges(n_dim, tile_n)
    last_k_idx = len(k_ranges) - 1

    def make_spec(mi: int, ni: int, ki: int) -> TileSpec:
        ms, me = m_ranges[mi]
        ns, ne = n_ranges[ni]
        ks, ke = k_ranges[ki]
        return TileSpec(
            m_idx=mi,
            n_idx=ni,
            k_idx=ki,
            m_start=ms,
            m_end=me,
            n_start=ns,
            n_end=ne,
            k_start=ks,
            k_end=ke,
            is_first_k=(ki == 0),
            is_last_k=(ki == last_k_idx),
        )

    if loop_order == "M-N-K":
        for mi in range(len(m_ranges)):
            for ni in range(len(n_ranges)):
                for ki in range(len(k_ranges)):
                    yield make_spec(mi, ni, ki)
    elif loop_order == "K-M-N":
        for ki in range(len(k_ranges)):
            for mi in range(len(m_ranges)):
                for ni in range(len(n_ranges)):
                    yield make_spec(mi, ni, ki)
    elif loop_order == "K-N-M":
        for ki in range(len(k_ranges)):
            for ni in range(len(n_ranges)):
                for mi in range(len(m_ranges)):
                    yield make_spec(mi, ni, ki)


def pad_tile_2d(
    tile: np.ndarray,
    target_rows: int,
    target_cols: int,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Zero-pad a 2D slice into (target_rows, target_cols)."""
    rows, cols = tile.shape
    if rows > target_rows or cols > target_cols:
        raise ValueError(
            f"Tile shape {tile.shape} exceeds target ({target_rows}, {target_cols})"
        )
    padded = np.zeros((target_rows, target_cols), dtype=dtype)
    padded[:rows, :cols] = tile
    return padded
