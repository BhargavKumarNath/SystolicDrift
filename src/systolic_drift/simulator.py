"""Cycle-stepped 2D systolic array simulator for weight-stationary and output-stationary dataflows."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from systolic_drift.dtypes import (
    get_acc_caster,
    mac_step,
    normalize_dtype_name,
    quantize_tensor,
)
from systolic_drift.tiling import (
    SUPPORTED_LOOP_ORDERS,
    generate_tile_schedule,
    infer_tile_dims,
    pad_tile_2d,
)

DataflowType = Literal["weight_stationary", "output_stationary"]
LoopOrderType = Literal["M-N-K", "K-M-N", "K-N-M"]


@dataclass(frozen=True)
class SystolicConfig:
    """Hardware geometry, dataflow, loop schedule, and precision configuration."""

    array_rows: int = 8
    array_cols: int = 8
    dataflow: DataflowType = "weight_stationary"
    loop_order: LoopOrderType = "M-N-K"
    input_dtype: str = "bf16"
    acc_dtype: str = "fp32"
    tile_m: int | None = None
    tile_k: int | None = None
    tile_n: int | None = None
    fma: bool = False
    quantize_inputs: bool = True


@dataclass(frozen=True)
class SimulationStats:
    """Cycle counts, tile counts, and execution metadata from one matmul run."""

    total_cycles: int
    wavefront_cycles: int
    active_cycles: int
    preload_cycles: int
    drain_cycles: int
    num_tiles: int
    num_macs: int
    weight_loads: int
    output_drains: int
    tile_shape: tuple[int, int, int]
    addition_order_tag: str


@dataclass
class SimulationResult:
    """Output matrix together with execution stats and configuration."""

    output: np.ndarray
    stats: SimulationStats
    config: SystolicConfig


class SystolicArraySimulator:
    """Cycle-stepped 2D PE grid simulator for matrix multiplication C = A @ B."""

    def __init__(self, config: SystolicConfig) -> None:
        if config.dataflow not in ("weight_stationary", "output_stationary"):
            raise ValueError(f"Unsupported dataflow '{config.dataflow}'")
        if config.loop_order not in SUPPORTED_LOOP_ORDERS:
            raise ValueError(f"Unsupported loop_order '{config.loop_order}'")

        self.config = config
        self.input_dtype = normalize_dtype_name(config.input_dtype)
        self.acc_dtype = normalize_dtype_name(config.acc_dtype)
        self.acc_cast: Callable[[np.ndarray], np.ndarray] = get_acc_caster(self.acc_dtype)
        self.work_dtype = np.float64 if self.acc_dtype == "fp64" else np.float32
        self.tile_dims = infer_tile_dims(
            array_rows=config.array_rows,
            array_cols=config.array_cols,
            dataflow=config.dataflow,
            tile_m=config.tile_m,
            tile_k=config.tile_k,
            tile_n=config.tile_n,
        )

    def matmul(self, a: np.ndarray, b: np.ndarray) -> SimulationResult:
        """Compute C = A @ B on the cycle-stepped 2D systolic array."""
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D matrices, got shapes {a.shape} and {b.shape}")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")

        if self.config.quantize_inputs:
            a_work = quantize_tensor(a, self.input_dtype).astype(self.work_dtype, copy=False)
            b_work = quantize_tensor(b, self.input_dtype).astype(self.work_dtype, copy=False)
        else:
            a_work = np.asarray(a, dtype=self.work_dtype)
            b_work = np.asarray(b, dtype=self.work_dtype)

        if self.config.dataflow == "weight_stationary":
            return self._run_weight_stationary(a_work, b_work)
        return self._run_output_stationary(a_work, b_work)

    def _run_weight_stationary(self, a: np.ndarray, b: np.ndarray) -> SimulationResult:
        m_dim, k_dim = a.shape
        _, n_dim = b.shape
        rows = self.config.array_rows
        tm, tk, tn = self.tile_dims

        c_acc = np.zeros((m_dim, n_dim), dtype=self.work_dtype)

        wavefront_cycles = 0
        active_cycles = 0
        preload_cycles = 0
        num_tiles = 0
        weight_loads = 0
        prev_kn: tuple[int, int] | None = None

        for tile in generate_tile_schedule(
            m_dim, k_dim, n_dim, tm, tk, tn, loop_order=self.config.loop_order
        ):
            kn_key = (tile.k_idx, tile.n_idx)
            if kn_key != prev_kn:
                weight_loads += 1
                preload_cycles += rows
                prev_kn = kn_key

            a_tile = a[tile.m_start : tile.m_end, tile.k_start : tile.k_end]
            b_tile = b[tile.k_start : tile.k_end, tile.n_start : tile.n_end]

            tile_psum, wf_cyc, act_cyc = self._step_ws_tile(a_tile, b_tile)
            wavefront_cycles += wf_cyc
            active_cycles += act_cyc
            num_tiles += 1

            c_slice = c_acc[tile.m_start : tile.m_end, tile.n_start : tile.n_end]
            c_acc[tile.m_start : tile.m_end, tile.n_start : tile.n_end] = self.acc_cast(
                c_slice + tile_psum
            )

        stats = SimulationStats(
            total_cycles=preload_cycles + wavefront_cycles,
            wavefront_cycles=wavefront_cycles,
            active_cycles=active_cycles,
            preload_cycles=preload_cycles,
            drain_cycles=0,
            num_tiles=num_tiles,
            num_macs=m_dim * k_dim * n_dim,
            weight_loads=weight_loads,
            output_drains=0,
            tile_shape=self.tile_dims,
            addition_order_tag=f"ws_chunked_k{tk}",
        )
        return SimulationResult(output=c_acc, stats=stats, config=self.config)

    def _run_output_stationary(self, a: np.ndarray, b: np.ndarray) -> SimulationResult:
        m_dim, k_dim = a.shape
        _, n_dim = b.shape
        rows, cols = self.config.array_rows, self.config.array_cols
        tm, tk, tn = self.tile_dims

        c_acc = np.zeros((m_dim, n_dim), dtype=self.work_dtype)

        wavefront_cycles = 0
        active_cycles = 0
        drain_cycles = 0
        num_tiles = 0
        output_drains = 0

        keep_stationary_across_k = self.config.loop_order == "M-N-K"
        pe_acc = np.zeros((rows, cols), dtype=self.work_dtype)

        for tile in generate_tile_schedule(
            m_dim, k_dim, n_dim, tm, tk, tn, loop_order=self.config.loop_order
        ):
            a_tile = a[tile.m_start : tile.m_end, tile.k_start : tile.k_end]
            b_tile = b[tile.k_start : tile.k_end, tile.n_start : tile.n_end]

            if keep_stationary_across_k:
                if tile.is_first_k:
                    pe_acc.fill(0)
                pe_acc, wf_cyc, act_cyc = self._step_os_tile(a_tile, b_tile, pe_acc)
                wavefront_cycles += wf_cyc
                active_cycles += act_cyc
                num_tiles += 1

                if tile.is_last_k:
                    c_acc[tile.m_start : tile.m_end, tile.n_start : tile.n_end] = pe_acc[
                        : tile.m_size, : tile.n_size
                    ]
                    output_drains += 1
                    drain_cycles += rows
            else:
                pe_acc.fill(0)
                pe_acc, wf_cyc, act_cyc = self._step_os_tile(a_tile, b_tile, pe_acc)
                wavefront_cycles += wf_cyc
                active_cycles += act_cyc
                num_tiles += 1

                tile_psum = pe_acc[: tile.m_size, : tile.n_size]
                c_slice = c_acc[tile.m_start : tile.m_end, tile.n_start : tile.n_end]
                c_acc[tile.m_start : tile.m_end, tile.n_start : tile.n_end] = self.acc_cast(
                    c_slice + tile_psum
                )
                output_drains += 1
                drain_cycles += rows

        order_tag = "os_seq_k" if keep_stationary_across_k else f"os_chunked_k{tk}"
        stats = SimulationStats(
            total_cycles=wavefront_cycles + drain_cycles,
            wavefront_cycles=wavefront_cycles,
            active_cycles=active_cycles,
            preload_cycles=0,
            drain_cycles=drain_cycles,
            num_tiles=num_tiles,
            num_macs=m_dim * k_dim * n_dim,
            weight_loads=0,
            output_drains=output_drains,
            tile_shape=self.tile_dims,
            addition_order_tag=order_tag,
        )
        return SimulationResult(output=c_acc, stats=stats, config=self.config)

    def _step_ws_tile(
        self, a_tile: np.ndarray, b_tile: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        """Execute one Weight-Stationary tile cycle-by-cycle on the (R, C) PE grid."""
        m_t, k_t = a_tile.shape
        _, n_t = b_tile.shape
        rows, cols = self.config.array_rows, self.config.array_cols
        fma = self.config.fma
        acc_cast = self.acc_cast

        w_grid = pad_tile_2d(b_tile, rows, cols, dtype=self.work_dtype)
        a_pad = pad_tile_2d(a_tile, m_t, rows, dtype=self.work_dtype)

        sched_len = m_t + rows - 1
        a_sched = np.zeros((sched_len, rows), dtype=self.work_dtype)
        for r in range(rows):
            a_sched[r : r + m_t, r] = a_pad[:, r]

        a_reg = np.zeros((rows, cols), dtype=self.work_dtype)
        psum_reg = np.zeros((rows, cols), dtype=self.work_dtype)
        out_sched = np.zeros((m_t + cols - 1, cols), dtype=self.work_dtype)

        total_t = m_t + rows + cols - 2
        active_cyc = m_t + k_t + n_t - 2
        last_row = rows - 1

        for t in range(total_t):
            if t < sched_len:
                a_reg[:, 0] = a_sched[t]
            else:
                a_reg[:, 0] = 0
            psum_reg[0, :] = 0

            psum_out = mac_step(psum_reg, a_reg, w_grid, acc_cast, fma=fma)

            if t >= last_row:
                out_sched[t - last_row] = psum_out[last_row]

            a_reg[:, 1:] = a_reg[:, :-1]
            psum_reg[1:, :] = psum_out[:-1, :]

        out = np.empty((m_t, n_t), dtype=self.work_dtype)
        for c in range(n_t):
            out[:, c] = out_sched[c : c + m_t, c]

        return out, total_t, active_cyc

    def _step_os_tile(
        self,
        a_tile: np.ndarray,
        b_tile: np.ndarray,
        pe_acc: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Execute one Output-Stationary tile cycle-by-cycle on the (R, C) PE grid."""
        m_t, k_t = a_tile.shape
        _, n_t = b_tile.shape
        rows, cols = self.config.array_rows, self.config.array_cols
        fma = self.config.fma
        acc_cast = self.acc_cast

        a_pad = pad_tile_2d(a_tile, rows, k_t, dtype=self.work_dtype)
        b_pad = pad_tile_2d(b_tile, k_t, cols, dtype=self.work_dtype)

        sched_a_len = k_t + rows - 1
        a_sched = np.zeros((sched_a_len, rows), dtype=self.work_dtype)
        for r in range(rows):
            a_sched[r : r + k_t, r] = a_pad[r, :]

        sched_b_len = k_t + cols - 1
        b_sched = np.zeros((sched_b_len, cols), dtype=self.work_dtype)
        for c in range(cols):
            b_sched[c : c + k_t, c] = b_pad[:, c]

        a_reg = np.zeros((rows, cols), dtype=self.work_dtype)
        b_reg = np.zeros((rows, cols), dtype=self.work_dtype)

        total_t = k_t + rows + cols - 2
        active_cyc = k_t + m_t + n_t - 2

        for t in range(total_t):
            if t < sched_a_len:
                a_reg[:, 0] = a_sched[t]
            else:
                a_reg[:, 0] = 0

            if t < sched_b_len:
                b_reg[0, :] = b_sched[t]
            else:
                b_reg[0, :] = 0

            pe_acc = mac_step(pe_acc, a_reg, b_reg, acc_cast, fma=fma)

            a_reg[:, 1:] = a_reg[:, :-1]
            b_reg[1:, :] = b_reg[:-1, :]

        return pe_acc, total_t, active_cyc
