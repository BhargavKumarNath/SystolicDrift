"""Unit, invariance, numerical-drift, and cycle-accounting tests for SystolicDrift."""

import numpy as np
import pytest

from systolic_drift import (
    SystolicArraySimulator,
    SystolicConfig,
    compute_fp64_reference,
    compute_tile_ranges,
    evaluate_simulation,
    generate_synthetic_workloads,
    generate_tile_schedule,
    load_workloads,
    normalize_dtype_name,
    pad_tile_2d,
    quantize_tensor,
    round_to_bf16,
    round_to_fp8_e4m3,
    round_to_fp8_e5m2,
    round_to_fp16,
)
from systolic_drift.simulator import DataflowType, LoopOrderType


class TestDtypes:
    def test_normalize_aliases(self) -> None:
        assert normalize_dtype_name("float32") == "fp32"
        assert normalize_dtype_name("bfloat16") == "bf16"
        assert normalize_dtype_name("Half") == "fp16"
        assert normalize_dtype_name("fp8") == "fp8_e4m3"
        assert normalize_dtype_name("E5M2") == "fp8_e5m2"
        with pytest.raises(ValueError):
            normalize_dtype_name("int4")

    def test_bf16_round_to_nearest_even_and_special_values(self) -> None:
        # 1.0 + 0.5 ulp (LSB even -> rounds down to 1.0)
        even_tie = np.uint32(0x3F808000).view(np.float32)
        # 1.0 + 1.5 ulp (LSB odd -> rounds up to 1.0 + 2 ulp)
        odd_tie = np.uint32(0x3F818000).view(np.float32)
        arr = np.array([0.0, -0.0, even_tie, odd_tie, np.inf, -np.inf, np.nan], dtype=np.float32)
        out = round_to_bf16(arr)
        bits = out.view(np.uint32)

        assert bits[0] == 0x00000000
        assert bits[1] == 0x80000000
        assert bits[2] == 0x3F800000
        assert bits[3] == 0x3F820000
        assert np.isposinf(out[4])
        assert np.isneginf(out[5])
        assert np.isnan(out[6])

    def test_fp16_saturation_and_subnormals(self) -> None:
        arr = np.array([1e-6, 1.0, 65504.0, 70000.0, -70000.0], dtype=np.float32)
        out = round_to_fp16(arr)
        assert np.isclose(out[0], 1e-6, atol=1e-7)
        assert out[1] == 1.0
        assert out[2] == 65504.0
        assert np.isposinf(out[3])
        assert np.isneginf(out[4])

    def test_fp8_e4m3_and_e5m2_quantization(self) -> None:
        vals = np.array(
            [0.0, -0.0, 0.001, 0.001953125, 1.0625, 1.1875, 500.0, -500.0],
            dtype=np.float32,
        )
        e4m3 = round_to_fp8_e4m3(vals)
        assert e4m3[0] == 0.0
        assert np.signbit(e4m3[1])
        assert e4m3[3] == np.float32(0.001953125)
        assert e4m3[4] == np.float32(1.0)
        assert e4m3[5] == np.float32(1.25)
        assert e4m3[6] == np.float32(448.0)
        assert e4m3[7] == np.float32(-448.0)

        e5m2 = round_to_fp8_e5m2(np.array([1.0, 500.0, 100000.0], dtype=np.float32))
        assert e5m2[0] == np.float32(1.0)
        assert e5m2[1] == np.float32(512.0)
        assert e5m2[2] == np.float32(57344.0)


class TestTiling:
    def test_compute_tile_ranges_non_multiple(self) -> None:
        assert compute_tile_ranges(19, 8) == [(0, 8), (8, 16), (16, 19)]
        assert compute_tile_ranges(8, 8) == [(0, 8)]
        with pytest.raises(ValueError):
            compute_tile_ranges(0, 8)
        with pytest.raises(ValueError):
            compute_tile_ranges(8, 0)

    def test_tile_schedule_loop_orders(self) -> None:
        sched_mnk = list(generate_tile_schedule(16, 16, 16, 8, 8, 8, loop_order="M-N-K"))
        sched_kmn = list(generate_tile_schedule(16, 16, 16, 8, 8, 8, loop_order="K-M-N"))
        sched_knm = list(generate_tile_schedule(16, 16, 16, 8, 8, 8, loop_order="K-N-M"))

        assert len(sched_mnk) == len(sched_kmn) == len(sched_knm) == 8
        assert [(t.m_idx, t.n_idx, t.k_idx) for t in sched_mnk[:3]] == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
        ]
        assert [(t.k_idx, t.m_idx, t.n_idx) for t in sched_kmn[:3]] == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
        ]
        assert [(t.k_idx, t.n_idx, t.m_idx) for t in sched_knm[:3]] == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
        ]
        assert sched_mnk[0].is_first_k and not sched_mnk[0].is_last_k
        assert not sched_mnk[1].is_first_k and sched_mnk[1].is_last_k

    def test_pad_tile_2d_bounds(self) -> None:
        tile = np.ones((3, 5), dtype=np.float32)
        padded = pad_tile_2d(tile, 8, 8)
        assert padded.shape == (8, 8)
        assert np.sum(padded) == 15.0
        with pytest.raises(ValueError):
            pad_tile_2d(tile, 2, 8)


class TestSystolicSimulator:
    @pytest.mark.parametrize("array_size", [(8, 8), (16, 16)])
    @pytest.mark.parametrize("dataflow", ["weight_stationary", "output_stationary"])
    @pytest.mark.parametrize("loop_order", ["M-N-K", "K-M-N", "K-N-M"])
    def test_single_tile_fp32_matches_reference(
        self, array_size: tuple[int, int], dataflow: DataflowType, loop_order: LoopOrderType
    ) -> None:
        rows, cols = array_size
        rng = np.random.default_rng(101)
        a = rng.standard_normal((rows, rows)).astype(np.float32)
        b = rng.standard_normal((rows, cols)).astype(np.float32)

        cfg = SystolicConfig(
            array_rows=rows,
            array_cols=cols,
            dataflow=dataflow,
            loop_order=loop_order,
            input_dtype="fp32",
            acc_dtype="fp32",
        )
        res = SystolicArraySimulator(cfg).matmul(a, b)
        ref = compute_fp64_reference(a, b, input_dtype="fp32")
        report = evaluate_simulation(res, ref)

        assert res.stats.num_tiles == 1
        assert report.rel_l2_error < 1e-6
        assert report.cosine_sim > 0.9999999

    @pytest.mark.parametrize("acc_dtype", ["fp32", "bf16", "fp16"])
    def test_single_tile_ws_and_os_are_bitwise_identical(self, acc_dtype: str) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 8)).astype(np.float32)
        b = rng.standard_normal((8, 8)).astype(np.float32)

        res_ws = SystolicArraySimulator(
            SystolicConfig(8, 8, "weight_stationary", "M-N-K", "bf16", acc_dtype)
        ).matmul(a, b)
        res_os = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "M-N-K", "bf16", acc_dtype)
        ).matmul(a, b)

        assert np.array_equal(res_ws.output, res_os.output)

    @pytest.mark.parametrize(
        "shape,array_shape",
        [
            ((1, 1, 1), (8, 8)),
            ((1, 17, 1), (8, 8)),
            ((13, 29, 17), (8, 8)),
            ((19, 37, 23), (16, 16)),
            ((9, 15, 11), (4, 8)),
            ((11, 13, 7), (8, 4)),
        ],
    )
    @pytest.mark.parametrize("dataflow", ["weight_stationary", "output_stationary"])
    def test_non_multiple_and_rectangular_shapes(
        self,
        shape: tuple[int, int, int],
        array_shape: tuple[int, int],
        dataflow: DataflowType,
    ) -> None:
        m, k, n = shape
        r, c = array_shape
        rng = np.random.default_rng(m * 100 + k * 10 + n)
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)

        res = SystolicArraySimulator(
            SystolicConfig(r, c, dataflow, "M-N-K", "bf16", "fp32")
        ).matmul(a, b)
        ref = compute_fp64_reference(a, b, input_dtype="bf16")
        report = evaluate_simulation(res, ref)

        assert res.output.shape == (m, n)
        assert report.rel_l2_error < 2e-6
        assert res.stats.num_macs == m * k * n
        assert 0.0 < report.utilization <= 1.0

    def test_ws_streaming_tile_m_larger_than_array_rows(self) -> None:
        rng = np.random.default_rng(55)
        a = rng.standard_normal((25, 19)).astype(np.float32)
        b = rng.standard_normal((19, 13)).astype(np.float32)

        res = SystolicArraySimulator(
            SystolicConfig(8, 8, "weight_stationary", "M-N-K", "fp16", "fp32", tile_m=16)
        ).matmul(a, b)
        ref = compute_fp64_reference(a, b, input_dtype="fp16")
        assert evaluate_simulation(res, ref).rel_l2_error < 1e-6

    def test_determinism_and_prequantized_equivalence(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.standard_normal((17, 33)).astype(np.float32)
        b = rng.standard_normal((33, 19)).astype(np.float32)

        cfg = SystolicConfig(8, 8, "output_stationary", "K-M-N", "fp8_e4m3", "bf16")
        sim = SystolicArraySimulator(cfg)
        res1 = sim.matmul(a, b)
        res2 = sim.matmul(a, b)
        assert np.array_equal(res1.output, res2.output)
        assert res1.stats == res2.stats

        a_q = quantize_tensor(a, "fp8_e4m3")
        b_q = quantize_tensor(b, "fp8_e4m3")
        cfg_pre = SystolicConfig(
            8, 8, "output_stationary", "K-M-N", "fp8_e4m3", "bf16", quantize_inputs=False
        )
        res3 = SystolicArraySimulator(cfg_pre).matmul(a_q, b_q)
        assert np.array_equal(res1.output, res3.output)

    def test_order_dependent_cancellation_drift(self) -> None:
        # In BF16 sequential accumulation (os_seq_k), chunk 0 sums to 1.0 and gets absorbed
        # into +1024.0 at k=8 before -1024.0 cancels at k=15, yielding 0.0.
        # Chunked schedules (k=8) reduce chunk 0 (1.0) and chunk 1 (0.0) separately, yielding 1.0.
        a = np.zeros((1, 16), dtype=np.float32)
        b = np.ones((16, 1), dtype=np.float32)
        a[0, 0:8] = 0.125
        a[0, 8] = 1024.0
        a[0, 15] = -1024.0

        ref = compute_fp64_reference(a, b, input_dtype="bf16")
        assert ref[0, 0] == pytest.approx(1.0)

        res_seq = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "M-N-K", "bf16", "bf16")
        ).matmul(a, b)
        res_chk = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "K-M-N", "bf16", "bf16")
        ).matmul(a, b)
        res_ws = SystolicArraySimulator(
            SystolicConfig(8, 8, "weight_stationary", "M-N-K", "bf16", "bf16")
        ).matmul(a, b)

        assert res_seq.stats.addition_order_tag == "os_seq_k"
        assert res_chk.stats.addition_order_tag == "os_chunked_k8"
        assert res_ws.stats.addition_order_tag == "ws_chunked_k8"
        assert res_seq.output[0, 0] == pytest.approx(0.0)
        assert res_chk.output[0, 0] == pytest.approx(1.0)
        assert res_ws.output[0, 0] == pytest.approx(1.0)

    def test_cycle_accounting_and_dataflow_reuse_invariants(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.standard_normal((24, 24)).astype(np.float32)
        b = rng.standard_normal((24, 16)).astype(np.float32)

        ws_mnk = SystolicArraySimulator(
            SystolicConfig(8, 8, "weight_stationary", "M-N-K", "bf16", "bf16")
        ).matmul(a, b)
        ws_knm = SystolicArraySimulator(
            SystolicConfig(8, 8, "weight_stationary", "K-N-M", "bf16", "bf16")
        ).matmul(a, b)

        # K-N-M keeps each (k, n) weight tile stationary across all 3 M tiles
        assert ws_knm.stats.weight_loads < ws_mnk.stats.weight_loads
        assert ws_knm.stats.preload_cycles < ws_mnk.stats.preload_cycles
        assert ws_knm.stats.total_cycles == (
            ws_knm.stats.preload_cycles + ws_knm.stats.wavefront_cycles + ws_knm.stats.drain_cycles
        )
        assert np.array_equal(ws_mnk.output, ws_knm.output)

        os_mnk = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "M-N-K", "bf16", "bf16")
        ).matmul(a, b)
        os_kmn = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "K-M-N", "bf16", "bf16")
        ).matmul(a, b)

        # M-N-K keeps output stationary across all 3 K tiles, draining once per (m, n) tile
        assert os_mnk.stats.output_drains < os_kmn.stats.output_drains
        assert os_mnk.stats.drain_cycles < os_kmn.stats.drain_cycles
        assert os_mnk.stats.total_cycles < os_kmn.stats.total_cycles

    def test_precision_hierarchy_and_fma_on_synthetic_workload(self) -> None:
        workloads = generate_synthetic_workloads(seed=42, m_dim=16, k_dim=64, n_dim=48)
        wl = workloads["attn_q_proj"]
        ref = compute_fp64_reference(wl.activations, wl.weights, input_dtype="bf16")

        err_by_acc = {}
        for acc in ("fp32", "fp16", "bf16"):
            res = SystolicArraySimulator(
                SystolicConfig(8, 8, "weight_stationary", "M-N-K", "bf16", acc)
            ).matmul(wl.activations, wl.weights)
            err_by_acc[acc] = evaluate_simulation(res, ref).rel_l2_error

        assert err_by_acc["fp32"] < err_by_acc["fp16"] < err_by_acc["bf16"]

        res_nofma = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "M-N-K", "bf16", "bf16", fma=False)
        ).matmul(wl.activations, wl.weights)
        res_fma = SystolicArraySimulator(
            SystolicConfig(8, 8, "output_stationary", "M-N-K", "bf16", "bf16", fma=True)
        ).matmul(wl.activations, wl.weights)
        assert evaluate_simulation(res_fma, ref).rel_l2_error <= evaluate_simulation(
            res_nofma, ref
        ).rel_l2_error * 1.1

    def test_invalid_configs_and_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            SystolicArraySimulator(SystolicConfig(8, 8, "weight_stationary", "M-N-K", tile_k=16))
        with pytest.raises(ValueError):
            SystolicArraySimulator(SystolicConfig(8, 8, "output_stationary", "M-N-K", tile_m=16))
        with pytest.raises(ValueError):
            SystolicArraySimulator(SystolicConfig(8, 8, "invalid", "M-N-K"))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            SystolicArraySimulator(SystolicConfig(8, 8, "weight_stationary", "N-K-M"))  # type: ignore[arg-type]

        sim = SystolicArraySimulator(SystolicConfig())
        with pytest.raises(ValueError):
            sim.matmul(np.ones((4, 8)), np.ones((7, 4)))
        with pytest.raises(ValueError):
            sim.matmul(np.ones((4, 8, 1)), np.ones((8, 4)))

    def test_synthetic_fallback_loader_determinism(self) -> None:
        w1 = load_workloads(source="synthetic", seed=123, seq_len=12, max_k=32, max_n=24)
        w2 = load_workloads(source="synthetic", seed=123, seq_len=12, max_k=32, max_n=24)
        assert set(w1.keys()) == {"attn_q_proj", "mlp_down_proj"}
        assert w1["attn_q_proj"].shape == (12, 32, 24)
        assert np.array_equal(w1["attn_q_proj"].activations, w2["attn_q_proj"].activations)
        assert np.array_equal(w1["mlp_down_proj"].weights, w2["mlp_down_proj"].weights)
