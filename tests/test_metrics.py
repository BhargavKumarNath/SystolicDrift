"""Unit and edge-case tests for numerical drift metrics."""

import numpy as np
import pytest

from systolic_drift import (
    SystolicArraySimulator,
    SystolicConfig,
    compute_fp64_reference,
    cosine_similarity,
    evaluate_simulation,
    max_abs_error,
    mean_abs_error,
    relative_l2_error,
    rmse,
    top_k_percent_outlier_error,
)


def test_exact_match_metrics() -> None:
    ref = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float64)
    pred = ref.copy()

    assert cosine_similarity(pred, ref) == 1.0
    assert relative_l2_error(pred, ref) == 0.0
    assert max_abs_error(pred, ref) == 0.0
    assert mean_abs_error(pred, ref) == 0.0
    assert rmse(pred, ref) == 0.0

    top = top_k_percent_outlier_error(pred, ref, top_fraction=0.01)
    assert top["top_rel_l2"] == 0.0
    assert top["top_max_abs"] == 0.0
    assert top["top_count"] == 1


def test_analytical_orthogonal_opposite_and_scaled() -> None:
    r = np.array([[3.0, 0.0]], dtype=np.float64)
    p_ortho = np.array([[0.0, 4.0]], dtype=np.float64)
    assert cosine_similarity(p_ortho, r) == pytest.approx(0.0)
    assert relative_l2_error(p_ortho, r) == pytest.approx(5.0 / 3.0)

    p_opp = -r
    assert cosine_similarity(p_opp, r) == pytest.approx(-1.0)
    assert relative_l2_error(p_opp, r) == pytest.approx(2.0)

    p_scaled = 2.0 * r
    assert cosine_similarity(p_scaled, r) == pytest.approx(1.0)
    assert relative_l2_error(p_scaled, r) == pytest.approx(1.0)


def test_top_1_percent_outlier_isolation() -> None:
    ref = np.ones((10, 10), dtype=np.float64)
    ref[4, 7] = 100.0
    pred = ref.copy()

    # Error only on a non-outlier element does not affect top-1% error
    pred[0, 0] += 5.0
    top_clean = top_k_percent_outlier_error(pred, ref, top_fraction=0.01)
    assert top_clean["top_count"] == 1
    assert top_clean["top_max_abs"] == 0.0
    assert top_clean["top_rel_l2"] == 0.0

    # Error on the single largest element is captured strictly
    pred[4, 7] -= 2.5
    top_dirty = top_k_percent_outlier_error(pred, ref, top_fraction=0.01)
    assert top_dirty["top_count"] == 1
    assert top_dirty["top_max_abs"] == pytest.approx(2.5)
    assert top_dirty["top_rel_l2"] == pytest.approx(2.5 / 100.0)


def test_zero_and_non_finite_edge_cases() -> None:
    zeros = np.zeros((4, 4), dtype=np.float64)
    ones = np.ones((4, 4), dtype=np.float64)

    assert cosine_similarity(zeros, zeros) == 1.0
    assert relative_l2_error(zeros, zeros) == 0.0
    assert cosine_similarity(ones, zeros) == 0.0
    assert np.isinf(relative_l2_error(ones, zeros))

    with_inf = ones.copy()
    with_inf[1, 2] = np.inf
    assert np.isnan(cosine_similarity(with_inf, ones))
    assert np.isnan(relative_l2_error(with_inf, ones))
    assert np.isnan(max_abs_error(with_inf, ones))
    assert np.isnan(top_k_percent_outlier_error(with_inf, ones)["top_rel_l2"])


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        relative_l2_error(np.ones((0,)), np.ones((0,)))
    with pytest.raises(ValueError):
        top_k_percent_outlier_error(np.ones((2, 2)), np.ones((2, 2)), top_fraction=0.0)
    with pytest.raises(ValueError):
        top_k_percent_outlier_error(np.ones((2, 2)), np.ones((2, 2)), top_fraction=1.5)


def test_evaluate_simulation_serialization() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((12, 16)).astype(np.float32)
    b = rng.standard_normal((16, 10)).astype(np.float32)

    res = SystolicArraySimulator(
        SystolicConfig(8, 8, "weight_stationary", "M-N-K", "bf16", "fp16")
    ).matmul(a, b)
    ref = compute_fp64_reference(a, b, input_dtype="bf16")
    report = evaluate_simulation(res, ref)
    as_dict = report.to_dict()

    assert as_dict["non_finite_count"] == 0
    assert as_dict["num_macs"] == 12 * 16 * 10
    assert as_dict["addition_order_tag"] == "ws_chunked_k8"
    assert 0.99 < as_dict["cosine_sim"] <= 1.0
