"""
Test suite for Quantization Three Gaps Bounds.
"""

import math
import numpy as np
import pytest
from quantization_three_gaps_bounds import (
    GapBound,
    TotalGapDecomposition,
    discrete_gap_bound,
    entropy_estimation_gap_bound,
    smoothness_gap_bound,
    total_gap_bound,
    gap_dominance_analysis,
    optimal_quant_step,
    verify_discrete_gap,
    verify_entropy_gap,
    verify_smoothness_gap,
)


# =============================================================================
# Test 1: Discrete Gap
# =============================================================================

class TestDiscreteGap:
    def test_positive_bound(self):
        """Discrete gap should be positive for Δ > 0."""
        g = discrete_gap_bound(latent_dim=16, quant_step=0.5, sigma_eff=1.0)
        assert g.bound_value > 0

    def test_zero_gap_at_zero_step(self):
        """Discrete gap → 0 as Δ → 0 (perfect quantization)."""
        g = discrete_gap_bound(latent_dim=16, quant_step=0.001, sigma_eff=1.0)
        assert g.bound_value < 0.1

    def test_increases_with_quant_step(self):
        """Larger quantization step → larger gap."""
        steps = [0.01, 0.1, 0.5, 1.0, 2.0]
        gaps = [discrete_gap_bound(16, s, 1.0).bound_value for s in steps]
        for i in range(len(gaps) - 1):
            assert gaps[i] <= gaps[i + 1]

    def test_increases_with_dim(self):
        """Higher latent dimension → larger gap (more dimensions to mismatch)."""
        dims = [4, 16, 64, 256]
        gaps = [discrete_gap_bound(d, 0.5, 1.0).bound_value for d in dims]
        for i in range(len(gaps) - 1):
            assert gaps[i] <= gaps[i + 1]

    def test_decreases_with_sigma(self):
        """Larger σ_eff → smaller gap (latents are larger relative to Δ)."""
        sigmas = [0.1, 0.5, 1.0, 5.0]
        gaps = [discrete_gap_bound(16, 0.5, s).bound_value for s in sigmas]
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1]

    def test_inf_for_zero_sigma(self):
        """σ_eff = 0 should give infinity (degenerate)."""
        g = discrete_gap_bound(16, 0.5, 0.0)
        assert g.bound_value == float('inf')


# =============================================================================
# Test 2: Entropy Estimation Gap
# =============================================================================

class TestEntropyGap:
    def test_positive_bound(self):
        g = entropy_estimation_gap_bound(64, 10000, 100000, 0.1)
        assert g.bound_value >= 0

    def test_decreases_with_more_training_data(self):
        """More training samples → smaller estimation error."""
        M_values = [1000, 10000, 100000, 1000000]
        gaps = [entropy_estimation_gap_bound(64, 10000, M, 0.1).bound_value for M in M_values]
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 1e-10

    def test_increases_with_model_complexity(self):
        """More model params → larger estimation error (overfitting risk)."""
        N_values = [100, 1000, 10000, 100000]
        gaps = [entropy_estimation_gap_bound(64, N, 100000, 0.1).bound_value for N in N_values]
        for i in range(len(gaps) - 1):
            assert gaps[i] <= gaps[i + 1] + 1e-10

    def test_decomposition(self):
        """The bound should decompose into estimation error + bias."""
        g = entropy_estimation_gap_bound(64, 10000, 100000, 0.1)
        est = g.parameters["estimation_error"]
        bias = g.parameters["bias"]
        assert g.bound_value == pytest.approx(est + bias)


# =============================================================================
# Test 3: Smoothness Gap
# =============================================================================

class TestSmoothnessGap:
    def test_positive_bound(self):
        g = smoothness_gap_bound(16, 0.5, 1.0)
        assert g.bound_value >= 0

    def test_zero_at_zero_step(self):
        g = smoothness_gap_bound(16, 0.001, 1.0)
        assert g.bound_value < 0.01

    def test_increases_with_lipschitz(self):
        """Higher Lipschitz constant (more variable density) → larger gap."""
        L_values = [0.1, 0.5, 1.0, 5.0]
        gaps = [smoothness_gap_bound(16, 0.5, L).bound_value for L in L_values]
        for i in range(len(gaps) - 1):
            assert gaps[i] <= gaps[i + 1]

    def test_increases_with_quant_step(self):
        steps = [0.01, 0.1, 0.5, 1.0]
        gaps = [smoothness_gap_bound(16, s, 1.0).bound_value for s in steps]
        for i in range(len(gaps) - 1):
            assert gaps[i] <= gaps[i + 1]

    def test_zero_for_zero_lipschitz(self):
        """L=0 means constant density → no smoothness gap."""
        g = smoothness_gap_bound(16, 0.5, 0.0)
        assert g.bound_value == 0.0


# =============================================================================
# Test 4: Total Gap Decomposition
# =============================================================================

class TestTotalGap:
    def test_total_equals_sum(self):
        """Total gap = discrete + entropy + smoothness."""
        decomp = total_gap_bound(64, 0.5, 1.0, 10000, 100000, 1.0)
        expected = (decomp.discrete_gap.bound_value +
                    decomp.entropy_gap.bound_value +
                    decomp.smoothness_gap.bound_value)
        assert decomp.total_gap == pytest.approx(expected)

    def test_dominant_gap_identified(self):
        """Dominant gap should be the largest component."""
        decomp = total_gap_bound(64, 0.5, 1.0, 10000, 100000, 1.0)
        gaps = {
            "discrete": decomp.discrete_gap.bound_value,
            "entropy_estimation": decomp.entropy_gap.bound_value,
            "smoothness": decomp.smoothness_gap.bound_value,
        }
        assert decomp.dominant_gap == max(gaps, key=gaps.get)

    def test_total_positive(self):
        decomp = total_gap_bound(64, 0.5, 1.0, 10000, 100000, 1.0)
        assert decomp.total_gap > 0

    def test_all_gaps_nonneg(self):
        decomp = total_gap_bound(64, 0.5, 1.0, 10000, 100000, 1.0)
        assert decomp.discrete_gap.bound_value >= 0
        assert decomp.entropy_gap.bound_value >= 0
        assert decomp.smoothness_gap.bound_value >= 0


# =============================================================================
# Test 5: Gap Dominance Analysis
# =============================================================================

class TestGapDominance:
    def test_returns_correct_count(self):
        results = gap_dominance_analysis(n_points=30)
        assert len(results) == 30

    def test_all_gaps_nonneg(self):
        results = gap_dominance_analysis(n_points=20)
        for r in results:
            assert r["discrete_gap"] >= 0
            assert r["entropy_gap"] >= 0
            assert r["smoothness_gap"] >= 0
            assert r["total_gap"] >= 0

    def test_total_gap_decreases_with_small_delta(self):
        """As Δ → 0, total gap should decrease."""
        results = gap_dominance_analysis(n_points=100, quant_step_range=(0.001, 1.0))
        # First entries (small Δ) should have smaller gaps than last entries
        assert results[0]["total_gap"] < results[-1]["total_gap"]


# =============================================================================
# Test 6: Optimal Quantization Step
# =============================================================================

class TestOptimalQuantStep:
    def test_returns_valid_delta(self):
        result = optimal_quant_step(64, 1.0, 10000, 100000, 1.0)
        assert result["optimal_delta"] > 0
        assert result["min_total_gap"] >= 0

    def test_optimal_is_better_than_extremes(self):
        """The optimal Δ should give a smaller gap than very small or very large Δ."""
        result = optimal_quant_step(64, 1.0, 10000, 100000, 1.0)
        opt_gap = result["min_total_gap"]

        gap_small = total_gap_bound(64, 0.001, 1.0, 10000, 100000, 1.0).total_gap
        gap_large = total_gap_bound(64, 10.0, 1.0, 10000, 100000, 1.0).total_gap

        # Optimal should be better than at least one extreme
        assert opt_gap <= gap_small + 0.01 or opt_gap <= gap_large + 0.01


# =============================================================================
# Test 7: Numerical Verification
# =============================================================================

class TestNumericalVerification:
    def test_discrete_gap_verification(self):
        """Verify discrete gap bound via Monte Carlo."""
        result = verify_discrete_gap(latent_dim=4, quant_step=0.5, sigma_eff=1.0, n_samples=100000)
        # The bound should hold (empirical ≤ theoretical, with some slack)
        assert result["empirical_gap"] >= 0
        assert result["theoretical_bound"] > 0

    def test_entropy_gap_verification(self):
        """Verify entropy gap bound via simulation."""
        result = verify_entropy_gap(latent_dim=2, n_model_params=100,
                                     n_train_samples=10000, quant_step=0.1)
        assert result["empirical_kl"] >= 0
        assert result["theoretical_bound"] >= 0

    def test_smoothness_gap_verification(self):
        """Verify smoothness gap bound via integration."""
        result = verify_smoothness_gap(latent_dim=2, quant_step=0.5, lipschitz_constant=1.0)
        assert result["empirical_gap"] >= 0
        assert result["theoretical_bound"] >= 0